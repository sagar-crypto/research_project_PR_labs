import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds a fixed positional embedding to the input.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p= dropout)

        #Precompute the positional encoding once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding, same shape
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """
    Prediction-based Transformer autoencoder with a single linear head.
    Designed for forecasting the next `pred_len` steps from the first
    `window_len - pred_len` inputs of each sliding window.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        window_len: int = 20,
        pred_len: int = 10,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.window_len = window_len
        self.pred_len = pred_len

        self.src_embed = nn.Linear(d_in, d_model)
        self.tgt_embed = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.output_head = nn.Linear(d_model, d_in)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Xavier-uniform init for all weight tensors (dims > 1)
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Creates a causal mask of shape (sz, sz) with -inf on upper triangle.
        This prevents the decoder from attending to future positions.
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src:       (batch_size, window_len - pred_len, d_in)
            tgt_input: (batch_size, pred_len,          d_in)
                       (for teacher forcing; typically the true next pred_len steps shifted right)
        Returns:
            forecast:  (batch_size, pred_len, d_in)
        """
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        memory  = self.encoder(src_emb)

        tgt_emb = self.tgt_embed(tgt_input) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1)).to(src.device)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )

        return self.output_head(decoded)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Runs the encoder part only.
        Args:  src: (B, L_ctx, d_in)
        Returns: encoder memory H: (B, L_ctx, d_model)
        """
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        H = self.encoder(x)       # (B, L, d_model)
        return H
    

class FaultClassifier(nn.Module):
    """
    Pools encoder features from a pretrained TransformerAutoencoder
    and predicts fault / no-fault.
    """
    def __init__(self, backbone: TransformerAutoencoder, dropout: float = 0.1, pool: str = "mean"):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.d_model, 1)  # binary

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        H = self.backbone.encode(src)               # (B, L, d_model)
        z = H.mean(dim=1) if self.pool == "mean" else H[:, -1, :]
        z = self.drop(z)
        return self.head(z).squeeze(-1)             # (B,)

