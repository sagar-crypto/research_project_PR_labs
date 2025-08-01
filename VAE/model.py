import torch
from typing import List, Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from constants import (
    BN_EPS, BN_MOMENTUM,
    USE_DROPOUT, DROPOUT_RATE,
    ENCODER_BLOCKS, ADAPT_POOL_OUTPUT_SIZE,
    DECODER_BLOCKS, FINAL_KERNEL, FINAL_PADDING,
)


class Conv_block(nn.Module):
    def __init__(
        self,
        in_ch, out_ch,
        kernel, pad, stride=1,
        is_conv=True,
        use_dropout=USE_DROPOUT,
        dropout_rate=DROPOUT_RATE
    ) -> None:
        super().__init__()
        self.pool_op = (
            nn.AvgPool1d(2) if is_conv
            else nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        self.conv = nn.Conv1d(in_ch, out_ch,
                              kernel_size=kernel,
                              padding=pad,
                              stride=stride)
        self.bn = nn.BatchNorm1d(out_ch,
                                 eps=BN_EPS,
                                 momentum=BN_MOMENTUM)
        self.act = nn.GELU()
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        return self.pool_op(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_size: int) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_channels,
                                  eps=BN_EPS,
                                  momentum=BN_MOMENTUM)

        # build conv‐blocks from constants
        ch_in = in_channels
        self.blocks = nn.ModuleList()
        for spec in ENCODER_BLOCKS:
            blk = Conv_block(
                in_ch=ch_in,
                out_ch=spec["out_ch"],
                kernel=spec["kernel"],
                pad=spec["pad"],
                stride=spec["stride"],
            )
            self.blocks.append(blk)
            ch_in = spec["out_ch"]

        # adaptive pool to fixed size
        self.adapt_pool = nn.AdaptiveAvgPool1d(ADAPT_POOL_OUTPUT_SIZE)
        flat_dim = ADAPT_POOL_OUTPUT_SIZE * ch_in

        self.encode_mean   = nn.Linear(flat_dim, latent_size)
        self.encode_logvar = nn.Linear(flat_dim, latent_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        x = x.permute(0,2,1)      # (B, in_ch, L)
        x = self.bn0(x)
        skips : List[Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            skips.append(x)
        # x4 = skips[-1]
        x_pooled = self.adapt_pool(skips[-1])
        flat     = x_pooled.flatten(1)
        mean     = self.encode_mean(flat)
        logvar   = self.encode_logvar(flat)
        return mean, logvar, skips



class Decoder(nn.Module):
    def __init__(
        self,
        length: int,
        in_channels: int,
        latent_size: int,
        out_channels: int
    ) -> None:
        super().__init__()
        # store for final conv / length
        self.length = length

        self.adapt_nn = nn.Linear(latent_size, in_channels)
        self.relu     = nn.ReLU()
        self.tanh     = nn.Tanh()

        enc_chs = [spec["out_ch"] for spec in ENCODER_BLOCKS][::-1]

        # mirror ENCODER_BLOCKS in reverse with upsampling
        self.blocks = nn.ModuleList()
        ch_in = in_channels   # because we concat skip
        for idx, spec in enumerate(DECODER_BLOCKS):
            skip_ch = enc_chs[idx]
            in_ch   = ch_in + skip_ch
            blk = Conv_block(
                in_ch=in_ch,
                out_ch=spec["out_ch"],
                kernel=spec["kernel"],
                pad=spec["pad"],
                is_conv=False,
            )
            self.blocks.append(blk)
            ch_in = spec["out_ch"]

        # final conv to out_channels
        self.decode_conv = nn.Conv1d(
            ch_in,
            out_channels,
            kernel_size=FINAL_KERNEL,
            padding=FINAL_PADDING,
        )

    def forward(self, z: Tensor, skip_connections: List[Tensor]) -> Tensor:
        x = self.relu(self.adapt_nn(z)).unsqueeze(-1)
        # up to skip length
        target_len = skip_connections[-1].size(2)
        x = F.interpolate(x,
                          size=target_len,
                          mode='linear',
                          align_corners=True)

        for i, blk in enumerate(self.blocks):
            enc_feat = skip_connections[-(i+1)]
            x = F.interpolate(
                x,
                size=enc_feat.size(2),
                mode="linear",
                align_corners=True
            )
            x = torch.cat([x, enc_feat], dim=1)
            x = blk(x)

        x = F.interpolate(x,
                          size=self.length,
                          mode="linear",
                          align_corners=True)

        x = self.decode_conv(x)
        return self.tanh(x)



class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        length: int,
        latent_size: int = 16,
        encoder_out_channels: int = 128
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels,
                               latent_size=latent_size)
        self.decoder = Decoder(length,
                               in_channels=encoder_out_channels,
                               latent_size=latent_size, out_channels=in_channels)

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        logvar = torch.clamp(logvar, min=-10, max=10)
        std    = torch.exp(0.5 * logvar).clamp(1e-5, 10.0)
        eps    = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # x: (B, L, C) → feed to encoder
        mean, logvar, skip_connections = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z, skip_connections)
        return recon, mean, logvar

def initialize_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)




