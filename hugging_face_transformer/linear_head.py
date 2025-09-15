# hugging_face_transformer/linear_head.py
import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerForPrediction

class HFWithWindowHead(nn.Module):
    """
    Pool hidden states from the Hugging Face TimeSeriesTransformer and classify the window.
    We call the full model (not encoder.forward) to get hidden states.
    """
    def __init__(self, base_model: TimeSeriesTransformerForPrediction, d_model: int, pool: str = "mean"):
        super().__init__()
        self.base = base_model

        pool = (pool or "mean").lower()
        if pool == "mean":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pool='{pool}', use 'mean' or 'max'.")

        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        **_
    ):
        # We only need hidden states for the head; this avoids probabilistic-loss issues.
        out = self.base(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            output_hidden_states=True,
            return_dict=True,
        )

        # Prefer decoder hidden states (aligned with the prediction horizon).
        # Fallback to encoder if decoder_hidden_states is None.
        if getattr(out, "decoder_hidden_states", None):
            h = out.decoder_hidden_states[-1]      # (B, pred_len, d_model)
        elif getattr(out, "encoder_hidden_states", None):
            h = out.encoder_hidden_states[-1]      # (B, ctx_len, d_model)
        else:
            # Some versions expose only encoder_last_hidden_state
            h = out.encoder_last_hidden_state      # (B, ctx_len, d_model)

        x = h.transpose(1, 2)                      # (B, d_model, T)
        x = self.pool(x).squeeze(-1)               # (B, d_model)
        logit = self.head(x)                       # (B, 1)
        return logit
