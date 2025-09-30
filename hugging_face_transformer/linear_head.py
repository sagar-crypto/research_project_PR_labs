import torch
import torch.nn as nn

class HFWithWindowHead(nn.Module):
    """
    Pool encoder hidden states from Hugging Face TimeSeriesTransformer and classify the window.
    Works across HF versions by trying both encoder signatures; falls back to full model if needed.
    """
    def __init__(self, base_model, d_model: int, pool: str = "mean"):
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

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, d_model)
        x = h.transpose(1, 2)        # (B, d_model, T)
        x = self.pool(x).squeeze(-1) # (B, d_model)
        return x

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor = None,   # ignored; kept for API compatibility
        **_,
    ):
        enc = self.base.model.encoder

        # Try HF ≥/≤ variants without relying on names
        try:
            # Variant A: values/observed_mask/time_features
            enc_out = enc(
                values=past_values,
                observed_mask=past_observed_mask,
                time_features=past_time_features,
            )
            h = enc_out[0]  # (B, ctx_len, d_model)
            return self.head(self._pool(h))
        except TypeError:
            pass

        try:
            # Variant B: past_values/past_observed_mask/past_time_features
            enc_out = enc(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                past_time_features=past_time_features,
            )
            h = enc_out[0]  # (B, ctx_len, d_model)
            return self.head(self._pool(h))
        except TypeError:
            pass

        # Last-resort fallback: call full model and use encoder_hidden_states
        # (still encoder-based; we won’t touch decoder outputs)
        if future_time_features is None:
            # build a trivial 1-step future ramp on the same device
            B, L, _ = past_values.shape
            future_time_features = past_time_features[:, -1:, :]

        out = self.base(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            output_hidden_states=True,
            return_dict=True,
        )

        if getattr(out, "encoder_hidden_states", None) is not None:
            h = out.encoder_hidden_states[-1]  # (B, ctx_len, d_model)
        else:
            # Very old versions: fall back to last encoder state field
            h = out.encoder_last_hidden_state  # (B, ctx_len, d_model)

        return self.head(self._pool(h))
