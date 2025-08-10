import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction
)
from hugging_face_transformer.data_processing import ParquetTimeSeriesDataset
from config import DATA_PATH, CHECKPOINT_TRANSFORMERS_DIR, TRANSFORMERS_DIR, CLUSTERING_TRANSFORMERS_DIR, HUGGING_FACE_TRANSFORMERS_DIR


def load_config(path):
    return json.load(open(path, 'r'))

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ParquetTimeSeriesDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=cfg["sample_rate"],
        window_ms=cfg["window_ms"],
        pred_ms=cfg["pred_ms"],
        stride_ms=cfg["stride_ms"],
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    hf_cfg = TimeSeriesTransformerConfig(
        input_size=ds.n_features,                 # e.g., 144
        context_length=ds.seq_len,
        prediction_length=ds.pred_len,
        lags_sequence=[0],
        num_time_features=1,                      # time features very important
        num_dynamic_real_features=0,
        d_model=cfg["d_model"],
        encoder_attention_heads=cfg["nhead"],
        decoder_attention_heads=cfg["nhead"],
        encoder_layers=cfg["num_encoder_layers"],
        decoder_layers=cfg["num_decoder_layers"],
        encoder_ffn_dim=cfg["dim_feedforward"],
        decoder_ffn_dim=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    model = TimeSeriesTransformerForPrediction(hf_cfg).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # guard rails (first batch only)
    printed = False

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for batch_data in loader:
            if len(batch_data) == 3:
                ctx, tgt, _times = batch_data
            else:
                ctx, tgt = batch_data

            B, L, D = ctx.shape
            assert D == ds.n_features, f"ctx last dim {D} != ds.n_features {ds.n_features}"

            ctx = ctx.to(device)
            tgt = tgt.to(device)
            dt = 1.0 / cfg["sample_rate"]
            past_time_features = _times.to(device)

            future_time_features = _times[:, -1:, :].to(device) + dt * torch.arange(
                                    1, ds.pred_len + 1, device=device
                                ).view(1, -1, 1)

            batch = {
                "past_values": ctx,
                "past_observed_mask": torch.ones_like(ctx, device=device),
                "future_values": tgt,
                "future_observed_mask": torch.ones_like(tgt, device=device),
                "past_time_features": past_time_features,       # (B, L, 1)
                "future_time_features": future_time_features,   # (B, pred_len, 1)
            }

            if not printed:
                print("past_values", batch["past_values"].shape)     # (B, L, 144)
                print("future_values", batch["future_values"].shape) # (B, pred, 144)
                printed = True

            optimizer.zero_grad()
            out = model(**batch)      # (B, pred_len, 144)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total += loss.item() * B

        print(f"Epoch {epoch:02d}: loss={total/len(ds):.6f}")

if __name__ == "__main__":
    cfg = load_config(f"{HUGGING_FACE_TRANSFORMERS_DIR}/hyper_parameter.json")
    main(cfg)
