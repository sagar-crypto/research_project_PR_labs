# train_hf_head.py
import sys, os, json, time, platform, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from hugging_face_transformer.data_processing import ParquetTimeSeriesDataset
from hugging_face_transformer.linear_head import HFWithWindowHead

# ---- config/env ----
from config import DATA_PATH, HUGGING_FACE_TRANSFORMERS_DIR, CHECKPOINT_HUGGING_FACE_DIR

def load_config(path): return json.load(open(path, "r"))

def hb(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_events_map(csv_path: str):
    """CSV -> {file_stem: [(start_sec, end_sec), ...]}.
       Accepts columns: 
         - key: one of [replica_id, file_stem, stem, id, file] or a 'path' column
         - time: (start_ms,end_ms) OR (start_sec,end_sec) OR (start,end) in seconds
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)

    # file key
    key_col = None
    for c in ["replica_id", "file_stem", "stem", "id", "file"]:
        if c in df.columns:
            key_col = c; break
    if key_col is None and "path" in df.columns:
        df["replica_id"] = df["path"].apply(lambda p: Path(str(p)).stem)
        key_col = "replica_id"
    if key_col is None:
        return {}

    # time columns → seconds
    if {"start_ms","end_ms"}.issubset(df.columns):
        df["_s"] = df["start_ms"].astype(float) / 1000.0
        df["_e"] = df["end_ms"].astype(float) / 1000.0
    elif {"start_sec","end_sec"}.issubset(df.columns):
        df["_s"] = df["start_sec"].astype(float)
        df["_e"] = df["end_sec"].astype(float)
    elif {"start","end"}.issubset(df.columns):
        df["_s"] = df["start"].astype(float)
        df["_e"] = df["end"].astype(float)
    else:
        return {}

    m = {}
    for k, g in df.groupby(key_col):
        m[str(k)] = [(float(a), float(b)) for a, b in zip(g["_s"], g["_e"])]
    return m

def robust_load_base_from_ckpt(wrapper: HFWithWindowHead, ckpt_path: Path, device):
    hb(f"loading base weights from {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False) if hasattr(torch.load, "__call__") else torch.load(ckpt_path, map_location=device)
    state = payload.get("model", payload)  # support both {model: ...} and raw sd
    # If the checkpoint was saved from the *plain base*, keys start with "model.*"
    # If saved from a wrapper like HFWithWindowHead, keys may start with "base.model.*"
    base_sd = wrapper.base.state_dict()
    new_sd = {}

    if any(k.startswith("base.") for k in state.keys()):
        # strip "base." prefix
        for k, v in state.items():
            if k.startswith("base."):
                k2 = k[len("base."):]
                new_sd[k2] = v
    else:
        # assume keys match base directly or start with "model."
        for k, v in state.items():
            if k.startswith("model."):
                new_sd[k] = v
            else:
                # keys not under model.* are ignored (e.g., head.*)
                pass

    missing, unexpected = wrapper.base.load_state_dict(new_sd, strict=False)
    if missing:
        hb(f"load warning: missing in checkpoint → {len(missing)} keys (ok if head-only)")
    if unexpected:
        hb(f"load warning: unexpected in checkpoint → {len(unexpected)} keys (ignored)")
    hb("base weights loaded")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hb(f"start | host={platform.node()} torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")

    # ---- knobs (env) ----
    RUN_NAME       = os.getenv("RUN_NAME", "tst_run")     # same run name you used for backbone ckpts
    HEAD_EPOCHS    = int(os.getenv("HEAD_EPOCHS", "5"))
    BATCH_SIZE     = int(os.getenv("HEAD_BSZ", "64"))
    LR_HEAD        = float(os.getenv("HEAD_LR", "1e-3"))
    LABEL_SCOPE    = os.getenv("LABEL_SCOPE", "future")   # "window"|"future"|"context"|"center"
    LABELS_CSV     = os.getenv("LABELS_CSV", "/home/vault/iwi5/iwi5305h/new_dataset_90kv/labels_for_parquet.csv")


    events_map  = load_events_map(LABELS_CSV)

    ckpt_dir = Path(os.path.expandvars(CHECKPOINT_HUGGING_FACE_DIR)).expanduser() / RUN_NAME
    last_ckpt = ckpt_dir / "last.pt"
    if not last_ckpt.exists():
        raise FileNotFoundError(f"No backbone checkpoint found at {last_ckpt}")

    # ---- dataset (with labels) ----
    cfg = load_config(f"{HUGGING_FACE_TRANSFORMERS_DIR}/hyper_parameter.json")
    ds = ParquetTimeSeriesDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=cfg["sample_rate"],
        window_ms=cfg["window_ms"],
        pred_ms=cfg["pred_ms"],
        stride_ms=cfg["stride_ms"],
        feature_range=(cfg.get("feature_min",0.0), cfg.get("feature_max",1.0)),
        events_map=events_map,                # you fill this with your label map if not in dataset already
        label_scope=LABEL_SCOPE,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=(device.type=="cuda"))

    # ---- base + head ----
    hf_cfg = TimeSeriesTransformerConfig(
        input_size=ds.n_features,
        context_length=ds.seq_len,
        prediction_length=ds.pred_len,
        lags_sequence=[0],
        distribution_output="normal",     # irrelevant for head-only; we never use HF loss
        num_time_features=1,
        num_dynamic_real_features=0,
        d_model=cfg["d_model"],
        encoder_attention_heads=cfg["nhead"],
        decoder_attention_heads=cfg["nhead"],
        encoder_layers=cfg["num_encoder_layers"],
        decoder_layers=cfg["num_decoder_layers"],
        encoder_ffn_dim=cfg["dim_feedforward"],
        decoder_ffn_dim=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        output_hidden_states=True,
    )
    base  = TimeSeriesTransformerForPrediction(hf_cfg).to(device)
    model = HFWithWindowHead(base, d_model=cfg["d_model"], pool="mean").to(device)

    robust_load_base_from_ckpt(model, last_ckpt, device=device)

    # freeze backbone (already done in the module’s __init__, but be explicit)
    for p in model.base.parameters():
        p.requires_grad = False

    # train the head only
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = optim.Adam(head_params, lr=LR_HEAD, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        tp = fp = tn = fn = 0

        for batch in loader:
            if len(batch) != 4:
                raise RuntimeError("Dataset must return (ctx, tgt, times, y) for head training.")
            ctx, _, _, y = batch
            B, L, _ = ctx.shape

            ctx = ctx.to(device, non_blocking=True)
            y   = y.to(device, non_blocking=True).view(-1)   # (B,)

            # normalized time ramps on the SAME device
            ptf = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1) / float(max(1, L))
            ptf = ptf.expand(B, L, 1)

            pred_len = getattr(ds, "pred_len")  # from your dataset
            ftf = torch.arange(L, L + pred_len, device=device, dtype=torch.float32).view(1, pred_len, 1) / float(max(1, L))
            ftf = ftf.expand(B, pred_len, 1)

            batch_enc = {
                "past_values": ctx,
                "past_observed_mask": torch.ones_like(ctx, dtype=torch.bool, device=device),
                "past_time_features": ptf,
                "future_time_features": ftf,
            }

            optimizer.zero_grad(set_to_none=True)
            logit = model(**batch_enc).view(-1)   # (B,)
            loss  = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss) * B

            # running metrics @ 0.5
            with torch.no_grad():
                prob = torch.sigmoid(logit)
                pred = (prob >= 0.5)
                yb   = (y >= 0.5)
                tp += (pred & yb).sum().item()
                tn += ((~pred) & (~yb)).sum().item()
                fp += (pred & (~yb)).sum().item()
                fn += ((~pred) & yb).sum().item()

        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)
        print(f"[head] epoch {epoch:02d} | loss={total_loss/len(ds):.6f} | "
            f"TP={tp} FP={fp} TN={tn} FN={fn} | P={precision:.4f} R={recall:.4f}")


    # save head weights
    out_head = ckpt_dir / "head_only.pth"
    torch.save({"head_state": model.head.state_dict()}, out_head)
    hb(f"saved head to {out_head}")

if __name__ == "__main__":
    main()
