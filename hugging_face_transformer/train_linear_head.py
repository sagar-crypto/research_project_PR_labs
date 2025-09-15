# train_hf_head.py
import sys, os, json, time, platform, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from hugging_face_transformer.data_processing import ParquetTimeSeriesDataset
from hugging_face_transformer.linear_head import HFWithWindowHead

from sklearn.metrics import precision_recall_curve

# ---- config/env ----
from config import DATA_PATH, HUGGING_FACE_TRANSFORMERS_DIR, CHECKPOINT_HUGGING_FACE_DIR

def load_config(path): return json.load(open(path, "r"))
def hb(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_events_map(csv_path: str):
    """CSV -> {file_stem: [(start_sec, end_sec), ...]}."""
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)

    # --- key column ---
    key_col = None
    for c in ["replica_id", "file_stem", "stem", "id", "file", "filename", "path"]:
        if c in df.columns:
            key_col = c; break
    if key_col is None: return {}

    # normalize to stem
    def to_stem(v): return Path(str(v)).stem
    df["_key"] = df[key_col].apply(to_stem)

    # --- to seconds ---
    if {"start_ms","end_ms"}.issubset(df.columns):
        df["_s"] = df["start_ms"].astype(float) / 1000.0
        df["_e"] = df["end_ms"].astype(float) / 1000.0
    elif {"start_sec","end_sec"}.issubset(df.columns):
        df["_s"] = df["start_sec"].astype(float)
        df["_e"] = df["end_sec"].astype(float)
    elif {"start","end"}.issubset(df.columns):
        df["_s"] = df["start"].astype(float)
        df["_e"] = df["end"].astype(float)
    elif {"start_s","end_s"}.issubset(df.columns):
        df["_s"] = df["start_s"].astype(float)
        df["_e"] = df["end_s"].astype(float)
    else:
        return {}

    m = {}
    for k, g in df.groupby("_key"):
        m[str(k)] = [(float(a), float(b)) for a, b in zip(g["_s"], g["_e"])]
    return m

def robust_load_base_from_ckpt(wrapper: HFWithWindowHead, ckpt_path: Path, device):
    hb(f"loading base weights from {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    new_sd = {}
    if any(k.startswith("base.") for k in state.keys()):
        for k, v in state.items():
            if k.startswith("base."): new_sd[k[len("base."):]] = v
    else:
        for k, v in state.items():
            if k.startswith("model."): new_sd[k] = v
    missing, unexpected = wrapper.base.load_state_dict(new_sd, strict=False)
    if missing:   hb(f"load warning: missing {len(missing)} keys (ok if head-only)")
    if unexpected:hb(f"load warning: unexpected {len(unexpected)} keys (ignored)")
    hb("base weights loaded")

def build_sampler(labels, pos_boost=1.0):
    # inverse frequency weights; optionally boost positives
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts   = torch.bincount(labels_t, minlength=2).float()
    neg, pos = float(counts[0].item()), float(counts[1].item())
    w_neg, w_pos = 1.0, (neg / max(1.0, pos)) * pos_boost
    weights = torch.where(labels_t == 1, torch.tensor(w_pos), torch.tensor(w_neg)).float()
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True), neg, pos

def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for ctx, _, _, y in loader:
            B, L, _ = ctx.shape
            ctx = ctx.to(device)
            y   = y.to(device).view(-1)
            ptf = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1) / float(max(1, L))
            ptf = ptf.expand(B, L, 1)
            # simple ftf ramp (model.forward expects it even if we only care about enc states)
            pred_len = loader.dataset.dataset.pred_len if hasattr(loader.dataset, "dataset") else loader.dataset.pred_len
            ftf = torch.arange(L, L + pred_len, device=device, dtype=torch.float32).view(1, pred_len, 1) / float(max(1, L))
            ftf = ftf.expand(B, pred_len, 1)
            logit = model(
                past_values=ctx,
                past_observed_mask=torch.ones_like(ctx, dtype=torch.bool, device=device),
                past_time_features=ptf,
                future_time_features=ftf,
            ).view(-1)
            all_logits.append(logit.cpu())
            all_y.append(y.cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy().astype(int)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hb(f"start | host={platform.node()} torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")

    # ---- knobs (env) ----
    RUN_NAME        = os.getenv("RUN_NAME", "tst_run")
    HEAD_EPOCHS     = int(os.getenv("HEAD_EPOCHS", "15"))
    BATCH_SIZE      = int(os.getenv("HEAD_BSZ", "64"))
    LR_HEAD         = float(os.getenv("HEAD_LR", "1e-3"))
    HEAD_POOL       = os.getenv("HEAD_POOL", "max")  # "mean" or "max"
    LABEL_SCOPE     = os.getenv("LABEL_SCOPE", "context")  # IMPORTANT: match encoder-only view
    LABELS_CSV      = os.getenv("LABELS_CSV", "/home/vault/iwi5/iwi5305h/new_dataset_90kv/labels_for_parquet.csv")
    VAL_FRACTION    = float(os.getenv("VAL_FRAC", "0.2"))
    POS_BOOST       = float(os.getenv("POS_BOOST", "1.0"))  # >1.0 to oversample positives more
    UNFREEZE_LAST   = int(os.getenv("UNFREEZE_LAST", "0"))  # 1 to unfreeze last encoder block
    ENC_LR          = float(os.getenv("ENC_LR", "1e-4"))

    events_map = load_events_map(LABELS_CSV)

    ckpt_dir = Path(os.path.expandvars(CHECKPOINT_HUGGING_FACE_DIR)).expanduser() / RUN_NAME
    last_ckpt = ckpt_dir / "last.pt"
    if not last_ckpt.exists():
        raise FileNotFoundError(f"No backbone checkpoint found at {last_ckpt}")

    # ---- dataset ----
    cfg = load_config(f"{HUGGING_FACE_TRANSFORMERS_DIR}/hyper_parameter.json")
    ds_full = ParquetTimeSeriesDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=cfg["sample_rate"],
        window_ms=cfg["window_ms"],
        pred_ms=cfg["pred_ms"],
        stride_ms=cfg["stride_ms"],
        feature_range=(cfg.get("feature_min",0.0), cfg.get("feature_max",1.0)),
        events_map=events_map,
        label_scope=LABEL_SCOPE,
    )

    # split train/val
    n_total = len(ds_full)
    n_val   = max(1, int(round(VAL_FRACTION * n_total)))
    n_train = max(1, n_total - n_val)
    ds_train, ds_val = random_split(ds_full, [n_train, n_val])

    # label arrays for sampler/pos_weight on TRAIN only
    train_labels = []
    for i in range(len(ds_train)):
        _, _, _, y = ds_train[i]
        train_labels.append(int(y.item() >= 0.5))
    sampler, neg, pos = build_sampler(train_labels, pos_boost=POS_BOOST)

    # BCEWithLogitsLoss(pos_weight)
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32).to(device)

    # loaders
    train_loader = DataLoader(
        ds_train, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
        drop_last=True, num_workers=0, pin_memory=(device.type=="cuda")
    )
    val_loader = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=(device.type=="cuda")
    )

    hb(f"[data] train={len(ds_train)} val={len(ds_val)} | pos-rate train={sum(train_labels)/max(1,len(train_labels)):.4f}")

    # ---- base + head ----
    hf_cfg = TimeSeriesTransformerConfig(
        input_size=ds_full.n_features,
        context_length=ds_full.seq_len,
        prediction_length=ds_full.pred_len,
        lags_sequence=[0],
        distribution_output="normal",      # irrelevant for head-only
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
    model = HFWithWindowHead(base, d_model=cfg["d_model"], pool=HEAD_POOL).to(device)

    robust_load_base_from_ckpt(model, last_ckpt, device=device)

    # freeze all backbone
    for p in model.base.parameters():
        p.requires_grad = False

    # optional: unfreeze last encoder block (small LR)
    param_groups = [{"params": model.head.parameters(), "lr": LR_HEAD, "weight_decay": 1e-4}]
    if UNFREEZE_LAST:
        try:
            last_blk = model.base.model.encoder.layers[-1]
            for p in last_blk.parameters(): p.requires_grad = True
            param_groups.append({"params": last_blk.parameters(), "lr": ENC_LR, "weight_decay": 1e-4})
            hb("🔓 unfroze last encoder block")
        except Exception as e:
            hb(f"could not unfreeze last encoder block: {e}")

    optimizer = optim.Adam(param_groups)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {"f1": -1.0, "thr": 0.5}

    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        tp = fp = tn = fn = 0

        for batch in train_loader:
            if len(batch) != 4:
                raise RuntimeError("Dataset must return (ctx, tgt, times, y) for head training.")
            ctx, _, _, y = batch
            B, L, _ = ctx.shape

            ctx = ctx.to(device, non_blocking=True)
            y   = y.to(device, non_blocking=True).view(-1)

            # normalized time ramps (same device)
            ptf = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1) / float(max(1, L))
            ptf = ptf.expand(B, L, 1)
            pred_len = ds_full.pred_len
            ftf = torch.arange(L, L + pred_len, device=device, dtype=torch.float32).view(1, pred_len, 1) / float(max(1, L))
            ftf = ftf.expand(B, pred_len, 1)

            optimizer.zero_grad(set_to_none=True)
            logit = model(
                past_values=ctx,
                past_observed_mask=torch.ones_like(ctx, dtype=torch.bool, device=device),
                past_time_features=ptf,
                future_time_features=ftf,
            ).view(-1)
            loss  = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss) * B

            # quick running metrics @ 0.5
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

        # ---- validate & choose threshold ----
        logits_val, y_val = collect_logits(model, val_loader, device)
        probs_val = 1.0 / (1.0 + np.exp(-logits_val))
        P, R, T   = precision_recall_curve(y_val, probs_val)
        F1        = 2 * P * R / (P + R + 1e-9)
        if len(T) > 0:
            i = int(np.nanargmax(F1[:-1]))  # last P,R point has no threshold
            thr = float(T[i])
            f1b = float(F1[i])
            if f1b > best["f1"]:
                best["f1"] = f1b
                best["thr"] = thr
        else:
            thr = 0.5

        # report val metrics at both 0.5 and tuned thr
        def conf_at(th):
            pred = (probs_val >= th).astype(np.int64)
            tp = int(((pred == 1) & (y_val == 1)).sum())
            tn = int(((pred == 0) & (y_val == 0)).sum())
            fp = int(((pred == 1) & (y_val == 0)).sum())
            fn = int(((pred == 0) & (y_val == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            f1   = 2 * prec * rec / max(1e-12, (prec + rec))
            return tp, fp, tn, fn, prec, rec, f1

        tp05, fp05, tn05, fn05, p05, r05, f105 = conf_at(0.5)
        tpb, fpb, tnb, fnb, pb, rb, f1b = conf_at(best["thr"])

        print(
            f"[head] epoch {epoch:02d} | loss/train={total_loss/len(ds_train):.6f} | "
            f"train@0.5 P={precision:.4f} R={recall:.4f} | "
            f"val@0.5 TP={tp05} FP={fp05} TN={tn05} FN={fn05} | P={p05:.4f} R={r05:.4f} F1={f105:.4f} | "
            f"val@τ*={best['thr']:.3f} → P={pb:.4f} R={rb:.4f} F1={f1b:.4f}"
        )

    # save head weights + chosen threshold
    out_head = ckpt_dir / "head_only.pth"
    torch.save({"head_state": model.head.state_dict(), "best_thr": best["thr"]}, out_head)
    hb(f"saved head to {out_head} (τ*={best['thr']:.4f})")

if __name__ == "__main__":
    main()
