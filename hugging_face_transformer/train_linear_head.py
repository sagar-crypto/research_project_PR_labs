# train_hf_head.py
import sys, os, json, time, platform, random
from pathlib import Path
from torch import amp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from torch.utils.data import Subset

from hugging_face_transformer.data_processing import ParquetTimeSeriesDataset
from hugging_face_transformer.linear_head import HFWithWindowHead
from sklearn.metrics import precision_recall_curve
from hugging_face_transformer.memmap_dataset import NpyTimeSeriesSimple
from sklearn.model_selection import StratifiedShuffleSplit

# ---- config/env ----
from config import DATA_PATH, HUGGING_FACE_TRANSFORMERS_DIR, CHECKPOINT_HUGGING_FACE_DIR, CACHE_FILE_PATH

def load_config(path): return json.load(open(path, "r"))
def hb(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def _take_one(loader):
    it = iter(loader)
    batch = next(it)
    return batch

# ---- small utils ----
def load_events_map(csv_path: str):
    """CSV -> {file_stem: [(start_sec, end_sec), ...]}."""
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)

    key_col = None
    for c in ["replica_id", "file_stem", "stem", "id", "file", "filename", "path"]:
        if c in df.columns:
            key_col = c; break
    if key_col is None: return {}

    def to_stem(v): return Path(str(v)).stem
    df["_key"] = df[key_col].apply(to_stem)

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

def conf_at(th, probs_val, y_val):
            pred = (probs_val >= th).astype(np.int64)
            tp = int(((pred == 1) & (y_val == 1)).sum())
            tn = int(((pred == 0) & (y_val == 0)).sum())
            fp = int(((pred == 1) & (y_val == 0)).sum())
            fn = int(((pred == 0) & (y_val == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            f1   = 2 * prec * rec / max(1e-12, (prec + rec))
            return tp, fp, tn, fn, prec, rec, f1

def robust_load_base_from_ckpt(wrapper: HFWithWindowHead, ckpt_path: Path, device):
    hb(f"loading base weights from {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload.get("model", payload)

    # normalize keys to match wrapper.base's state_dict
    new_sd = {}
    if any(k.startswith("base.") for k in state.keys()):
        for k, v in state.items():
            if k.startswith("base."):
                new_sd[k[len("base."):]] = v
    else:
        for k, v in state.items():
            if k.startswith("model."):
                new_sd[k] = v
            else:
                new_sd[k] = v  # best-effort

    # >>> minimal fix: keep only params whose shapes match the current model <<<
    base_sd = wrapper.base.state_dict()
    filtered_sd = {
        k: v for k, v in new_sd.items()
        if k in base_sd and hasattr(v, "shape") and base_sd[k].shape == v.shape
    }

    missing, unexpected = wrapper.base.load_state_dict(filtered_sd, strict=False)
    hb(f"base weights loaded | applied={len(filtered_sd)} skipped={len(new_sd) - len(filtered_sd)}")
    if missing:
        hb(f"load warning: missing {len(missing)} keys (ok if head-only / resized inputs)")
    if unexpected:
        hb(f"load warning: unexpected {len(unexpected)} keys (ignored)")


def stratified_file_split(ds_full, val_fraction=0.2):
    # Map each sample index -> file stem
    stems = []
    labels = []
    for i in range(len(ds_full)):
        _, _, _, y = ds_full[i]
        labels.append(int(y.item() >= 0.5))
        file_i = int(np.searchsorted(ds_full.cum_counts, i, side="right") - 1)
        stems.append(Path(ds_full.files[file_i]).stem)
    stems = np.array(stems)
    labels = np.array(labels, dtype=np.int64)

    # Build file-level label: does this file contain ANY positives?
    uniq, inv = np.unique(stems, return_inverse=True)
    file_has_pos = np.zeros(len(uniq), dtype=np.int64)
    np.maximum.at(file_has_pos, inv, labels)  # if any sample is 1 → file label = 1

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    tr_files_idx, va_files_idx = next(sss.split(uniq, file_has_pos))
    tr_files = set(uniq[tr_files_idx]); va_files = set(uniq[va_files_idx])

    tr_idx = [i for i, s in enumerate(stems) if s in tr_files]
    va_idx = [i for i, s in enumerate(stems) if s in va_files]

    from torch.utils.data import Subset
    tr_labels = labels[tr_idx]; va_labels = labels[va_idx]
    return Subset(ds_full, tr_idx), Subset(ds_full, va_idx), tr_labels, va_labels


def stratified_file_split_3way(ds_full, val_frac=0.1, test_frac=0.1, seed=42):
    # Per-sample labels + file stems (same as your 2-way)
    labels, stems = [], []
    for i in range(len(ds_full)):
        _, _, _, y = ds_full[i]
        labels.append(int(y.item() >= 0.5))
        file_i = int(np.searchsorted(ds_full.cum_counts, i, side="right") - 1)
        stems.append(Path(ds_full.files[file_i]).stem)
    labels = np.array(labels, dtype=np.int64)
    stems  = np.array(stems)

    # File-level label: does the file contain ANY positives?
    uniq, inv = np.unique(stems, return_inverse=True)
    file_has_pos = np.zeros(len(uniq), dtype=np.int64)
    np.maximum.at(file_has_pos, inv, labels)

    # 1) Hold out TEST files
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(uniq, file_has_pos))
    trainval_files = set(uniq[trainval_idx]); test_files = set(uniq[test_idx])

    # 2) From remaining, hold out VAL files
    uniq_tv = uniq[trainval_idx]
    file_has_pos_tv = file_has_pos[trainval_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac/(1.0 - test_frac), random_state=seed)
    tr_rel, va_rel = next(sss2.split(uniq_tv, file_has_pos_tv))
    train_files = set(uniq_tv[tr_rel]); val_files = set(uniq_tv[va_rel])

    # Map back to sample indices
    tr_idx = [i for i, s in enumerate(stems) if s in train_files]
    va_idx = [i for i, s in enumerate(stems) if s in val_files]
    te_idx = [i for i, s in enumerate(stems) if s in test_files]

    return Subset(ds_full, tr_idx), Subset(ds_full, va_idx), Subset(ds_full, te_idx), \
           labels[tr_idx], labels[va_idx], labels[te_idx]

def build_sampler(labels, pos_boost=1.0):
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts   = torch.bincount(labels_t, minlength=2).float()
    neg, pos = float(counts[0].item()), float(counts[1].item())
    w_neg, w_pos = 1.0, (neg / max(1.0, pos)) * pos_boost
    weights = torch.where(labels_t == 1, torch.tensor(w_pos), torch.tensor(w_neg)).float()
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True), neg, pos


def profile_input_pipeline(train_loader, model, device, criterion, use_amp, hb_print=print):
    """
    Fetch 1 batch (CPU/IO timing) and run 1 fwd+bwd (GPU timing).
    Prints two lines and returns a dict of timings in ms.
    """

    # A) CPU/IO: time one batch fetch
    t0 = time.perf_counter()
    batch = _take_one(train_loader)
    t1 = time.perf_counter()
    fetch_ms = (t1 - t0) * 1000.0

    # B) GPU: time one forward+backward
    ctx, _, _, y = batch
    ctx = ctx.to(device, non_blocking=True)
    y   = y.to(device, non_blocking=True).view(-1)

    # simple ramp for time features
    L = ctx.shape[1]
    ptf = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1).expand(ctx.shape[0], L, 1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type=="cuda")
    with amp_ctx:
        logit = model(
            past_values=ctx,
            past_observed_mask=torch.ones_like(ctx, dtype=torch.bool, device=device),
            past_time_features=ptf,
        ).view(-1)
        loss = criterion(logit, y)

    loss.backward()  # fine (one step) — caller hasn't stepped optimizer yet
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    fwd_bwd_ms = (t3 - t2) * 1000.0

    # clean up grad so training starts fresh
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    hb_print(f"[profile] 1st batch fetch: {fetch_ms:.1f} ms")
    hb_print(f"[profile] 1 fwd+bwd on GPU: {fwd_bwd_ms:.1f} ms")

    return {"fetch_ms": fetch_ms, "fwd_bwd_ms": fwd_bwd_ms}


# ---- cache for time ramps to avoid per-step allocations ----
class RampCache:
    def __init__(self, device):
        self.device = device
        self.cache = {}  # (L, pred_len) -> (ptf[L,1], ftf[pred_len,1])
    def get(self, L, pred_len):
        key = (int(L), int(pred_len))
        if key not in self.cache:
            ptf = torch.arange(L, device=self.device, dtype=torch.float32).view(L, 1) / float(max(1, L))
            ftf = torch.arange(L, L + pred_len, device=self.device, dtype=torch.float32).view(pred_len, 1) / float(max(1, L))
            self.cache[key] = (ptf, ftf)
        return self.cache[key]

@torch.no_grad()
def collect_logits(model, loader, device, ramp_cache, use_amp=True):
    model.eval()
    all_logits, all_y = [], []
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type=="cuda")
    for ctx, _, _, y in loader:
        B, L, _ = ctx.shape
        ctx = ctx.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True).view(-1)

        ptf1, ftf1 = ramp_cache.get(L, loader.dataset.dataset.pred_len if hasattr(loader.dataset, "dataset") else loader.dataset.pred_len)
        ptf = ptf1.view(1, L, 1).expand(B, L, 1)
        ftf = ftf1.view(1, -1, 1).expand(B, ftf1.shape[0], 1)

        with amp_ctx:
            logit = model(
                past_values=ctx,
                past_observed_mask=ctx.new_ones(ctx.shape, dtype=torch.bool),
                past_time_features=ptf
            ).view(-1)
        all_logits.append(logit.float().cpu())
        all_y.append(y.cpu())

    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy().astype(int)

# ============================ main ============================
def main():
    # system / perf knobs
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hb(f"start | host={platform.node()} torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")

    # ---- env knobs ----
    RUN_NAME        = os.getenv("RUN_NAME", "tst_run")
    HEAD_EPOCHS     = int(os.getenv("HEAD_EPOCHS", "15"))
    BATCH_SIZE      = int(os.getenv("HEAD_BSZ", "1024"))   # ↑ bigger default; adjust by GPU RAM
    LR_HEAD         = float(os.getenv("HEAD_LR", "2e-3"))
    HEAD_POOL       = os.getenv("HEAD_POOL", "mean")       # "mean" or "max"
    LABEL_SCOPE     = os.getenv("LABEL_SCOPE", "context")
    LABELS_CSV      = os.getenv("LABELS_CSV", "/home/vault/iwi5/iwi5305h/new_dataset_90kv/labels_for_parquet.csv")
    VAL_FRACTION    = float(os.getenv("VAL_FRAC", "0.2"))
    POS_BOOST       = float(os.getenv("POS_BOOST", "1.0"))
    UNFREEZE_LAST   = int(os.getenv("UNFREEZE_LAST", "0"))
    ENC_LR          = float(os.getenv("ENC_LR", "5e-5"))
    # dataloader perf
    NUM_WORKERS     = int(os.getenv("NUM_WORKERS", "16"))
    PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "8"))
    PERSISTENT_W    = bool(int(os.getenv("PERSISTENT_WORKERS", "1")))
    # training perf
    USE_AMP         = bool(int(os.getenv("AMP", "1")))
    ACC_STEPS       = int(os.getenv("ACC_STEPS", "1"))     # gradient accumulation
    VAL_FRAC  = float(os.getenv("VAL_FRAC",  "0.10"))
    TEST_FRAC = float(os.getenv("TEST_FRAC", "0.10"))

    events_map = load_events_map(LABELS_CSV)

    ckpt_dir = Path(os.path.expandvars(CHECKPOINT_HUGGING_FACE_DIR)).expanduser() / RUN_NAME
    last_ckpt = ckpt_dir / "last.pt"
    if not last_ckpt.exists():
        raise FileNotFoundError(f"No backbone checkpoint found at {last_ckpt}")

    # ---- dataset ----
    cfg = load_config(f"{HUGGING_FACE_TRANSFORMERS_DIR}/hyper_parameter.json")
    USE_MEMMAP = bool(int(os.getenv("USE_MEMMAP", "0")))

    if USE_MEMMAP:
        npy_dir = os.path.join(CACHE_FILE_PATH, "_npy_cache")  # where you saved .npy
        ds_full = NpyTimeSeriesSimple(
            npy_dir=npy_dir,
            window=int(cfg["window_ms"] / 1000.0 * cfg["sample_rate"]),
            pred=int(cfg["pred_ms"] / 1000.0 * cfg["sample_rate"]),
            stride=int(cfg["stride_ms"] / 1000.0 * cfg["sample_rate"]),
            events_map=events_map,
            label_scope=LABEL_SCOPE,
        )
    else:
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

    ds_train, ds_val, ds_test, train_labels, val_labels, test_labels = \
        stratified_file_split_3way(ds_full, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=42)

    hb(f"[data] train={len(ds_train)} val={len(ds_val)} | "
    f"pos-rate train={train_labels.mean():.4f} val={val_labels.mean():.4f}")

    # --- compute true class prior on TRAIN ---
    pos = int((train_labels == 1).sum())
    neg = int((train_labels == 0).sum())
    p_real = pos / max(1, pos + neg)
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
    hb(f"[data] train prior p(pos)={p_real:.4f} | pos_weight={pos_weight.item():.3f}")

    # --- loaders: NO sampler; use natural class distribution ---
    train_loader = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=PERSISTENT_W, prefetch_factor=PREFETCH_FACTOR
    )

    val_loader = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_W,
        prefetch_factor=PREFETCH_FACTOR
    )
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                         pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_W,
                         prefetch_factor=PREFETCH_FACTOR)

    hb(f"[data] train={len(ds_train)} val={len(ds_val)} | pos-rate train={sum(train_labels)/max(1,len(train_labels)):.4f}")

    # ---- base + head ----
    hf_cfg = TimeSeriesTransformerConfig(
        input_size=ds_full.n_features,
        context_length=ds_full.seq_len,
        prediction_length=ds_full.pred_len,
        lags_sequence=[0],
        distribution_output="normal",
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

    with torch.no_grad():
        logit_p = float(np.log(p_real / max(1e-12, 1.0 - p_real)))
        model.head.bias.data.fill_(logit_p)
        hb(f"[init] head bias set to logit(p)={logit_p:.3f}")


    # freeze all backbone (optionally unfreeze last)
    for p in model.base.parameters(): p.requires_grad = False
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
    scaler = amp.GradScaler('cuda', enabled=USE_AMP and device.type=="cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {"f1": -1.0, "thr": 0.5}
    ramp_cache = RampCache(device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP and device.type=="cuda")

    if bool(int(os.getenv("PROFILE_ONCE", "1"))):
        profile_input_pipeline(train_loader, model, device, criterion, USE_AMP, hb_print=hb)

    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        tp = fp = tn = fn = 0

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            if len(batch) != 4:
                raise RuntimeError("Dataset must return (ctx, tgt, times, y).")
            ctx, _, _, y = batch
            B, L, _ = ctx.shape

            ctx = ctx.to(device, non_blocking=True)
            y   = y.to(device, non_blocking=True).view(-1)

            ptf1, ftf1 = ramp_cache.get(L, ds_full.pred_len)
            ptf = ptf1.view(1, L, 1).expand(B, L, 1)
            ftf = ftf1.view(1, ds_full.pred_len, 1).expand(B, ds_full.pred_len, 1)

            with amp_ctx:
                logit = model(
                    past_values=ctx,
                    past_observed_mask=torch.ones_like(ctx, dtype=torch.bool, device=device),
                    past_time_features=ptf,
                ).view(-1)
                loss = criterion(logit, y) / ACC_STEPS

            scaler.scale(loss).backward()

            if step % ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss) * B * ACC_STEPS

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

        # ---- validate & τ* ----
        logits_val, y_val = collect_logits(model, val_loader, device, ramp_cache, use_amp=USE_AMP)
        probs_val = 1.0 / (1.0 + np.exp(-logits_val))
        P, R, T   = precision_recall_curve(y_val, probs_val)
        F1        = 2 * P * R / (P + R + 1e-9)
        if len(T) > 0:
            best_idx = int(np.nanargmax(F1[:-1]))  # thresholds aligned with P/R except last point
            thr = float(T[best_idx]); f1b = float(F1[best_idx])
            if f1b > best["f1"]:
                best["f1"], best["thr"] = f1b, thr
        else:
            thr = 0.5

        tp05, fp05, tn05, fn05, p05, r05, f105 = conf_at(0.5, probs_val, y_val)
        tpb, fpb, tnb, fnb, pb, rb, f1b = conf_at(best["thr"], probs_val, y_val)

        print(
            f"[head] epoch {epoch:02d} | loss/train={total_loss/len(ds_train):.6f} | "
            f"train@0.5 P={precision:.4f} R={recall:.4f} | "
            f"val@0.5 TP={tp05} FP={fp05} TN={tn05} FN={fn05} | P={p05:.4f} R={r05:.4f} F1={f105:.4f} | "
            f"val@τ*={best['thr']:.3f} → P={pb:.4f} R={rb:.4f} F1={f1b:.4f}"
        )

    fixed_thr = float(best["thr"])

    with torch.inference_mode():
        logits_te, y_te = collect_logits(model, test_loader, device, ramp_cache, use_amp=USE_AMP)
    probs_te = 1.0 / (1.0 + np.exp(-logits_te))

    tp, fp, tn, fn, p_te, r_te, f1_te = conf_at(fixed_thr, probs_te, y_te)
    hb(f"[TEST @ τ*={fixed_thr:.3f}] TP={tp} FP={fp} TN={tn} FN={fn} | "
    f"P={p_te:.4f} R={r_te:.4f} F1={f1_te:.4f}")

    # save head + best threshold
    out_head = ckpt_dir / "head_only.pth"
    torch.save({"head_state": model.head.state_dict(), "best_thr": best["thr"]}, out_head)
    hb(f"saved head to {out_head} (τ*={best['thr']:.4f})")

if __name__ == "__main__":
    main()
