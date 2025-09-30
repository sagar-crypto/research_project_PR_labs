# train_classification_head.py
import sys
from pathlib import Path
import pandas as pd
from typing import Tuple
from glob import glob
import pyarrow.parquet as pq

# Add project root (one level up from this script) to Python’s import path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os, json, math, torch, numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    f1_score, precision_recall_curve, average_precision_score, roc_curve
)

from transformer.data_processing import TransformerWindowDataset
from transformer.model import TransformerAutoencoder, FaultClassifier
from transformer.utils_label import make_event_map
import random
from torch.utils.data import Subset
from transformer.v2_head import FaultClassifierV2

from config import DATA_PATH, CHECKPOINT_TRANSFORMERS_DIR, TRANSFORMERS_DIR, CLASSIFY_CFG_PATH

# -----------------------------
# Config helpers
# -----------------------------
CFG_PATH             = os.path.join(TRANSFORMERS_DIR, "hyperParameters.json")   # model/data hyperparams

def load_cfg(p):
    with open(p, "r") as f:
        return json.load(f)

# -----------------------------
# Dataloader helpers
# -----------------------------
def collate_classify(batch):
    xs, ys = zip(*batch)  # list of tensors, list of scalars/tensors
    xs = torch.stack([x.contiguous() for x in xs], 0)  # (B, L, D)
    ys = torch.stack([
        y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.float32)
        for y in ys
    ], 0)  # (B,)
    return xs, ys

def _nrows_parquet(path):
    try:
        return pq.ParquetFile(path).metadata.num_rows  # fast, metadata only
    except Exception:
        return pd.read_parquet(path, engine="pyarrow").shape[0]

@torch.no_grad()
def sanity_forward(clf, loader, device):
    clf.eval()
    x, y = next(iter(loader))
    x, y = x.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
    out = clf(x)  # (B,)
    print(f"[sanity] x:{tuple(x.shape)} -> logits:{tuple(out.shape)} ; y:{tuple(y.shape)}")

def make_weighted_sampler(subset):
    labels = []
    for i in range(len(subset)):
        _, y = subset[i]
        labels.append(int(y.item() if torch.is_tensor(y) else y))
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t, minlength=2).float()   # [neg, pos]
    class_w = (1.0 / counts.clamp_min(1.0))                  # inverse freq
    weights = class_w[labels_t]
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(labels), replacement=True)

@torch.no_grad()
def collect_logits(clf, loader, device):
    clf.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x, y = x.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        all_logits.append(clf(x))
        all_y.append(y)
    return torch.cat(all_logits, 0), torch.cat(all_y, 0)

# -----------------------------
# Threshold helpers
# -----------------------------
def tune_threshold(probs: np.ndarray, y_np: np.ndarray) -> Tuple[float, float]:
    ths = np.linspace(0.05, 0.95, 19)
    f1s = np.array([
        f1_score(y_np, (probs >= t).astype(int), average="binary", zero_division=0)
        for t in ths
    ], dtype=float)
    i = int(f1s.argmax())
    return float(ths[i]), float(f1s[i])

def best_threshold_from_pr(val_logits, val_y):
    probs = torch.sigmoid(val_logits).detach().cpu().numpy()
    y     = val_y.detach().cpu().numpy().astype(int)
    prec, rec, thr = precision_recall_curve(y, probs)   # thr has len-1
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    i  = int(np.nanargmax(f1[:-1]))                     # ignore last undefined point
    return float(thr[i]), float(f1[i]), probs, y

def metrics_at(probs, y, thr):
    pred = (probs >= thr).astype(int)
    acc  = accuracy_score(y, pred)
    P, R, F1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return acc, P, R, F1

def confusion_from_probs(probs: np.ndarray, y: np.ndarray, thr: float):
    pred = (probs >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fpr = (fp / max(1, fp + tn)) * 100.0
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    return tp, tn, fp, fn, fpr, f1

def tau_for_target_precision(y_true: np.ndarray, probs: np.ndarray, target: float):
    P, R, T = precision_recall_curve(y_true, probs)
    if len(T) == 0:
        return None
    mask = P[:-1] >= float(target)
    if np.any(mask):
        return float(T[mask].max())
    return None

def tau_for_target_fpr(y_true: np.ndarray, probs: np.ndarray, target_fpr: float):
    fpr, tpr, thr = roc_curve(y_true, probs)
    if len(thr) == 0:
        return None
    i = int(np.argmin(np.abs(fpr - float(target_fpr))))
    return float(thr[i])

def consec_k(pred: np.ndarray, k: int = 3) -> np.ndarray:
    run = 0
    out = np.zeros_like(pred, dtype=np.int64)
    for i, p in enumerate(pred.astype(int)):
        run = run + 1 if p else 0
        out[i] = 1 if run >= k else 0
    return out

def confusion_from_binary(pred: np.ndarray, y: np.ndarray):
    pred = pred.astype(int); y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fpr = (fp / max(1, fp + tn)) * 100.0
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    return tp, tn, fp, fn, fpr, f1

# -----------------------------
# Alignment audit helpers
# -----------------------------
def _window_labels_from_intervals(n_samples, intervals_idx, win, stride):
    y = np.zeros(n_samples, dtype=np.uint8)
    for s, e in intervals_idx:
        s = int(max(0, min(n_samples, s)))
        e = int(max(0, min(n_samples, e)))
        if e > s:
            y[s:e] = 1
    labels = []
    for start in range(0, n_samples - win + 1, stride):
        labels.append(1 if y[start:start+win].any() else 0)
    return np.array(labels, dtype=np.uint8)

def _iter_intervals(se_any):
    if isinstance(se_any, tuple) and len(se_any) == 2:
        return [(float(se_any[0]), float(se_any[1]))]
    if isinstance(se_any, list):
        out = []
        for it in se_any:
            if isinstance(it, (tuple, list)) and len(it) == 2:
                out.append((float(it[0]), float(it[1])))
        return out
    return []

def _expected_pos_rate(parquet_path, se_list, sample_rate, window_ms, stride_ms,
                       unit="sec", offset_sec=0.0) -> float:
    n = _nrows_parquet(parquet_path)
    assert sample_rate > 0
    win    = int(round(window_ms  * sample_rate / 1000.0))
    stride = int(round(stride_ms * sample_rate / 1000.0))

    intervals = _iter_intervals(se_list)
    if unit not in ("sec", "ms"):
        raise ValueError(f"unit must be 'sec' or 'ms', got {unit!r}")

    def to_idx(t):
        t_sec = (float(t) / 1000.0) if unit == "ms" else float(t)
        return int(round((t_sec + float(offset_sec)) * sample_rate))

    intervals_idx = []
    for s, e in intervals:
        si, ei = to_idx(s), to_idx(e)
        if ei < si:
            si, ei = ei, si
        si = max(0, min(si, max(0, n - 1)))
        ei = max(0, min(ei, n))
        if ei <= si:
            continue
        intervals_idx.append((si, ei))

    if not intervals_idx:
        return 0.0

    wlbl = _window_labels_from_intervals(n, intervals_idx, win, stride)
    return float(wlbl.mean())

def _actual_pos_rate_from_dataset(ds, sample=2000):
    cnt, pos = 0, 0
    limit = min(sample, len(ds))
    for i in range(limit):
        _, y = ds[i]
        yy = int(y.item() if torch.is_tensor(y) else y)
        pos += yy; cnt += 1
    return (pos / max(1, cnt))

def audit_alignment(events_map, data_glob, sample_rate, window_ms, stride_ms, ds, max_files=20):
    files = sorted(glob(data_glob))[:max_files]
    if not files:
        print("[audit] no parquet files found"); return
    rates_sec, rates_ms = [], []
    for p in files:
        base = Path(p).stem
        se_list = events_map.get(base, [])
        if not se_list:
            continue
        rates_sec.append(_expected_pos_rate(p, se_list, sample_rate, window_ms, stride_ms, unit="sec"))
        rates_ms.append(_expected_pos_rate(p, se_list, sample_rate, window_ms, stride_ms, unit="ms"))
    actual = _actual_pos_rate_from_dataset(ds)
    print(f"[audit] expected pos-rate (seconds):      mean={np.mean(rates_sec):.4f} over {len(rates_sec)} files")
    print(f"[audit] expected pos-rate (milliseconds):  mean={np.mean(rates_ms):.4f} over {len(rates_ms)} files")
    print(f"[audit] actual dataset pos-rate (approx):  {actual:.4f}")

# -----------------------------
# Training epoch
# -----------------------------
def run_epoch(clf, loader, device, criterion=None, optimizer=None):
    train = optimizer is not None
    clf.train(train)
    # keep backbone dropout OFF while frozen
    if train and all(not p.requires_grad for p in clf.backbone.parameters()):
        clf.backbone.eval()

    all_logits, all_y = [], []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = clf(x)               # (B,)
        if criterion is not None:
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_y = torch.cat(all_y).numpy()
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(np.int64)

    acc = accuracy_score(all_y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_y, preds, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(all_y, probs) if len(np.unique(all_y)) == 2 else float("nan")
    except ValueError:
        auroc = float("nan")
    avg_loss = (total_loss / len(loader.dataset)) if criterion is not None else float("nan")
    return avg_loss, acc, prec, rec, f1, auroc

# -----------------------------
# Unfreeze + optimizer helpers
# -----------------------------
def unfreeze_encoder(clf, patterns=("backbone.encoder",)):
    """
    Enable grads for encoder params whose names contain any of patterns.
    Patterns are matched against the FULL name like 'backbone.encoder.layers.0...'.
    """
    pats = tuple(patterns) if patterns else None
    matched = []
    for n, p in clf.backbone.encoder.named_parameters():
        full = f"backbone.encoder.{n}"   # <-- build full name
        if (pats is None) or any(tag in full for tag in pats):
            p.requires_grad = True
            matched.append(full)
    return matched

def make_two_group_optimizer(clf, lr_head=1e-3, lr_enc=3e-4, wd=1e-4):
    head_params, enc_params = [], []
    for n, p in clf.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if n.startswith("backbone.encoder.") else head_params).append(p)
    groups = [{"params": head_params, "lr": lr_head, "weight_decay": wd}]
    if enc_params:  # only add if non-empty
        groups.append({"params": enc_params, "lr": lr_enc, "weight_decay": wd})
    print(f"[optim] head params: {len(head_params)} | encoder params: {len(enc_params)}")
    return optim.Adam(groups)

# -----------------------------
# Build classifier from config
# -----------------------------
def build_classifier(backbone, head_cfg):
    name = (head_cfg.get("name") or "linear_mean").lower()
    dropout = float(head_cfg.get("dropout", 0.10))
    if name == "linear_mean":
        return FaultClassifier(backbone, dropout=dropout, pool="mean")
    elif name == "linear_last":
        return FaultClassifier(backbone, dropout=dropout, pool="last")
    elif name == "v2":
        return FaultClassifierV2(backbone, dropout=dropout)
    else:
        raise ValueError(f"Unknown head name: {name!r}. Use 'linear_mean', 'linear_last', or 'v2'.")

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    model_cfg = load_cfg(CFG_PATH)                # data/model hyperparams (existing file)
    clf_cfg   = load_cfg(CLASSIFY_CFG_PATH)       # training + head choice (new file)

    # ----- globals-from-config (kept local to main) -----
    SEED = clf_cfg.get("seed", None)
    RUN_ALIGNMENT_AUDIT = bool(clf_cfg.get("run_alignment_audit", True))

    thr_cfg = clf_cfg.get("thresholding", {})
    TARGET_PRECISION = thr_cfg.get("target_precision", None)
    TARGET_FPR       = thr_cfg.get("target_fpr", None)
    K_SMOOTH         = thr_cfg.get("k_smooth", 3)

    LABELS_CSV = os.getenv("LABELS_CSV") or clf_cfg.get("labels_csv")
    if not LABELS_CSV:
        raise ValueError("labels_csv not set (and LABELS_CSV env not provided).")

    CLS_EPOCHS    = int(clf_cfg.get("epochs", 40))
    FREEZE_EPOCHS = int(clf_cfg.get("freeze_epochs", 2))
    UNFREEZE_LAYERS = tuple(clf_cfg.get("unfreeze_layers", ["backbone.encoder"]))

    opt_cfg = clf_cfg.get("optimizer", {})
    LR_HEAD = float(opt_cfg.get("lr_head", 1e-3))
    LR_ENC  = float(opt_cfg.get("lr_encoder", 3e-4))
    WD      = float(opt_cfg.get("weight_decay", 1e-4))

    head_cfg = clf_cfg.get("head", {})

    # Determinism
    if SEED is not None:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    events_map = make_event_map(LABELS_CSV)  # dict: "replica_123" -> (start_s, end_s) or list of such

    ds = TransformerWindowDataset(
        pattern      = f"{DATA_PATH}/replica_*.parquet",
        sample_rate  = model_cfg["sample_rate"],
        window_ms    = model_cfg["window_ms"],
        pred_ms      = model_cfg["pred_ms"],
        stride_ms    = model_cfg["stride_ms"],
        feature_range= (model_cfg.get("feature_min",0.0), model_cfg.get("feature_max",1.0)),
        level1_filter="",
        mode         ="classify",
        events_map   = events_map,
        label_scope  ="window",
    )

    # --- group split by files (train/val/test) ---
    n_files = len(ds.paths)
    n_train_f = int(0.70 * n_files)
    n_val_f   = int(0.15 * n_files)

    idx_files = np.arange(n_files)
    rng = np.random.RandomState(SEED if SEED is not None else 42)
    rng.shuffle(idx_files)

    train_files = set(idx_files[:n_train_f])
    val_files   = set(idx_files[n_train_f:n_train_f + n_val_f])
    test_files  = set(idx_files[n_train_f + n_val_f:])

    # map each window index -> file index via cum_counts
    file_id_of_idx = np.empty(len(ds), dtype=np.int32)
    for fi in range(n_files):
        a, b = ds.cum_counts[fi], ds.cum_counts[fi + 1]
        file_id_of_idx[a:b] = fi

    train_idx = np.where(np.isin(file_id_of_idx, list(train_files)))[0]
    val_idx   = np.where(np.isin(file_id_of_idx, list(val_files)))[0]
    test_idx  = np.where(np.isin(file_id_of_idx, list(test_files)))[0]

    train_idx_list = [int(i) for i in np.asarray(train_idx).ravel()]
    val_idx_list   = [int(i) for i in np.asarray(val_idx).ravel()]
    test_idx_list  = [int(i) for i in np.asarray(test_idx).ravel()]

    train_ds = Subset(ds, train_idx_list)
    val_ds   = Subset(ds, val_idx_list)
    test_ds  = Subset(ds, test_idx_list)

    # --- sampler: balance batches ---
    sampler = make_weighted_sampler(train_ds)

    num_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    bsz = model_cfg.get("batch_size", 64)

    train_loader = DataLoader(
        train_ds, batch_size=bsz, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        drop_last=True, collate_fn=collate_classify
    )
    val_loader = DataLoader(
        val_ds, batch_size=bsz, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        drop_last=False, collate_fn=collate_classify
    )
    test_loader = DataLoader(
        test_ds, batch_size=bsz, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        drop_last=False, collate_fn=collate_classify
    )

    # --- backbone ---
    backbone = TransformerAutoencoder(
        d_in=len(ds.keep_cols),
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nhead", 8),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 4),
        num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=model_cfg.get("dropout", 0.1),
        window_len=ds.window_len,
        pred_len=ds.pred_len
    ).to(device)

    ckpt_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "latest.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["model_state"], strict=True)
        backbone.to(dtype=torch.float32)
        print(f"Loaded backbone weights from {ckpt_path}")
    else:
        print("⚠️  No forecasting checkpoint found; training classifier from random backbone.")

    # --- head from config ---
    clf = build_classifier(backbone, head_cfg).to(device)

    # freeze backbone at start
    for p in clf.backbone.parameters():
        p.requires_grad = False

    sanity_forward(clf, train_loader, device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam((p for p in clf.parameters() if p.requires_grad),
                           lr=LR_HEAD, weight_decay=WD)

    best = {"f1": -1.0, "state": None, "thr": 0.5}

    if RUN_ALIGNMENT_AUDIT:
        audit_alignment(
            events_map = events_map,
            data_glob  = f"{DATA_PATH}/replica_*.parquet",
            sample_rate= model_cfg["sample_rate"],
            window_ms  = model_cfg["window_ms"],
            stride_ms  = model_cfg["stride_ms"],
            ds         = ds,
            max_files  = 30,
        )

    for epoch in range(1, CLS_EPOCHS + 1):
        # Optional partial unfreeze after a few epochs
        if epoch == (FREEZE_EPOCHS + 1):
            clf.backbone.train()
            patterns = tuple(UNFREEZE_LAYERS) if UNFREEZE_LAYERS else ("backbone.encoder",)
            names = unfreeze_encoder(clf, patterns=patterns) or []
            if (not names) and patterns != ("backbone.encoder",):
                print("[unfreeze] no params matched", patterns, "— falling back to ('backbone.encoder',)")
                names = unfreeze_encoder(clf, patterns=("backbone.encoder",)) or []
            print(f"[unfreeze] enabled grads for {len(names)} params")
            if names:
                print("[unfreeze] sample:", names[:5])
            optimizer = make_two_group_optimizer(clf, lr_head=LR_HEAD, lr_enc=LR_ENC, wd=WD)
            print("🔓 Encoder unfrozen with two-group LRs (head=%.0e, enc=%.0e)" % (LR_HEAD, LR_ENC))

        # ---- train ----
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_auc = run_epoch(
            clf, train_loader, device, criterion, optimizer
        )

        # ---- validate ----
        val_logits, val_y = collect_logits(clf, val_loader, device)
        y_np      = val_y.cpu().numpy().astype(int)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

        # diagnostics
        print("val probs: min/med/mean/max:",
              float(val_probs.min()), float(np.median(val_probs)),
              float(val_probs.mean()), float(val_probs.max()))
        print("val pos rate:", y_np.mean(), " frac>=0.5:", float((val_probs >= 0.5).mean()))

        with torch.no_grad():
            val_loss = criterion(val_logits, val_y.float()).item()

        try:
            val_auc = roc_auc_score(y_np, val_probs)
        except Exception:
            val_auc = float("nan")

        best_thr, best_f1, _, _ = best_threshold_from_pr(val_logits, val_y)
        accT, pT, rT, f1T = metrics_at(val_probs, y_np, best_thr)
        print(f"[{epoch:02d}] val loss {val_loss:.4f} | AUC {val_auc:.3f} | @τ={best_thr:.3f} → acc {accT:.3f} P {pT:.3f} R {rT:.3f} F1 {f1T:.3f}")

        if f1T > best["f1"]:
            best["f1"]   = f1T
            best["thr"]  = float(best_thr)
            best["state"] = {k: v.detach().cpu() for k, v in clf.state_dict().items()}

    # --- AFTER training loop: evaluate TEST at VAL threshold, save artifacts ---
    if best["state"] is not None:
        clf.load_state_dict(best["state"], strict=False)
        clf.eval()

        # collect VAL again for policy thresholds
        val_logits2, val_y2 = collect_logits(clf, val_loader, device)
        val_probs2 = torch.sigmoid(val_logits2).cpu().numpy()
        val_y_np2  = val_y2.cpu().numpy().astype(int)

        # collect TEST
        test_logits, test_y = collect_logits(clf, test_loader, device)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_y_np  = test_y.cpu().numpy().astype(int)

        try:
            roc_auc = roc_auc_score(test_y_np, test_probs)
        except Exception:
            roc_auc = float("nan")
        pr_auc  = average_precision_score(test_y_np, test_probs)

        thr_val = float(best["thr"])
        tp, tn, fp, fn, fpr, f1 = confusion_from_probs(test_probs, test_y_np, thr_val)

        out_json = {
            "head_name": head_cfg.get("name", "linear_mean"),
            "head_dropout": head_cfg.get("dropout", 0.10),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "FPR_percent": round(fpr, 4),
            "F1_percent_at_val_thr": round(f1*100, 4),
            "val_selected_threshold": round(thr_val, 6),
            "ROC_AUC": round(float(roc_auc), 6) if roc_auc == roc_auc else None,
            "PR_AUC":  round(float(pr_auc), 6),
        }

        if (test_probs >= thr_val).sum() == 0:
            print("[warn] zero predicted positives at VAL threshold on TEST; also reporting q=0.95 fallback.")
            thr_q = float(np.quantile(test_probs, 0.95))
            tp_q, tn_q, fp_q, fn_q, fpr_q, f1_q = confusion_from_probs(test_probs, test_y_np, thr_q)
            out_json.update({
                "fallback_quantile": 0.95,
                "fallback_threshold": round(thr_q, 6),
                "F1_percent_at_fallback_thr": round(f1_q*100, 4),
                "FPR_percent_at_fallback_thr": round(fpr_q, 4),
                "TP_at_fallback_thr": tp_q,
                "FP_at_fallback_thr": fp_q,
                "TN_at_fallback_thr": tn_q,
                "FN_at_fallback_thr": fn_q,
            })

        # --- Optional: target operating points ---
        if TARGET_PRECISION is not None:
            tau_prec = tau_for_target_precision(val_y_np2, val_probs2, TARGET_PRECISION)
            if tau_prec is not None:
                pred = (test_probs >= tau_prec).astype(int)
                tp2, tn2, fp2, fn2, fpr2, f12 = confusion_from_binary(pred, test_y_np)
                prec2 = tp2 / max(1, tp2 + fp2)
                rec2  = tp2 / max(1, tp2 + fn2)
                out_json.update({
                    "target_precision": float(TARGET_PRECISION),
                    "tau_at_target_precision": round(float(tau_prec), 6),
                    "precision_at_target_tau": round(float(prec2), 4),
                    "recall_at_target_tau": round(float(rec2), 4),
                    "F1_percent_at_target_tau": round(float(f12*100), 4),
                    "FPR_percent_at_target_tau": round(float(fpr2), 4),
                    "TP_at_target_tau": tp2, "FP_at_target_tau": fp2,
                    "TN_at_target_tau": tn2, "FN_at_target_tau": fn2,
                })

        if TARGET_FPR is not None:
            tau_fpr = tau_for_target_fpr(val_y_np2, val_probs2, TARGET_FPR)
            if tau_fpr is not None:
                pred = (test_probs >= tau_fpr).astype(int)
                tp3, tn3, fp3, fn3, fpr3, f13 = confusion_from_binary(pred, test_y_np)
                prec3 = tp3 / max(1, tp3 + fp3)
                rec3  = tp3 / max(1, tp3 + fn3)
                out_json.update({
                    "target_fpr": float(TARGET_FPR),
                    "tau_at_target_fpr": round(float(tau_fpr), 6),
                    "precision_at_fpr_tau": round(float(prec3), 4),
                    "recall_at_fpr_tau": round(float(rec3), 4),
                    "F1_percent_at_fpr_tau": round(float(f13*100), 4),
                    "FPR_percent_at_fpr_tau": round(float(fpr3), 4),
                    "TP_at_fpr_tau": tp3, "FP_at_fpr_tau": fp3,
                    "TN_at_fpr_tau": tn3, "FN_at_fpr_tau": fn3,
                })

        # --- Optional: K-consecutive smoothing on VAL-threshold predictions ---
        if K_SMOOTH is not None and K_SMOOTH > 1:
            pred_raw = (test_probs >= thr_val).astype(int)
            pred_evt = consec_k(pred_raw, k=int(K_SMOOTH))
            tp_s, tn_s, fp_s, fn_s, fpr_s, f1_s = confusion_from_binary(pred_evt, test_y_np)
            prec_s = tp_s / max(1, tp_s + fp_s)
            rec_s  = tp_s / max(1, tp_s + fn_s)
            out_json.update({
                "k_smooth": int(K_SMOOTH),
                "precision_at_val_thr_smoothed": round(float(prec_s), 4),
                "recall_at_val_thr_smoothed": round(float(rec_s), 4),
                "F1_percent_at_val_thr_smoothed": round(float(f1_s*100), 4),
                "FPR_percent_at_val_thr_smoothed": round(float(fpr_s), 4),
                "TP_at_val_thr_smoothed": tp_s, "FP_at_val_thr_smoothed": fp_s,
                "TN_at_val_thr_smoothed": tn_s, "FN_at_val_thr_smoothed": fn_s,
            })

        # save metrics + artifacts
        with open(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_metrics.json"), "w") as f:
            json.dump(out_json, f, indent=2)

        P, R, T = precision_recall_curve(test_y_np, test_probs)
        pr_df = pd.DataFrame({"precision": P[:-1], "recall": R[:-1], "threshold": T})
        pr_df.to_csv(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_prcurve.csv"), index=False)
        np.save(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_test_probs.npy"), test_probs)
        np.save(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_test_labels.npy"), test_y_np)

        out_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_head.pth")
        thr_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_threshold.npy")
        torch.save(best["state"], out_path)
        np.save(thr_path, np.array([best.get("thr", 0.5)], dtype=np.float32))
        print(f"✓ Saved best head to {out_path}; thr={best['thr']:.4f}")
        print("✓ Wrote test metrics JSON and PR curve CSV.")
    else:
        print("No improvement recorded; nothing saved.")

if __name__ == "__main__":
    main()
