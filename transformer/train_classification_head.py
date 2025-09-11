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
    f1_score, precision_recall_curve, average_precision_score
)

from transformer.data_processing import TransformerWindowDataset
from transformer.model import TransformerAutoencoder, FaultClassifier
from transformer.utils_label import make_event_map

from config import DATA_PATH, CHECKPOINT_TRANSFORMERS_DIR, TRANSFORMERS_DIR, CLUSTERING_TRANSFORMERS_DIR

# =============================
# Hyperparameters / knobs
# =============================
CLS_EPOCHS = 40                 # total classification epochs
FREEZE_EPOCHS = 2               # keep backbone frozen for first N epochs
HEAD_DROPOUT = 0.10             # regularization in the head
UNFREEZE_LAYERS = ("backbone.encoder",)  # name patterns to unfreeze later
RUN_ALIGNMENT_AUDIT = True      # set False after you’re confident

LABELS_CSV = os.getenv("LABELS_CSV", "/home/vault/iwi5/iwi5305h/new_dataset_90kv/labels_for_parquet.csv")
CFG_PATH   = os.path.join(TRANSFORMERS_DIR, "hyperParameters.json")

# =============================
# Utilities
# =============================

def load_cfg(p):
    with open(p, "r") as f:
        return json.load(f)

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

# ---------- interval helpers for the audit ----------

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

# ---------- training epoch ----------

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

# ---------- unfreeze helpers ----------

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

# =============================
# Main
# =============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(CFG_PATH)
    events_map = make_event_map(LABELS_CSV)  # dict: "replica_123" -> (start_s, end_s) or list of such

    ds = TransformerWindowDataset(
        pattern      = f"{DATA_PATH}/replica_*.parquet",
        sample_rate  = cfg["sample_rate"],
        window_ms    = cfg["window_ms"],
        pred_ms      = cfg["pred_ms"],
        stride_ms    = cfg["stride_ms"],
        feature_range= (cfg.get("feature_min",0.0), cfg.get("feature_max",1.0)),
        level1_filter="",
        mode         ="classify",
        events_map   = events_map,
        label_scope  ="window",
    )

    # --- split (note: true group-split by file is better; this is a simple random split) ---
    n_train = int(0.70 * len(ds))
    n_val   = int(0.15 * len(ds))
    n_test  = len(ds) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    # --- sampler: balance batches ---
    sampler = make_weighted_sampler(train_ds)

    num_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    bsz = cfg.get("batch_size", 64)

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

    backbone = TransformerAutoencoder(
        d_in=len(ds.keep_cols),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 4),
        num_decoder_layers=cfg.get("num_decoder_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
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

    clf = FaultClassifier(backbone, dropout=HEAD_DROPOUT, pool="mean").to(device)

    # freeze backbone at start
    for p in clf.backbone.parameters():
        p.requires_grad = False

    sanity_forward(clf, train_loader, device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam((p for p in clf.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)

    best = {"f1": -1.0, "state": None, "thr": 0.5}

    if RUN_ALIGNMENT_AUDIT:
        audit_alignment(
            events_map = events_map,
            data_glob  = f"{DATA_PATH}/replica_*.parquet",
            sample_rate= cfg["sample_rate"],
            window_ms  = cfg["window_ms"],
            stride_ms  = cfg["stride_ms"],
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
            optimizer = make_two_group_optimizer(clf, lr_head=1e-3, lr_enc=3e-4, wd=1e-4)
            print("🔓 Encoder unfrozen with two-group LRs (head=1e-3, enc=3e-4)")

        # ---- train ----
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_auc = run_epoch(
            clf, train_loader, device, criterion, optimizer
        )

        # ---- validate ----
        val_logits, val_y = collect_logits(clf, val_loader, device)
        y_np      = val_y.cpu().numpy().astype(int)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

        # diagnostics
        print("val probs: min/med/mean/max:", float(val_probs.min()), float(np.median(val_probs)), float(val_probs.mean()), float(val_probs.max()))
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

        with open(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_metrics.json"), "w") as f:
            json.dump(out_json, f, indent=2)

        # PR curve + raw dumps
        P, R, T = precision_recall_curve(test_y_np, test_probs)
        pr_df = pd.DataFrame({"precision": P[:-1], "recall": R[:-1], "threshold": T})
        pr_df.to_csv(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_prcurve.csv"), index=False)
        np.save(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_test_probs.npy"), test_probs)
        np.save(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "fault_classifier_test_labels.npy"), test_y_np)

        # Save best weights + threshold
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
