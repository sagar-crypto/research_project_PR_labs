# transformer/utils.py
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import WeightedRandomSampler
from torch import nn
import torch.optim as optim

# -------------------- Simple IO -------------------- #
def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# -------------------- Collate & Dataloader helpers -------------------- #
def collate_classify(batch):
    xs, ys = zip(*batch)
    xs = torch.stack([x.contiguous() for x in xs], 0)  # (B, L, D)
    ys = torch.stack(
        [y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.float32) for y in ys], 0
    )  # (B,)
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
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    out = clf(x)  # (B,)
    print(f"[sanity] x:{tuple(x.shape)} -> logits:{tuple(out.shape)} ; y:{tuple(y.shape)}")


def make_weighted_sampler(subset):
    labels = []
    for i in range(len(subset)):
        _, y = subset[i]
        labels.append(int(y.item() if torch.is_tensor(y) else y))
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t, minlength=2).float()  # [neg, pos]
    class_w = (1.0 / counts.clamp_min(1.0))  # inverse freq
    weights = class_w[labels_t]
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(labels), replacement=True)


@torch.no_grad()
def collect_logits(clf, loader, device):
    clf.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        all_logits.append(clf(x))
        all_y.append(y)
    return torch.cat(all_logits, 0), torch.cat(all_y, 0)


# -------------------- Threshold & metrics helpers -------------------- #
def best_threshold_from_pr(val_logits: torch.Tensor, val_y: torch.Tensor):
    probs = torch.sigmoid(val_logits).detach().cpu().numpy()
    y = val_y.detach().cpu().numpy().astype(int)
    prec, rec, thr = precision_recall_curve(y, probs)  # thr has len-1
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    i = int(np.nanargmax(f1[:-1]))  # ignore last undefined point
    return float(thr[i]), float(f1[i]), probs, y


def metrics_at(probs: np.ndarray, y: np.ndarray, thr: float):
    pred = (probs >= thr).astype(int)
    acc = accuracy_score(y, pred)
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
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
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
    pred = pred.astype(int)
    y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fpr = (fp / max(1, fp + tn)) * 100.0
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    return tp, tn, fp, fn, fpr, f1


# -------------------- Alignment audit helpers -------------------- #
def _window_labels_from_intervals(n_samples, intervals_idx, win, stride):
    y = np.zeros(n_samples, dtype=np.uint8)
    for s, e in intervals_idx:
        s = int(max(0, min(n_samples, s)))
        e = int(max(0, min(n_samples, e)))
        if e > s:
            y[s:e] = 1
    labels = []
    for start in range(0, n_samples - win + 1, stride):
        labels.append(1 if y[start : start + win].any() else 0)
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


def _expected_pos_rate(parquet_path, se_list, sample_rate, window_ms, stride_ms, unit="sec", offset_sec=0.0):
    n = _nrows_parquet(parquet_path)
    assert sample_rate > 0
    win = int(round(window_ms * sample_rate / 1000.0))
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
        pos += yy
        cnt += 1
    return pos / max(1, cnt)


def audit_alignment(events_map, data_glob, sample_rate, window_ms, stride_ms, ds, max_files=20):
    from glob import glob

    files = sorted(glob(data_glob))[:max_files]
    if not files:
        print("[audit] no parquet files found")
        return
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


# -------------------- Training epoch -------------------- #
def run_epoch(clf, loader, device, criterion=None, optimizer=None):
    train = optimizer is not None
    clf.train(train)
    # keep backbone dropout OFF while frozen
    if train and all(not p.requires_grad for p in clf.backbone.parameters()):
        clf.backbone.eval()

    all_logits, all_y = [], []
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = clf(x)  # (B,)
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
        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(all_y, probs) if len(np.unique(all_y)) == 2 else float("nan")
    except Exception:
        auroc = float("nan")
    avg_loss = (total_loss / len(loader.dataset)) if criterion is not None else float("nan")
    return avg_loss, acc, prec, rec, f1, auroc


# -------------------- Unfreeze + optimizer helpers -------------------- #
def unfreeze_encoder(clf, patterns=("backbone.encoder",)):
    """
    Enable grads for encoder params whose names contain any of patterns.
    Patterns are matched against the FULL name like 'backbone.encoder.layers.0...'.
    """
    pats = tuple(patterns) if patterns else None
    matched = []
    for n, p in clf.backbone.encoder.named_parameters():
        full = f"backbone.encoder.{n}"
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


# -------------------- Head builder -------------------- #
def build_classifier(backbone, head_cfg):
    """
    Build a classification head on top of a given backbone.
    head_cfg:
      - name: 'linear_mean' | 'linear_last' | 'v2'
      - dropout: float
    """
    from transformer.model import FaultClassifier
    from transformer.v2_head import FaultClassifierV2

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
    


def ae_train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train one epoch for seq2seq AE:
      output = model(src, tgt)
      loss   = MSE(output, tgt)
    """
    model.train()
    running_loss = 0.0
    for src, tgt in tqdm(loader, desc="  Training", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * src.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def ae_evaluate(model, loader, criterion, device):
    """Eval loop matching the train loop contract."""
    model.eval()
    running_loss = 0.0
    for src, tgt in tqdm(loader, desc="  Evaluating", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src, tgt)
        loss = criterion(output, tgt)
        running_loss += loss.item() * src.size(0)
    return running_loss / len(loader.dataset)
