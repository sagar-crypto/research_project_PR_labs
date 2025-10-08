# hugging_face_transformer/utils.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, Subset
import signal


# -------------------- tiny i/o + logging -------------------- #
def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def hb(msg: str) -> None:
    """Lightweight timestamped print."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- labels / events map -------------------- #
def load_events_map(csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    CSV -> {file_stem: [(start_sec, end_sec), ...]}.
    Supports start/end columns in _ms, _sec, or plain start/end.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    import pandas as pd

    df = pd.read_csv(csv_path)

    key_col = None
    for c in ["replica_id", "file_stem", "stem", "id", "file", "filename", "path"]:
        if c in df.columns:
            key_col = c
            break
    if key_col is None:
        return {}

    df["_key"] = df[key_col].apply(lambda v: Path(str(v)).stem)

    if {"start_ms", "end_ms"}.issubset(df.columns):
        df["_s"] = df["start_ms"].astype(float) / 1000.0
        df["_e"] = df["end_ms"].astype(float) / 1000.0
    elif {"start_sec", "end_sec"}.issubset(df.columns):
        df["_s"] = df["start_sec"].astype(float)
        df["_e"] = df["end_sec"].astype(float)
    elif {"start", "end"}.issubset(df.columns):
        df["_s"] = df["start"].astype(float)
        df["_e"] = df["end"].astype(float)
    elif {"start_s", "end_s"}.issubset(df.columns):
        df["_s"] = df["start_s"].astype(float)
        df["_e"] = df["end_s"].astype(float)
    else:
        return {}

    m: Dict[str, List[Tuple[float, float]]] = {}
    for k, g in df.groupby("_key"):
        m[str(k)] = [(float(a), float(b)) for a, b in zip(g["_s"], g["_e"])]
    return m


# -------------------- metrics / thresholds -------------------- #
def conf_at(th: float, probs: np.ndarray, y: np.ndarray):
    """Return TP, FP, TN, FN, precision, recall, F1 at a threshold."""
    pred = (probs >= th).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    return tp, fp, tn, fn, float(prec), float(rec), float(f1)


# -------------------- checkpoint loading -------------------- #
def robust_load_base_from_ckpt(wrapper, ckpt_path: Path, device: torch.device) -> None:
    """
    Load backbone weights into HFWithWindowHead.base with shape checks and flexible key prefixes.
    """
    hb(f"loading base weights from {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload.get("model", payload)

    # normalize keys relative to wrapper.base
    new_sd = {}
    if any(k.startswith("base.") for k in state.keys()):
        for k, v in state.items():
            if k.startswith("base."):
                new_sd[k[len("base.") :]] = v
    else:
        new_sd = dict(state)

    base_sd = wrapper.base.state_dict()
    filtered_sd = {k: v for k, v in new_sd.items() if k in base_sd and hasattr(v, "shape") and base_sd[k].shape == v.shape}

    missing, unexpected = wrapper.base.load_state_dict(filtered_sd, strict=False)
    hb(f"base weights loaded | applied={len(filtered_sd)} skipped={len(new_sd) - len(filtered_sd)}")
    if missing:
        hb(f"load warning: missing {len(missing)} keys (ok if head-only / resized inputs)")
    if unexpected:
        hb(f"load warning: unexpected {len(unexpected)} keys (ignored)")


# -------------------- splits & sampler -------------------- #
def stratified_file_split_3way(ds_full, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Split by *files* (not samples), stratified on 'file has any positives'.
    Returns Subset train/val/test + their per-sample labels arrays.
    """
    labels, stems = [], []
    for i in range(len(ds_full)):
        _, _, _, y = ds_full[i]
        labels.append(int(y.item() >= 0.5))
        file_i = int(np.searchsorted(ds_full.cum_counts, i, side="right") - 1)
        stems.append(Path(ds_full.files[file_i]).stem)
    labels = np.array(labels, dtype=np.int64)
    stems = np.array(stems)

    uniq, inv = np.unique(stems, return_inverse=True)
    file_has_pos = np.zeros(len(uniq), dtype=np.int64)
    np.maximum.at(file_has_pos, inv, labels)

    from sklearn.model_selection import StratifiedShuffleSplit

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(uniq, file_has_pos))

    uniq_tv = uniq[trainval_idx]
    file_has_pos_tv = file_has_pos[trainval_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac / (1.0 - test_frac), random_state=seed)
    tr_rel, va_rel = next(sss2.split(uniq_tv, file_has_pos_tv))

    train_files = set(uniq_tv[tr_rel])
    val_files = set(uniq_tv[va_rel])
    test_files = set(uniq[test_idx])

    tr_idx = [i for i, s in enumerate(stems) if s in train_files]
    va_idx = [i for i, s in enumerate(stems) if s in val_files]
    te_idx = [i for i, s in enumerate(stems) if s in test_files]

    return Subset(ds_full, tr_idx), Subset(ds_full, va_idx), Subset(ds_full, te_idx), labels[tr_idx], labels[va_idx], labels[te_idx]


def build_sampler(labels, pos_boost=1.0):
    """
    WeightedRandomSampler that boosts positives (not used in current train loop but available).
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t, minlength=2).float()
    neg, pos = float(counts[0].item()), float(counts[1].item())
    w_neg, w_pos = 1.0, (neg / max(1.0, pos)) * pos_boost
    weights = torch.where(labels_t == 1, torch.tensor(w_pos), torch.tensor(w_neg)).float()
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True), neg, pos


# -------------------- pipeline profiling -------------------- #
def _take_one(loader):
    it = iter(loader)
    return next(it)


def profile_input_pipeline(train_loader, model, device, criterion, use_amp: bool, hb_print=print):
    """
    Fetch 1 batch (CPU/IO timing) and run 1 fwd+bwd (GPU timing).
    Prints two lines and returns a dict of timings in ms.
    """
    t0 = time.perf_counter()
    batch = _take_one(train_loader)
    t1 = time.perf_counter()
    fetch_ms = (t1 - t0) * 1000.0

    ctx, _, _, y = batch
    ctx = ctx.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).view(-1)

    L = ctx.shape[1]
    ptf = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1).expand(ctx.shape[0], L, 1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda")
    with amp_ctx:
        logit = model(
            past_values=ctx,
            past_observed_mask=torch.ones_like(ctx, dtype=torch.bool, device=device),
            past_time_features=ptf,
        ).view(-1)
        loss = criterion(logit, y)

    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    fwd_bwd_ms = (t3 - t2) * 1000.0

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    hb_print(f"[profile] 1st batch fetch: {fetch_ms:.1f} ms")
    hb_print(f"[profile] 1 fwd+bwd on GPU: {fwd_bwd_ms:.1f} ms")
    return {"fetch_ms": fetch_ms, "fwd_bwd_ms": fwd_bwd_ms}


# -------------------- ramp cache & eval -------------------- #
class RampCache:
    """
    Cache simple time features (ramp) for context/pred windows to avoid per-batch allocations.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, L: int, pred_len: int):
        key = (int(L), int(pred_len))
        if key not in self.cache:
            ptf = torch.arange(L, device=self.device, dtype=torch.float32).view(L, 1) / float(max(1, L))
            ftf = torch.arange(L, L + pred_len, device=self.device, dtype=torch.float32).view(pred_len, 1) / float(max(1, L))
            self.cache[key] = (ptf, ftf)
        return self.cache[key]


@torch.no_grad()
def collect_logits(model, loader, device: torch.device, ramp_cache: RampCache, use_amp: bool = True):
    """
    Evaluate classifier head: returns (logits_np, labels_np).
    """
    model.eval()
    all_logits, all_y = [], []
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda")
    for ctx, _, _, y in loader:
        B, L, _ = ctx.shape
        ctx = ctx.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1)

        ptf1, _ = ramp_cache.get(L, loader.dataset.dataset.pred_len if hasattr(loader.dataset, "dataset") else loader.dataset.pred_len)
        ptf = ptf1.view(1, L, 1).expand(B, L, 1)

        with amp_ctx:
            logit = model(
                past_values=ctx,
                past_observed_mask=ctx.new_ones(ctx.shape, dtype=torch.bool),
                past_time_features=ptf,
            ).view(-1)
        all_logits.append(logit.float().cpu())
        all_y.append(y.cpu())

    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy().astype(int)



def start_watchdog(state: Dict[str, Any], interval: int = 300):
    """
    Background logger that prints liveness every `interval` seconds.
    Returns a threading.Event you can set() to stop.
    """
    import threading

    stop = threading.Event()

    def run():
        while not stop.is_set():
            print(f"[{time.strftime('%H:%M:%S')}] watchdog: alive | epoch={state.get('epoch',0)} step={state.get('step',0)}")
            stop.wait(interval)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return stop


def _rng_state() -> Dict[str, Any]:
    return {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: Dict[str, Any]) -> None:
    try:
        random.setstate(state.get("py_random", random.getstate()))
        if "np_random" in state:
            np.random.set_state(state["np_random"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


def _normalize_ckpt_keys_for_base_model(sd: dict, model: torch.nn.Module) -> dict:
    """
    Normalize checkpoint keys to match a bare HF TST model (strip 'base.' prefix, drop 'head.*', unwrap 'model' payload).
    """
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    fixed = {}
    for k, v in sd.items():
        if k.startswith("head."):
            continue
        if k.startswith("base."):
            k = k[len("base.") :]
        fixed[k] = v
    tgt = model.state_dict()
    return {k: v for k, v in fixed.items() if (k in tgt and hasattr(v, "shape") and tgt[k].shape == v.shape)}


def save_checkpoint(
    ckpt_dir: Path,
    keep_last_n: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_step: int,
    extra: dict | None = None,
) -> None:
    """
    Atomically save a checkpoint and maintain a 'last.pt' pointer (symlink when possible).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "rng_state": _rng_state(),
        "extra": extra or {},
    }
    fname = ckpt_dir / f"ckpt-e{epoch:02d}-s{global_step:08d}.pt"
    tmp = str(fname) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, fname)

    last = ckpt_dir / "last.pt"
    try:
        if last.exists() or last.is_symlink():
            last.unlink()
    except Exception:
        pass
    try:
        last.symlink_to(fname.name)
    except Exception:
        torch.save(payload, last)

    if keep_last_n > 0:
        ckpts = sorted([p for p in ckpt_dir.glob("ckpt-*.pt") if p.is_file()], key=os.path.getmtime)
        while len(ckpts) > keep_last_n:
            old = ckpts.pop(0)
            try:
                old.unlink()
            except Exception:
                break
    print(f"[{time.strftime('%H:%M:%S')}] checkpoint saved: {fname.name}")


def try_resume(
    ckpt_dir: Path,
    resume_flag: bool,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
) -> tuple[int, int]:
    """
    Resume from ckpt_dir/last.pt if present; returns (start_epoch, global_step).
    """
    last = ckpt_dir / "last.pt"
    if not (resume_flag and last.exists()):
        return 1, 0
    target = last
    try:
        if last.is_symlink():
            target = last.resolve()
    except Exception:
        pass

    print(f"[{time.strftime('%H:%M:%S')}] resuming from {target}")
    try:
        ckpt = torch.load(target, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(target, map_location=device)

    sd = _normalize_ckpt_keys_for_base_model(ckpt, model)
    model.load_state_dict(sd, strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception:
        print(f"[{time.strftime('%H:%M:%S')}] optimizer state not loaded; continuing with fresh optimizer.")
    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            print(f"[{time.strftime('%H:%M:%S')}] grad scaler state not loaded; continuing fresh scaler.")

    _set_rng_state(ckpt.get("rng_state", {}))
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    return start_epoch, global_step


def install_signal_handlers(
    ckpt_dir: Path,
    keep_last_n: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    state_ref: dict,
):
    """
    Save a final checkpoint when SIGTERM/SIGINT are received.
    """
    def handler(signum, frame):
        print(f"[{time.strftime('%H:%M:%S')}] signal {signum} received — saving final checkpoint…")
        try:
            save_checkpoint(
                ckpt_dir,
                keep_last_n,
                model,
                optimizer,
                scaler,
                epoch=state_ref.get("epoch", 0),
                global_step=state_ref.get("gstep", 0),
                extra={"reason": f"signal_{signum}"},
            )
        finally:
            raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


def extract_pred_mean(out, n_features: int) -> torch.Tensor:
    """
    Extract the *mean* forecast from a HF TimeSeriesTransformerForPrediction output.
    Checks in order: logits, loc, params (both possible layouts).
    """
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if hasattr(out, "loc") and out.loc is not None:
        return out.loc
    if hasattr(out, "params") and out.params is not None:
        p = out.params
        if p.dim() == 3 and p.size(-1) == 2 * n_features:  # (B, pred_len, 2*D) -> [loc|scale]
            return p[..., :n_features]
        if p.dim() == 4 and p.size(-1) == 2:  # (B, pred_len, D, 2) -> [:,:,:,0]
            return p[..., 0]
    raise RuntimeError("Cannot find predicted mean (checked logits, loc, params).")
