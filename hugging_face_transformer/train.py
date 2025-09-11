import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time, platform, threading, os, random, json, signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction
)
from hugging_face_transformer.data_processing import ParquetTimeSeriesDataset
import torch.multiprocessing as mp
from pathlib import Path

# AMP imports (version-proof)
try:
    from torch.amp.autocast_mode import autocast          # PyTorch ≥ 2.x
    from torch.amp.grad_scaler import GradScaler
except Exception:
    from torch.cuda.amp import autocast, GradScaler       # older path

from config import DATA_PATH, HUGGING_FACE_TRANSFORMERS_DIR, CHECKPOINT_HUGGING_FACE_DIR

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

def load_config(path):
    return json.load(open(path, 'r'))

# fp16 off keeps things extra-stable while we clean up training
USE_FP16 = False

def hb(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def start_watchdog(state, interval=300):
    stop = threading.Event()
    def run():
        while not stop.is_set():
            hb(f"watchdog: alive | epoch={state.get('epoch',0)} step={state.get('step',0)}")
            stop.wait(interval)
    th = threading.Thread(target=run, daemon=True); th.start()
    return stop

# ---------- Checkpoint helpers ----------
def _rng_state():
    return {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _set_rng_state(state):
    try:
        random.setstate(state.get("py_random", random.getstate()))
        if "np_random" in state: np.random.set_state(state["np_random"])
        if "torch_cpu" in state: torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass

def _normalize_ckpt_keys_for_base_model(sd: dict, model: torch.nn.Module) -> dict:
    """
    Accept checkpoints saved from either:
      - plain HF model (keys like 'model.encoder.*'), or
      - wrapper with 'base.' prefix and optional 'head.*' keys.
    This strips a leading 'base.' and discards any 'head.*' params.
    It also drops keys that don't match shape.
    """
    # unwrap if payload was stored under "model"
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    fixed = {}
    for k, v in sd.items():
        if k.startswith("head."):
            continue
        if k.startswith("base."):
            k = k[len("base."):]
        fixed[k] = v

    tgt = model.state_dict()
    # keep only matching keys/shapes
    filtered = {k: v for k, v in fixed.items() if (k in tgt and tgt[k].shape == v.shape)}
    return filtered

def save_checkpoint(ckpt_dir: Path, keep_last_n: int, model, optimizer, scaler, epoch: int, global_step: int, extra: dict = None):
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

    # update last.pt (symlink if possible, else copy)
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

    # prune
    if keep_last_n > 0:
        ckpts = sorted([p for p in ckpt_dir.glob("ckpt-*.pt") if p.is_file()], key=os.path.getmtime)
        while len(ckpts) > keep_last_n:
            old = ckpts.pop(0)
            try: old.unlink()
            except Exception: break
    hb(f"checkpoint saved: {fname.name}")

def try_resume(ckpt_dir: Path, resume_flag: bool, model, optimizer, scaler, device):
    last = ckpt_dir / "last.pt"
    if not (resume_flag and last.exists()):
        return 1, 0
    target = last
    try:
        if last.is_symlink():
            target = last.resolve()
    except Exception:
        pass

    hb(f"resuming from {target}")
    try:
        ckpt = torch.load(target, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(target, map_location=device)

    # tolerate wrapper checkpoints
    sd = _normalize_ckpt_keys_for_base_model(ckpt, model)
    model.load_state_dict(sd, strict=False)  # strict=False: tolerate harmless misses
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception:
        hb("optimizer state not loaded (shape mismatch or missing); continuing fresh opt state.")
    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            hb("grad scaler state not loaded; continuing fresh scaler.")

    _set_rng_state(ckpt.get("rng_state", {}))
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    return start_epoch, global_step

def install_signal_handlers(ckpt_dir, keep_last_n, model, optimizer, scaler, state_ref):
    def handler(signum, frame):
        hb(f"signal {signum} received — saving final checkpoint…")
        try:
            save_checkpoint(ckpt_dir, keep_last_n, model, optimizer, scaler,
                            epoch=state_ref.get("epoch", 0),
                            global_step=state_ref.get("gstep", 0),
                            extra={"reason": f"signal_{signum}"})
        finally:
            raise SystemExit(0)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try: signal.signal(sig, handler)
        except Exception: pass

# ----------------------------------------------------------------------
def main(cfg):
    # -------- centralized knobs (env-overridable) --------
    LOG_EVERY_STEPS   = int(os.getenv("LOG_EVERY_STEPS", "500"))   # 0 = never
    LOG_EVERY_SECONDS = float(os.getenv("LOG_EVERY_SECONDS", "180"))
    USE_WATCHDOG      = bool(int(os.getenv("WATCHDOG", "0")))
    WATCHDOG_EVERY_S  = int(os.getenv("WATCHDOG_EVERY_S", "300"))
    DO_FIRST_BATCH_PROBE = bool(int(os.getenv("FIRST_BATCH_PROBE", "0")))

    RUN_NAME          = os.getenv("RUN_NAME", "tst_run")
    SAVE_EVERY_EPOCHS = int(os.getenv("SAVE_EVERY_EPOCHS", "1"))
    SAVE_EVERY_STEPS  = int(os.getenv("SAVE_EVERY_STEPS", "0"))
    KEEP_LAST_N       = int(os.getenv("KEEP_LAST_N", "3"))
    RESUME            = bool(int(os.getenv("RESUME", "1")))
    # -----------------------------------------------------------

    set_seed(42)
    CKPT_HF_DIR = Path(os.path.expandvars(CHECKPOINT_HUGGING_FACE_DIR)).expanduser()
    CKPT_DIR = CKPT_HF_DIR / RUN_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hb(f"start | host={platform.node()} torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        hb(f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    hb(f"ckpt dir = {CKPT_DIR}")

    # dataset (we can ignore labels during training; dataset may still return y)
    t0 = time.time()
    ds = ParquetTimeSeriesDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=cfg["sample_rate"],
        window_ms=cfg["window_ms"],
        pred_ms=cfg["pred_ms"],
        stride_ms=cfg["stride_ms"],
        feature_range=(cfg.get("feature_min",0.0), cfg.get("feature_max",1.0)),
        events_map={},               # <- no labels needed for training
        label_scope="window",        # irrelevant when events_map is {}
    )
    hb(f"dataset ready in {time.time()-t0:.2f}s | len={len(ds)} | n_features={ds.n_features} | seq_len={ds.seq_len} | pred_len={ds.pred_len}")

    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=(device.type == "cuda"),
    )

    # model (keep distribution_output='normal' but we won't use out.loss)
    hf_cfg = TimeSeriesTransformerConfig(
        input_size=ds.n_features,
        context_length=ds.seq_len,
        prediction_length=ds.pred_len,
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
    )

    model = TimeSeriesTransformerForPrediction(hf_cfg).to(device)

    # plain MSE between predicted future (out.logits) and target (tgt)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = GradScaler(enabled=USE_FP16)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch, global_step = try_resume(CKPT_DIR, RESUME, model, optimizer, scaler, device)

    state = {"epoch": start_epoch-1, "step": 0, "gstep": global_step}
    watchdog = start_watchdog(state, interval=WATCHDOG_EVERY_S) if USE_WATCHDOG else None
    install_signal_handlers(CKPT_DIR, KEEP_LAST_N, model, optimizer, scaler, state)

    if DO_FIRST_BATCH_PROBE:
        hb("grabbing first batch …")
        tfb0 = time.time()
        first = next(iter(loader))
        hb(f"got first batch in {time.time()-tfb0:.2f}s | tuple_len={len(first)}")
        del first

    printed = False

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        state["epoch"] = epoch
        model.train()

        epoch_t0 = time.time()
        last_log  = time.time()
        sum_load = sum_move = sum_fwd = sum_bwd = 0.0
        sum_loss = 0.0
        n_steps = n_samples = 0

        step_load_start = time.time()
        for step, batch_data in enumerate(loader, 1):
            state["step"] = step

            # Accept either (ctx, tgt, times) or (ctx, tgt, times, y). Ignore y.
            if len(batch_data) == 4:
                ctx, tgt, _times, _y = batch_data
            elif len(batch_data) == 3:
                ctx, tgt, _times = batch_data
            else:
                raise RuntimeError("Dataset should return (ctx, tgt, times) or (ctx, tgt, times, y)")

            B, L, D = ctx.shape
            if not printed:
                hb(f"shapes: past={tuple(ctx.shape)} future={tuple(tgt.shape)} time={tuple(_times.shape)}")
                printed = True

            # sanitize inputs (cheap, safe)
            if not torch.isfinite(ctx).all():
                bad = (~torch.isfinite(ctx)).sum().item()
                hb(f"sanitizer: ctx had {bad} non-finite -> zeroed")
                ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(tgt).all():
                bad = (~torch.isfinite(tgt)).sum().item()
                hb(f"sanitizer: tgt had {bad} non-finite -> zeroed")
                tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

            t_move0 = time.time()
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            # Synthetic, normalized time ramps (0→1 over context length)
            ptf = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1) / float(max(1, L))
            ptf = ptf.expand(B, L, 1)
            ftf = torch.arange(L, L + ds.pred_len, device=device, dtype=torch.float32).view(1, ds.pred_len, 1) / float(max(1, L))
            ftf = ftf.expand(B, ds.pred_len, 1)

            batch = {
                "past_values": ctx,
                "past_observed_mask": torch.ones_like(ctx, dtype=torch.bool, device=device),
                "future_values": tgt,                           # teacher forcing (needed for decoder)
                "future_observed_mask": torch.ones_like(tgt, dtype=torch.bool, device=device),
                "past_time_features": ptf,
                "future_time_features": ftf,
            }

            t_move1 = time.time()

            optimizer.zero_grad(set_to_none=True)
            t_fwd0 = time.time()
            # We IGNORE out.loss (distribution NLL) and train on MSE(out.logits, tgt)
            with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_FP16):
                out  = model(**batch)
                loss = criterion(out.logits, tgt)
            t_fwd1 = time.time()

            t_bwd0 = time.time()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t_bwd1 = time.time()

            sum_load += (t_move0 - step_load_start)
            sum_move += (t_move1 - t_move0)
            sum_fwd  += (t_fwd1  - t_fwd0)
            sum_bwd  += (t_bwd1  - t_bwd0)
            sum_loss += float(loss)
            n_steps  += 1
            n_samples += B

            global_step += 1
            state["gstep"] = global_step

            do_step_log = False
            if LOG_EVERY_STEPS and (step % LOG_EVERY_STEPS == 0):
                do_step_log = True
            elif (time.time() - last_log) > LOG_EVERY_SECONDS:
                do_step_log = True
            if do_step_log:
                hb(f"ep {epoch} step {step} | mse={float(loss):.6f} | gstep={global_step}")
                last_log = time.time()

            if SAVE_EVERY_STEPS and (global_step % SAVE_EVERY_STEPS == 0):
                save_checkpoint(CKPT_DIR, KEEP_LAST_N, model, optimizer, scaler, epoch, global_step,
                                extra={"kind": "step"})

            step_load_start = time.time()

        epoch_dt = time.time() - epoch_t0
        avg_load = sum_load / max(1, n_steps)
        avg_move = sum_move / max(1, n_steps)
        avg_fwd  = sum_fwd  / max(1, n_steps)
        avg_bwd  = sum_bwd  / max(1, n_steps)
        samples_per_s = n_samples / max(1e-9, epoch_dt)

        print(
            f"Epoch {epoch:02d}: mean_mse={sum_loss/max(1,n_steps):.6f} | "
            f"steps={n_steps} | samples/s={samples_per_s:.1f} | "
            f"avg_load={avg_load:.3f}s avg_move={avg_move:.3f}s "
            f"avg_fwd={avg_fwd:.3f}s avg_bwd={avg_bwd:.3f}s"
        )

        if SAVE_EVERY_EPOCHS and (epoch % SAVE_EVERY_EPOCHS == 0):
            save_checkpoint(CKPT_DIR, KEEP_LAST_N, model, optimizer, scaler, epoch, global_step,
                            extra={"kind": "epoch"})

    if watchdog is not None:
        watchdog.set()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = load_config(f"{HUGGING_FACE_TRANSFORMERS_DIR}/hyper_parameter.json")
    main(cfg)
