# train_hf_backbone_mse.py
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time, platform, threading, os, random, json, signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# keep extra-stable while debugging
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
def _extract_pred_mean(out, n_features: int) -> torch.Tensor:
    # HF TST: prefer logits, else loc, else split params
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if hasattr(out, "loc") and out.loc is not None:
        return out.loc
    if hasattr(out, "params") and out.params is not None:
        p = out.params
        if p.dim() == 3 and p.size(-1) == 2 * n_features:   # (B, pred_len, 2*D) -> [loc|scale]
            return p[..., :n_features]
        if p.dim() == 4 and p.size(-1) == 2:                # (B, pred_len, D, 2) -> [:,:,:,0]
            return p[..., 0]
    raise RuntimeError("Cannot find predicted mean (checked logits, loc, params).")

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
    # unwrap if payload was stored under "model"
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    fixed = {}
    for k, v in sd.items():
        if k.startswith("head."):   # ignore any head params
            continue
        if k.startswith("base."):   # strip wrapper prefix
            k = k[len("base."):]
        fixed[k] = v
    tgt = model.state_dict()
    return {k: v for k, v in fixed.items() if (k in tgt and tgt[k].shape == v.shape)}

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

    sd = _normalize_ckpt_keys_for_base_model(ckpt, model)
    model.load_state_dict(sd, strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception:
        hb("optimizer state not loaded; continuing with fresh optimizer.")
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
    LOG_EVERY_STEPS   = int(os.getenv("LOG_EVERY_STEPS", "500"))
    LOG_EVERY_SECONDS = float(os.getenv("LOG_EVERY_SECONDS", "180"))
    USE_WATCHDOG      = bool(int(os.getenv("WATCHDOG", "0")))
    WATCHDOG_EVERY_S  = int(os.getenv("WATCHDOG_EVERY_S", "300"))
    DO_FIRST_BATCH_PROBE = bool(int(os.getenv("FIRST_BATCH_PROBE", "0")))

    RUN_NAME          = os.getenv("RUN_NAME", "tst_run")
    SAVE_EVERY_EPOCHS = int(os.getenv("SAVE_EVERY_EPOCHS", "1"))
    SAVE_EVERY_STEPS  = int(os.getenv("SAVE_EVERY_STEPS", "0"))
    KEEP_LAST_N       = int(os.getenv("KEEP_LAST_N", "3"))
    RESUME            = bool(int(os.getenv("RESUME", "1")))
    OVERFIT_ONE_BATCH = bool(int(os.getenv("OVERFIT_ONE_BATCH", "0")))
    # -----------------------------------------------------------

    set_seed(42)
    CKPT_HF_DIR = Path(os.path.expandvars(CHECKPOINT_HUGGING_FACE_DIR)).expanduser()
    CKPT_DIR = CKPT_HF_DIR / RUN_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hb(f"start | host={platform.node()} torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        hb(f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    hb(f"ckpt dir = {CKPT_DIR}")

    # dataset (ignore labels for backbone training)
    t0 = time.time()
    ds = ParquetTimeSeriesDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=cfg["sample_rate"],
        window_ms=cfg["window_ms"],
        pred_ms=cfg["pred_ms"],
        stride_ms=cfg["stride_ms"],
        feature_range=(cfg.get("feature_min",0.0), cfg.get("feature_max",1.0)),
        events_map={},               # no labels during backbone training
        label_scope="window",
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

    # model
    hf_cfg = TimeSeriesTransformerConfig(
        input_size=ds.n_features,
        context_length=ds.seq_len,
        prediction_length=ds.pred_len,
        lags_sequence=[0],
        distribution_output="normal",   # we won't use out.loss
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

    criterion = nn.MSELoss()
    assert hf_cfg.prediction_length == ds.pred_len, \
        f"config.prediction_length={hf_cfg.prediction_length} != ds.pred_len={ds.pred_len}"
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

    # ---- optional: overfit a single batch to sanity-check the pipeline ----
    if OVERFIT_ONE_BATCH:
        model.train()
        # 1) get one batch
        batch0 = next(iter(loader))
        if len(batch0) == 4:
            ctx0, tgt0, _times0, _y0 = batch0
        else:
            ctx0, tgt0, _times0 = batch0
        B, L, D = ctx0.shape
        pred_len = tgt0.shape[1]

        device = next(model.parameters()).device
        ctx0 = ctx0.to(device); tgt0 = tgt0.to(device)

        # 2) build correct time ramps (lengths L and pred_len) on the same device
        ptf0 = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1) / float(max(1, L))
        ptf0 = ptf0.expand(B, L, 1)
        ftf0 = torch.arange(L, L + pred_len, device=device, dtype=torch.float32).view(1, pred_len, 1) / float(max(1, L))
        ftf0 = ftf0.expand(B, pred_len, 1)

        batch_enc0 = {
            "past_values": ctx0,
            "past_observed_mask": torch.ones_like(ctx0, dtype=torch.bool, device=device),
            "future_values": tgt0,  # teacher forcing => decoder learns full horizon
            "future_observed_mask": torch.ones_like(tgt0, dtype=torch.bool, device=device),
            "past_time_features": ptf0,
            "future_time_features": ftf0,
        }

        # 3) overfit loop (no autocast; keep it simple so grads are definitely tracked)
        for i in range(200):
            optimizer.zero_grad(set_to_none=True)
            out0 = model(**batch_enc0)
            pred0 = _extract_pred_mean(out0, D)

            # sanity prints the first time
            if i == 0:
                print(f"[overfit] pred shape={tuple(pred0.shape)} tgt shape={tuple(tgt0.shape)}")

            # If your model still returns 1-step, make that obvious and bail with a clear error.
            if pred0.shape[1] != tgt0.shape[1]:
                raise RuntimeError(f"Model returned {pred0.shape[1]} steps but target has {tgt0.shape[1]}.\n"
                                "Check that future_time_features has length pred_len and that future_values is passed.")

            loss0 = criterion(pred0, tgt0)  # MSE on mean forecast
            loss0.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if i % 20 == 0:
                print(f"[overfit] step {i:03d} | loss={loss0.item():.6f}")

        # stop after the probe
        return

    printed = False

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        state["epoch"] = epoch
        model.train()

        epoch_t0 = time.time()
        last_log  = time.time()
        sum_load = sum_move = sum_fwd = sum_bwd = 0.0
        sum_loss = 0.0
        n_steps = n_samples = 0
        sum_model_mse = 0.0
        sum_naive_mse = 0.0

        step_load_start = time.time()
        for step, batch_data in enumerate(loader, 1):
            state["step"] = step

            # accept (ctx, tgt, times) or (ctx, tgt, times, y). Ignore y.
            if len(batch_data) >= 3:
                ctx, tgt, _times = batch_data[:3]
            else:
                raise RuntimeError("Dataset should return (ctx, tgt, times) or (ctx, tgt, times, y)")

            B, L, D = ctx.shape
            if not printed:
                hb(f"shapes: past={tuple(ctx.shape)} future={tuple(tgt.shape)} time={tuple(_times.shape)}")
                printed = True

            # sanitize inputs
            if not torch.isfinite(ctx).all():
                ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(tgt).all():
                tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

            t_move0 = time.time()
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            # normalized time ramps (0..1) on GPU
            ptf = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1) / float(max(1, L))
            ptf = ptf.expand(B, L, 1)
            ftf = torch.arange(L, L + ds.pred_len, device=device, dtype=torch.float32).view(1, ds.pred_len, 1) / float(max(1, L))
            ftf = ftf.expand(B, ds.pred_len, 1)

            batch = {
                "past_values": ctx,
                "past_observed_mask": torch.ones_like(ctx, dtype=torch.bool, device=device),
                "future_values": tgt,   # teacher forcing
                "future_observed_mask": torch.ones_like(tgt, dtype=torch.bool, device=device),
                "past_time_features": ptf,
                "future_time_features": ftf,
            }

            t_move1 = time.time()

            optimizer.zero_grad(set_to_none=True)
            t_fwd0 = time.time()
            with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_FP16):
                # try teacher forcing first
                out  = model(**batch)
                pred = _extract_pred_mean(out, D)  # (B, ?, D)

                skip_batch = False

                if pred.shape[1] != tgt.shape[1]:
                    if pred.shape[1] == 1:
                        if step == 1:
                            hb(f"[warn] decoder produced {pred.shape[1]} step with teacher forcing; "
                            f"retrying no-teacher mode (should yield {tgt.shape[1]}).")
                        # retry **without** future_values (no-teacher mode)
                        batch_no_tf = {
                            "past_values": batch["past_values"],
                            "past_observed_mask": batch["past_observed_mask"],
                            "past_time_features": batch["past_time_features"],
                            "future_time_features": batch["future_time_features"],
                        }
                        out  = model(**batch_no_tf)
                        pred = _extract_pred_mean(out, D)

                    if pred.shape[1] != tgt.shape[1]:
                        if step == 1:
                            hb(f"[error] pred_len={pred.shape[1]} != tgt_len={tgt.shape[1]} (B={B}, D={D}) — skipping this batch.")
                        skip_batch = True
            if skip_batch:
                # don’t backward/step on this batch
                step_load_start = time.time()
                continue

            # normal loss
            loss = criterion(pred, tgt)
            t_fwd1 = time.time()

            t_bwd0 = time.time()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t_bwd1 = time.time()

            # epoch accumulators
            sum_load += (t_move0 - step_load_start)
            sum_move += (t_move1 - t_move0)
            sum_fwd  += (t_fwd1  - t_fwd0)
            sum_bwd  += (t_bwd1  - t_bwd0)
            sum_loss += float(loss)
            n_steps  += 1
            n_samples += B

            # baseline vs model (no grad)
            with torch.no_grad():
                sum_model_mse += criterion(pred, tgt).item()
                naive = ctx[:, -1:, :].expand_as(tgt)
                sum_naive_mse += criterion(naive, tgt).item()

            global_step += 1
            state["gstep"] = global_step

            if (LOG_EVERY_STEPS and (step % LOG_EVERY_STEPS == 0)) or ((time.time() - last_log) > LOG_EVERY_SECONDS):
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
        mean_mse = sum_loss / max(1, n_steps)
        mean_model_mse = sum_model_mse / max(1, n_steps)
        mean_naive_mse = sum_naive_mse / max(1, n_steps)

        print(
            f"Epoch {epoch:02d}: mean_mse={mean_mse:.6f} | "
            f"model_mse={mean_model_mse:.6f} | naive_mse={mean_naive_mse:.6f} | "
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
