import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time, platform, threading, os, random, json
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

# --- AMP imports (version-proof) --------------------------------------------- # NEW
try:
    from torch.amp.autocast_mode import autocast          # PyTorch ≥ 2.x
    from torch.amp.grad_scaler import GradScaler
except Exception:
    from torch.cuda.amp import autocast, GradScaler       # older path

from config import DATA_PATH, CHECKPOINT_TRANSFORMERS_DIR, TRANSFORMERS_DIR, CLUSTERING_TRANSFORMERS_DIR, HUGGING_FACE_TRANSFORMERS_DIR

os.environ.setdefault("PYTHONUNBUFFERED", "1")            # unbuffered logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # quieter
os.environ.setdefault("HF_HUB_OFFLINE", "1")              # avoid network stalls
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# --- Logging controls (env-overridable) -------------------------------------- # NEW
LOG_EVERY_STEPS   = int(os.getenv("LOG_EVERY_STEPS", "500"))     # 0 = never
LOG_EVERY_SECONDS = float(os.getenv("LOG_EVERY_SECONDS", "180"))  # time gate
USE_WATCHDOG      = bool(int(os.getenv("WATCHDOG", "0")))         # 0=off, 1=on
WATCHDOG_EVERY_S  = int(os.getenv("WATCHDOG_EVERY_S", "300"))     # seconds
DO_FIRST_BATCH_PROBE = bool(int(os.getenv("FIRST_BATCH_PROBE", "0")))  # 0=off

def load_config(path):
    return json.load(open(path, 'r'))

USE_FP16 = True  # safe on V100; keep bf16 OFF
def hb(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# optional: watchdog (now gated by USE_WATCHDOG) -------------------------------- # CHANGED
def start_watchdog(state, interval=300):
    stop = threading.Event()
    def run():
        while not stop.is_set():
            hb(f"watchdog: alive | epoch={state.get('epoch',0)} step={state.get('step',0)}")
            stop.wait(interval)
    th = threading.Thread(target=run, daemon=True); th.start()
    return stop

def main(cfg):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hb(f"start | host={platform.node()} torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        hb(f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

    # ---- dataset/build timing ----
    t0 = time.time()
    ds = ParquetTimeSeriesDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=cfg["sample_rate"],
        window_ms=cfg["window_ms"],
        pred_ms=cfg["pred_ms"],
        stride_ms=cfg["stride_ms"],
    )
    hb(f"dataset ready in {time.time()-t0:.2f}s | len={len(ds)} | n_features={ds.n_features} | seq_len={ds.seq_len} | pred_len={ds.pred_len}")

    # DataLoader: safe debug settings (scale up later)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=(device.type == "cuda"),
    )

    # ---- HF config with lags_sequence = [0] to AVOID the lag error ----
    hf_cfg = TimeSeriesTransformerConfig(
        input_size=ds.n_features,
        context_length=ds.seq_len,
        prediction_length=ds.pred_len,
        lags_sequence=[0],            # keep dataset valid
        distribution_output="normal", # avoid StudentT df crashes
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
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = GradScaler(enabled=USE_FP16)  # NEW

    # watchdog now optional/quiet ------------------------------------------------ # CHANGED
    state = {"epoch": 0, "step": 0}
    watchdog = start_watchdog(state, interval=WATCHDOG_EVERY_S) if USE_WATCHDOG else None

    # first-batch probe is optional (off by default) ----------------------------- # CHANGED
    if DO_FIRST_BATCH_PROBE:
        hb("grabbing first batch …")
        tfb0 = time.time()
        first = next(iter(loader))
        hb(f"got first batch in {time.time()-tfb0:.2f}s | tuple_len={len(first)}")
        del first

    printed = False

    for epoch in range(1, cfg["epochs"] + 1):
        state["epoch"] = epoch
        model.train()
        total = 0.0

        # --- quiet stats accumulators (per-epoch) ------------------------------- # NEW
        epoch_t0 = time.time()
        last_log  = time.time()
        sum_load = sum_move = sum_fwd = sum_bwd = 0.0
        sum_loss = 0.0
        n_steps = n_samples = 0

        step_load_start = time.time()
        for step, batch_data in enumerate(loader, 1):
            state["step"] = step

            # unpack
            if len(batch_data) == 3:
                ctx, tgt, _times = batch_data
            else:
                ctx, tgt = batch_data
                raise RuntimeError("Dataset should return (ctx, tgt, times)")

            B, L, D = ctx.shape
            if not printed:
                hb(f"shapes: past={tuple(ctx.shape)} future={tuple(tgt.shape)} time={tuple(_times.shape)}")
                printed = True

            # move to device
            t_move0 = time.time()
            ctx  = ctx.to(device, non_blocking=True)
            tgt  = tgt.to(device, non_blocking=True)
            ptf  = _times.to(device, non_blocking=True)  # (B, L, 1)

            # create future time features to match tgt length
            dt = 1.0 / cfg["sample_rate"]
            ftf = ptf[:, -1:, :] + dt * torch.arange(1, ds.pred_len + 1, device=device).view(1, -1, 1)

            # build model inputs
            batch = {
                "past_values": ctx,
                "past_observed_mask": torch.ones_like(ctx, dtype=torch.bool, device=device),
                "future_values": tgt,
                "future_observed_mask": torch.ones_like(tgt, dtype=torch.bool, device=device),
                "past_time_features": ptf,
                "future_time_features": ftf,
            }

            t_move1 = time.time()

            # ---- forward (AMP) -------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            t_fwd0 = time.time()
            try:
                ctx_autocast = autocast('cuda', dtype=torch.float16, enabled=USE_FP16)  # new API
            except TypeError:
                ctx_autocast = autocast(dtype=torch.float16, enabled=USE_FP16)          # old API
            with ctx_autocast:
                out  = model(**batch)
                loss = out.loss if hasattr(out, "loss") else criterion(out.logits, tgt)
            t_fwd1 = time.time()

            # ---- backward + step (AMP-safe) -----------------------------------
            t_bwd0 = time.time()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t_bwd1 = time.time()

            total += float(loss) * B

            # --- accumulate quiet stats ---------------------------------------- # NEW
            sum_load += (t_move0 - step_load_start)
            sum_move += (t_move1 - t_move0)
            sum_fwd  += (t_fwd1  - t_fwd0)
            sum_bwd  += (t_bwd1  - t_bwd0)
            sum_loss += float(loss)
            n_steps  += 1
            n_samples += B

            # --- sparse step logging (gated by steps/time) ---------------------- # NEW
            do_step_log = False
            if LOG_EVERY_STEPS and (step % LOG_EVERY_STEPS == 0):
                do_step_log = True
            elif (time.time() - last_log) > LOG_EVERY_SECONDS:
                do_step_log = True
            if do_step_log:
                hb(f"ep {epoch} step {step} | loss={float(loss):.6f}")
                last_log = time.time()

            # next iteration load timer start
            step_load_start = time.time()

        # --- compact epoch summary --------------------------------------------- # NEW
        epoch_dt = time.time() - epoch_t0
        avg_load = sum_load / max(1, n_steps)
        avg_move = sum_move / max(1, n_steps)
        avg_fwd  = sum_fwd  / max(1, n_steps)
        avg_bwd  = sum_bwd  / max(1, n_steps)
        samples_per_s = n_samples / max(1e-9, epoch_dt)

        print(
            f"Epoch {epoch:02d}: mean_loss={sum_loss/max(1,n_steps):.6f} | "
            f"steps={n_steps} | samples/s={samples_per_s:.1f} | "
            f"avg_load={avg_load:.3f}s avg_move={avg_move:.3f}s "
            f"avg_fwd={avg_fwd:.3f}s avg_bwd={avg_bwd:.3f}s"
        )

    # cleanly stop watchdog (if it was started) --------------------------------- # CHANGED
    if watchdog is not None:
        watchdog.set()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = load_config(f"{HUGGING_FACE_TRANSFORMERS_DIR}/hyper_parameter.json")
    main(cfg)
