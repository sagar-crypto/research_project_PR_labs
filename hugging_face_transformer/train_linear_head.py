# hugging_face_transformer/train_hf_head.py
from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from sklearn.metrics import precision_recall_curve

from hugging_face_transformer.data_processing import ParquetTimeSeriesDataset
from hugging_face_transformer.linear_head import HFWithWindowHead
from hugging_face_transformer.memmap_dataset import NpyTimeSeriesSimple

from config import (
    DATA_PATH,
    HUGGING_FACE_TRANSFORMERS_DIR,
    CHECKPOINT_HUGGING_FACE_DIR,
    CACHE_FILE_PATH,
)

from hugging_face_transformer.utils import (
    load_config,
    hb,
    set_seed,
    load_events_map,
    conf_at,
    robust_load_base_from_ckpt,
    stratified_file_split_3way,
    profile_input_pipeline,
    RampCache,
    collect_logits,
)


def main():
    # perf knobs
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

    # ---- load config (JSON) ----
    cfg = load_config(f"{HUGGING_FACE_TRANSFORMERS_DIR}/hyper_parameter.json")

    # training / data knobs (with safe defaults if missing)
    RUN_NAME = cfg.get("run_name", "tst_run")
    HEAD_EPOCHS = int(cfg.get("head_epochs", 15))
    BATCH_SIZE = int(cfg.get("batch_size", 1024))
    LR_HEAD = float(cfg.get("head_lr", 2e-3))
    HEAD_POOL = cfg.get("head_pool", "mean")  # "mean" or "max"
    LABEL_SCOPE = cfg.get("label_scope", "context")  # "context" or "window"
    LABELS_CSV = cfg.get("labels_csv", "")  # optional
    VAL_FRAC = float(cfg.get("val_frac", 0.10))
    TEST_FRAC = float(cfg.get("test_frac", 0.10))
    UNFREEZE_LAST = int(cfg.get("unfreeze_last", 0))
    ENC_LR = float(cfg.get("encoder_lr", 5e-5))
    NUM_WORKERS = int(cfg.get("num_workers", 16))
    PREFETCH_FACTOR = int(cfg.get("prefetch_factor", 8))
    PERSISTENT_W = bool(int(cfg.get("persistent_workers", 1)))
    USE_AMP = bool(int(cfg.get("amp", 1)))
    ACC_STEPS = int(cfg.get("acc_steps", 1))
    USE_MEMMAP = bool(int(cfg.get("use_memmap", 0)))
    PROFILE_ONCE = int(cfg.get("profile_once", 1))

    events_map = load_events_map(LABELS_CSV) if LABELS_CSV else {}

    ckpt_dir = Path(os.path.expandvars(CHECKPOINT_HUGGING_FACE_DIR)).expanduser() / RUN_NAME
    last_ckpt = ckpt_dir / "last.pt"
    if not last_ckpt.exists():
        raise FileNotFoundError(f"No backbone checkpoint found at {last_ckpt}")

    # ---- dataset ----
    if USE_MEMMAP:
        npy_dir = os.path.join(CACHE_FILE_PATH, "_npy_cache")
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
            feature_range=(cfg.get("feature_min", 0.0), cfg.get("feature_max", 1.0)),
            events_map=events_map,
            label_scope=LABEL_SCOPE,
        )

    ds_train, ds_val, ds_test, train_labels, val_labels, test_labels = stratified_file_split_3way(
        ds_full, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=42
    )

    hb(
        f"[data] train={len(ds_train)} val={len(ds_val)} | "
        f"pos-rate train={train_labels.mean():.4f} val={val_labels.mean():.4f}"
    )

    # --- class prior (TRAIN) ---
    pos = int((train_labels == 1).sum())
    neg = int((train_labels == 0).sum())
    p_real = pos / max(1, pos + neg)
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
    hb(f"[data] train prior p(pos)={p_real:.4f} | pos_weight={pos_weight.item():.3f}")

    # --- loaders ---
    train_loader = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_W,
        prefetch_factor=PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_W,
        prefetch_factor=PREFETCH_FACTOR,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_W,
        prefetch_factor=PREFETCH_FACTOR,
    )

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
    base = TimeSeriesTransformerForPrediction(hf_cfg).to(device)
    model = HFWithWindowHead(base, d_model=cfg["d_model"], pool=HEAD_POOL).to(device)
    robust_load_base_from_ckpt(model, last_ckpt, device=device)

    with torch.no_grad():
        logit_p = float(np.log(p_real / max(1e-12, 1.0 - p_real)))
        model.head.bias.data.fill_(logit_p)
        hb(f"[init] head bias set to logit(p)={logit_p:.3f}")

    # freeze all backbone (optionally unfreeze last)
    for p in model.base.parameters():
        p.requires_grad = False
    param_groups = [{"params": model.head.parameters(), "lr": LR_HEAD, "weight_decay": 1e-4}]
    if UNFREEZE_LAST:
        try:
            last_blk = model.base.model.encoder.layers[-1]
            for p in last_blk.parameters():
                p.requires_grad = True
            param_groups.append({"params": last_blk.parameters(), "lr": ENC_LR, "weight_decay": 1e-4})
            hb("🔓 unfroze last encoder block")
        except Exception as e:
            hb(f"could not unfreeze last encoder block: {e}")

    optimizer = optim.Adam(param_groups)
    scaler = amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {"f1": -1.0, "thr": 0.5}
    ramp_cache = RampCache(device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP and device.type == "cuda")

    if PROFILE_ONCE:
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
            y = y.to(device, non_blocking=True).view(-1)

            ptf1, _ = ramp_cache.get(L, ds_full.pred_len)
            ptf = ptf1.view(1, L, 1).expand(B, L, 1)

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
                pred = prob >= 0.5
                yb = y >= 0.5
                tp += (pred & yb).sum().item()
                tn += ((~pred) & (~yb)).sum().item()
                fp += (pred & (~yb)).sum().item()
                fn += ((~pred) & yb).sum().item()

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        # ---- validate & τ* ----
        logits_val, y_val = collect_logits(model, val_loader, device, ramp_cache, use_amp=USE_AMP)
        probs_val = 1.0 / (1.0 + np.exp(-logits_val))
        P, R, T = precision_recall_curve(y_val, probs_val)
        F1 = 2 * P * R / (P + R + 1e-9)
        if len(T) > 0:
            best_idx = int(np.nanargmax(F1[:-1]))  # thresholds aligned with P/R except last point
            thr = float(T[best_idx])
            f1b = float(F1[best_idx])
            if f1b > best["f1"]:
                best["f1"], best["thr"] = f1b, thr
        else:
            thr = 0.5  # fallback (won't be used if T is empty)

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
    hb(
        f"[TEST @ τ*={fixed_thr:.3f}] TP={tp} FP={fp} TN={tn} FN={fn} | "
        f"P={p_te:.4f} R={r_te:.4f} F1={f1_te:.4f}"
    )

    # save head + best threshold
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_head = ckpt_dir / "head_only.pth"
    torch.save({"head_state": model.head.state_dict(), "best_thr": best["thr"]}, out_head)
    hb(f"saved head to {out_head} (τ*={best['thr']:.4f})")


if __name__ == "__main__":
    main()
