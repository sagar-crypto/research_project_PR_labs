# train_classification_head.py
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# Add project root (one level up from this script) to Python’s import path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import (
    DATA_PATH,
    CHECKPOINT_TRANSFORMERS_DIR,
    TRANSFORMERS_DIR,
    CLASSIFY_CFG_PATH,
)

from transformer.data_processing import TransformerWindowDataset
from transformer.model import TransformerAutoencoder
from transformer.utils import (
    # io/config
    load_json,
    # data
    collate_classify,
    make_weighted_sampler,
    collect_logits,
    sanity_forward,
    audit_alignment,
    # training
    run_epoch,
    unfreeze_encoder,
    make_two_group_optimizer,
    # thresholds/metrics
    best_threshold_from_pr,
    metrics_at,
    confusion_from_probs,
    confusion_from_binary,
    tau_for_target_precision,
    tau_for_target_fpr,
    consec_k,
    # heads
    build_classifier,
)

# -----------------------------
# Config path
# -----------------------------
CFG_PATH = os.path.join(TRANSFORMERS_DIR, "hyperParameters.json")  # model/data hyperparams


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    model_cfg = load_json(CFG_PATH)  # data/model hyperparams (existing file)
    clf_cfg = load_json(CLASSIFY_CFG_PATH)  # training + head choice (new file)

    # ----- globals-from-config (kept local to main) -----
    SEED = clf_cfg.get("seed", None)
    RUN_ALIGNMENT_AUDIT = bool(clf_cfg.get("run_alignment_audit", True))

    thr_cfg = clf_cfg.get("thresholding", {})
    TARGET_PRECISION = thr_cfg.get("target_precision", None)
    TARGET_FPR = thr_cfg.get("target_fpr", None)
    K_SMOOTH = thr_cfg.get("k_smooth", 3)

    LABELS_CSV = os.getenv("LABELS_CSV") or clf_cfg.get("labels_csv")
    if not LABELS_CSV:
        raise ValueError("labels_csv not set (and LABELS_CSV env not provided).")

    CLS_EPOCHS = int(clf_cfg.get("epochs", 40))
    FREEZE_EPOCHS = int(clf_cfg.get("freeze_epochs", 2))
    UNFREEZE_LAYERS = tuple(clf_cfg.get("unfreeze_layers", ["backbone.encoder"]))

    opt_cfg = clf_cfg.get("optimizer", {})
    LR_HEAD = float(opt_cfg.get("lr_head", 1e-3))
    LR_ENC = float(opt_cfg.get("lr_encoder", 3e-4))
    WD = float(opt_cfg.get("weight_decay", 1e-4))

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

    from transformer.utils_label import make_event_map  # local import to avoid circulars
    events_map = make_event_map(LABELS_CSV)  # dict: "replica_123" -> (start_s, end_s) or list of such

    ds = TransformerWindowDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=model_cfg["sample_rate"],
        window_ms=model_cfg["window_ms"],
        pred_ms=model_cfg["pred_ms"],
        stride_ms=model_cfg["stride_ms"],
        feature_range=(model_cfg.get("feature_min", 0.0), model_cfg.get("feature_max", 1.0)),
        level1_filter="",
        mode="classify",
        events_map=events_map,
        label_scope="window",
    )

    # --- group split by files (train/val/test) ---
    n_files  = len(ds.paths)
    VAL = 0.10
    TEST = 0.10
    n_train_f = int((1.0 - VAL - TEST) * n_files)  # ~80%
    n_val_f   = int(VAL * n_files)                  # ~10%

    idx_files = np.arange(n_files)
    rng = np.random.RandomState(SEED if SEED is not None else 42)
    rng.shuffle(idx_files)

    train_files = set(idx_files[:n_train_f])
    val_files   = set(idx_files[n_train_f : n_train_f + n_val_f])
    test_files  = set(idx_files[n_train_f + n_val_f :])

    # map each window index -> file index via cum_counts
    file_id_of_idx = np.empty(len(ds), dtype=np.int32)
    for fi in range(n_files):
        a, b = ds.cum_counts[fi], ds.cum_counts[fi + 1]
        file_id_of_idx[a:b] = fi

    train_idx = np.where(np.isin(file_id_of_idx, list(train_files)))[0]
    val_idx   = np.where(np.isin(file_id_of_idx, list(val_files)))[0]
    test_idx  = np.where(np.isin(file_id_of_idx, list(test_files)))[0]

    train_ds = Subset(ds, [int(i) for i in np.asarray(train_idx).ravel()])
    val_ds = Subset(ds, [int(i) for i in np.asarray(val_idx).ravel()])
    test_ds = Subset(ds, [int(i) for i in np.asarray(test_idx).ravel()])

    # --- sampler: balance batches ---
    sampler = make_weighted_sampler(train_ds)

    num_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    bsz = model_cfg.get("batch_size", 64)

    train_loader = DataLoader(
        train_ds,
        batch_size=bsz,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_classify,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bsz,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_classify,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bsz,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_classify,
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
        pred_len=ds.pred_len,
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
    optimizer = optim.Adam((p for p in clf.parameters() if p.requires_grad), lr=LR_HEAD, weight_decay=WD)

    best = {"f1": -1.0, "state": None, "thr": 0.5}

    # --- alignment diagnostics (optional) ---
    if RUN_ALIGNMENT_AUDIT:
        audit_alignment(
            events_map=events_map,
            data_glob=f"{DATA_PATH}/replica_*.parquet",
            sample_rate=model_cfg["sample_rate"],
            window_ms=model_cfg["window_ms"],
            stride_ms=model_cfg["stride_ms"],
            ds=ds,
            max_files=30,
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
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_auc = run_epoch(clf, train_loader, device, criterion, optimizer)

        # ---- validate ----
        val_logits, val_y = collect_logits(clf, val_loader, device)
        y_np = val_y.cpu().numpy().astype(int)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

        # diagnostics
        print(
            "val probs: min/med/mean/max:",
            float(val_probs.min()),
            float(np.median(val_probs)),
            float(val_probs.mean()),
            float(val_probs.max()),
        )
        print("val pos rate:", y_np.mean(), " frac>=0.5:", float((val_probs >= 0.5).mean()))

        with torch.no_grad():
            val_loss = criterion(val_logits, val_y.float()).item()

        try:
            val_auc = roc_auc_score(y_np, val_probs)
        except Exception:
            val_auc = float("nan")

        best_thr, best_f1, _, _ = best_threshold_from_pr(val_logits, val_y)
        accT, pT, rT, f1T = metrics_at(val_probs, y_np, best_thr)
        print(
            f"[{epoch:02d}] val loss {val_loss:.4f} | AUC {val_auc:.3f} | @τ={best_thr:.3f} → acc {accT:.3f} P {pT:.3f} R {rT:.3f} F1 {f1T:.3f}"
        )

        if f1T > best["f1"]:
            best["f1"] = f1T
            best["thr"] = float(best_thr)
            best["state"] = {k: v.detach().cpu() for k, v in clf.state_dict().items()}

    # --- AFTER training loop: evaluate TEST at VAL threshold, save artifacts ---
    if best["state"] is not None:
        clf.load_state_dict(best["state"], strict=False)
        clf.eval()

        # collect VAL again for policy thresholds
        val_logits2, val_y2 = collect_logits(clf, val_loader, device)
        val_probs2 = torch.sigmoid(val_logits2).cpu().numpy()
        val_y_np2 = val_y2.cpu().numpy().astype(int)

        # collect TEST
        test_logits, test_y = collect_logits(clf, test_loader, device)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_y_np = test_y.cpu().numpy().astype(int)

        try:
            roc_auc = roc_auc_score(test_y_np, test_probs)
        except Exception:
            roc_auc = float("nan")
        pr_auc = average_precision_score(test_y_np, test_probs)

        thr_val = float(best["thr"])
        tp, tn, fp, fn, fpr, f1 = confusion_from_probs(test_probs, test_y_np, thr_val)

        out_json = {
            "head_name": head_cfg.get("name", "linear_mean"),
            "head_dropout": head_cfg.get("dropout", 0.10),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "FPR_percent": round(fpr, 4),
            "F1_percent_at_val_thr": round(f1 * 100, 4),
            "val_selected_threshold": round(thr_val, 6),
            "ROC_AUC": round(float(roc_auc), 6) if roc_auc == roc_auc else None,
            "PR_AUC": round(float(pr_auc), 6),
        }

        if (test_probs >= thr_val).sum() == 0:
            print(
                "[warn] zero predicted positives at VAL threshold on TEST; also reporting q=0.95 fallback."
            )
            thr_q = float(np.quantile(test_probs, 0.95))
            tp_q, tn_q, fp_q, fn_q, fpr_q, f1_q = confusion_from_probs(test_probs, test_y_np, thr_q)
            out_json.update(
                {
                    "fallback_quantile": 0.95,
                    "fallback_threshold": round(thr_q, 6),
                    "F1_percent_at_fallback_thr": round(f1_q * 100, 4),
                    "FPR_percent_at_fallback_thr": round(fpr_q, 4),
                    "TP_at_fallback_thr": tp_q,
                    "FP_at_fallback_thr": fp_q,
                    "TN_at_fallback_thr": tn_q,
                    "FN_at_fallback_thr": fn_q,
                }
            )

        # --- target operating points ---
        if TARGET_PRECISION is not None:
            tau_prec = tau_for_target_precision(val_y_np2, val_probs2, TARGET_PRECISION)
            if tau_prec is not None:
                pred = (test_probs >= tau_prec).astype(int)
                tp2, tn2, fp2, fn2, fpr2, f12 = confusion_from_binary(pred, test_y_np)
                prec2 = tp2 / max(1, tp2 + fp2)
                rec2 = tp2 / max(1, tp2 + fn2)
                out_json.update(
                    {
                        "target_precision": float(TARGET_PRECISION),
                        "tau_at_target_precision": round(float(tau_prec), 6),
                        "precision_at_target_tau": round(float(prec2), 4),
                        "recall_at_target_tau": round(float(rec2), 4),
                        "F1_percent_at_target_tau": round(float(f12 * 100), 4),
                        "FPR_percent_at_target_tau": round(float(fpr2), 4),
                        "TP_at_target_tau": tp2,
                        "FP_at_target_tau": fp2,
                        "TN_at_target_tau": tn2,
                        "FN_at_target_tau": fn2,
                    }
                )

        if TARGET_FPR is not None:
            tau_fpr = tau_for_target_fpr(val_y_np2, val_probs2, TARGET_FPR)
            if tau_fpr is not None:
                pred = (test_probs >= tau_fpr).astype(int)
                tp3, tn3, fp3, fn3, fpr3, f13 = confusion_from_binary(pred, test_y_np)
                prec3 = tp3 / max(1, tp3 + fp3)
                rec3 = tp3 / max(1, tp3 + fn3)
                out_json.update(
                    {
                        "target_fpr": float(TARGET_FPR),
                        "tau_at_target_fpr": round(float(tau_fpr), 6),
                        "precision_at_fpr_tau": round(float(prec3), 4),
                        "recall_at_fpr_tau": round(float(rec3), 4),
                        "F1_percent_at_fpr_tau": round(float(f13 * 100), 4),
                        "FPR_percent_at_fpr_tau": round(float(fpr3), 4),
                        "TP_at_fpr_tau": tp3,
                        "FP_at_fpr_tau": fp3,
                        "TN_at_fpr_tau": tn3,
                        "FN_at_fpr_tau": fn3,
                    }
                )

        # --- Optional: K-consecutive smoothing on VAL-threshold predictions ---
        if K_SMOOTH is not None and K_SMOOTH > 1:
            pred_raw = (test_probs >= thr_val).astype(int)
            pred_evt = consec_k(pred_raw, k=int(K_SMOOTH))
            tp_s, tn_s, fp_s, fn_s, fpr_s, f1_s = confusion_from_binary(pred_evt, test_y_np)
            prec_s = tp_s / max(1, tp_s + fp_s)
            rec_s = tp_s / max(1, tp_s + fn_s)
            out_json.update(
                {
                    "k_smooth": int(K_SMOOTH),
                    "precision_at_val_thr_smoothed": round(float(prec_s), 4),
                    "recall_at_val_thr_smoothed": round(float(rec_s), 4),
                    "F1_percent_at_val_thr_smoothed": round(float(f1_s * 100), 4),
                    "FPR_percent_at_val_thr_smoothed": round(float(fpr_s), 4),
                    "TP_at_val_thr_smoothed": tp_s,
                    "FP_at_val_thr_smoothed": fp_s,
                    "TN_at_val_thr_smoothed": tn_s,
                    "FN_at_val_thr_smoothed": fn_s,
                }
            )

        # save metrics + artifacts
        os.makedirs(CHECKPOINT_TRANSFORMERS_DIR, exist_ok=True)
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
