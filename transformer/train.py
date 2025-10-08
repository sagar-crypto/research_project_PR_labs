# transformer/train.py
from __future__ import annotations

import sys
from pathlib import Path

# Add project root (one level up from this script) to Python’s import path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
import glob
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import mlflow

from transformer.data_processing import TransformerWindowDataset
from transformer.model import TransformerAutoencoder
from transformer.ploting_util import plot_reconstruction, plot_loss_curve
from transformer.utils import (
    load_json,
    ae_train_one_epoch,
    ae_evaluate,
)

from config import (
    DATA_PATH,
    CHECKPOINT_TRANSFORMERS_DIR,
    TRANSFORMERS_DIR,
    CLUSTERING_TRANSFORMERS_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Transformer Autoencoder for Time-Series Fault Detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(TRANSFORMERS_DIR, "hyperParameters.json"),
        help="Path to hyperparameter config JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{TRANSFORMERS_DIR}/hyperParameters.json",
        help="Path to hyperparameter config JSON file",
    )
    parser.add_argument(
        "--data_pattern",
        type=str,
        default=f"{DATA_PATH}/replica_*.parquet",
        help="Glob pattern for input data",
    )
    return parser.parse_args()


def main(
    sample_rate: int,
    window_ms: int,
    pred_ms: int,
    stride_ms: int,
    mode: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cfg: dict,
):
    # 1) Hyperparameters & MLflow setup
    mlflow.set_experiment("Transformer_Fault_Detection_new")
    mlflow.start_run()
    resume = True  # variable to check older checkpoints

    # Dataset & DataLoader
    dataset = TransformerWindowDataset(
        pattern=cfg.get("data_pattern", f"{DATA_PATH}/replica_*.parquet"),
        sample_rate=sample_rate,
        window_ms=window_ms,
        pred_ms=pred_ms,
        stride_ms=stride_ms,
        feature_range=(cfg.get("feature_min", 0.0), cfg.get("feature_max", 1.0)),
        level1_filter="",
        mode=mode,
    )

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerAutoencoder(
        d_in=len(dataset.keep_cols),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 4),
        num_decoder_layers=cfg.get("num_decoder_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
        window_len=dataset.window_len,
        pred_len=dataset.pred_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    d_model = cfg.get("d_model", 256)  # kept for parity with old signature
    warmup_steps = cfg.get("warmup_steps", 1000)  # ditto (not used directly here)

    best_val_loss = float("inf")
    start_epoch = 1

    # Ensure checkpoint dir exists before any writes
    os.makedirs(CHECKPOINT_TRANSFORMERS_DIR, exist_ok=True)

    if resume:
        ckpts = glob.glob(os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "epoch_*.pth"))
        if ckpts:
            latest = max(
                ckpts, key=lambda fn: int(os.path.basename(fn).split("_")[-1].rstrip(".pth"))
            )
            cp = torch.load(latest, map_location=device)
            model.load_state_dict(cp["model_state"])
            optimizer.load_state_dict(cp["opt_state"])
            start_epoch = cp["epoch"] + 1
            best_val_loss = cp["val_loss"]
            print(f"▶▶▶ Resumed from {latest}, starting at epoch {start_epoch}")

    # Training loop
    best_state = None
    train_losses_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "train_losses.npy")
    val_losses_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "val_losses.npy")

    if os.path.isfile(train_losses_path) and os.path.isfile(val_losses_path):
        train_losses = np.load(train_losses_path).tolist()
        val_losses = np.load(val_losses_path).tolist()
        start_epoch = len(train_losses) + 1
        print(f"Resuming loss history, starting at epoch {start_epoch}")
    else:
        train_losses = []
        val_losses = []
        start_epoch = max(start_epoch, 1)

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = ae_train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = ae_evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Store the train and validation loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        np.save(train_losses_path, np.array(train_losses))
        np.save(val_losses_path, np.array(val_losses))

        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        ckpt_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, f"epoch_{epoch:02d}.pth")
        ckpt = {
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "train_loss": train_loss,
        }
        torch.save(ckpt, ckpt_path)
        latest_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, "latest.pth")
        shutil.copyfile(ckpt_path, latest_path)

        # 2) Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = ckpt.copy()

        # Reconstruction curve plotting
        if epoch % 5 == 0:
            print(f"Scaler loaded from path : {dataset.scaler_path}")
            scaler: MinMaxScaler = joblib.load(dataset.scaler_path)
            # get one batch
            src, tgt = next(iter(val_loader))
            src, tgt = src.to(device), tgt.to(device)
            with torch.no_grad():
                tgt_input = torch.zeros_like(tgt)
                tgt_input[:, 1:, :] = tgt[:, :-1, :]
                recon = model(src, tgt_input)

            # call the util, passing the fitted scaler
            plot_path = plot_reconstruction(
                orig=tgt,
                recon=recon,
                scaler=scaler,
                epoch=epoch,
                feature_idx=0,
                sample_idx=0,
                output_dir=CLUSTERING_TRANSFORMERS_DIR,
                prefix="recon",
            )

    best_path = os.path.join(CHECKPOINT_TRANSFORMERS_DIR, f"best_model_{best_epoch:02d}.pth")
    train_losses = np.load(train_losses_path).tolist()
    val_losses = np.load(val_losses_path).tolist()
    validation_loss_curve = plot_loss_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        start_epoch=epoch,
        output_dir=CLUSTERING_TRANSFORMERS_DIR,
        filename=f"loss_curve{epoch}.png",
    )
    torch.save(best_state, best_path)
    print(f"Training complete. Best epoch={best_epoch}, val_loss={best_val_loss:.6f}")
    print(f"Best model saved to {best_path}")

    mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_json(args.config)

    # Hyperparameters from config
    sample_rate = cfg["sample_rate"]
    window_ms = cfg["window_ms"]
    pred_ms = cfg["pred_ms"]
    stride_ms = cfg["stride_ms"]
    mode = cfg["mode"]
    batch_size = cfg["batch_size"]
    num_epochs = cfg["epochs"]
    learning_rate = cfg["learning_rate"]

    main(
        sample_rate,
        window_ms,
        pred_ms,
        stride_ms,
        mode,
        batch_size,
        num_epochs,
        learning_rate,
        cfg,
    )
