import numpy as np
import torch
import mlflow
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_processing import StreamingWindowDataset
from model import VAE, initialize_weights
from ploting_util import plot_reconstruction
from clustering_util import cluster_latent_features
from config import PROJECT_ROOT, DATA_PATH
import re
import os, glob



CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
resume = True



def main(
    sample_rate: int,
    window_ms: int,
    stride_ms: int,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    min_cluster_size: int,
    plot_interval: int = 10,
):
    # 1) Hyperparameters & MLflow setup
    mlflow.set_experiment("VAE_Fault_Detection")
    mlflow.start_run()

    MEASUREMENTS = [
    "Sekundärstrom L1 in A",
    "Sekundärstrom L2 in A",
    "Sekundärstrom L3 in A",
    "Sekundärspannung L1 in V",
    "Sekundärspannung L2 in V",
    "Sekundärspannung L3 in V",
    ]

    for meas in MEASUREMENTS:
        print(f"\n▶▶▶ Training VAE for measurement: {meas}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 2) DataLoader
        dataset = StreamingWindowDataset(
            pattern=f"{DATA_PATH}/replica_*.parquet",
            sample_rate=sample_rate,
            window_ms=window_ms,
            stride_ms=stride_ms,
            feature_min=0.0,
            feature_max=1.0,
            level1_filter=re.escape(meas),
        )
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # small, non-shuffled loader for snapshots
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        seq_len = int(sample_rate * (window_ms / 1000.0))
        _, n_channels = dataset[0].shape

        model = VAE(
            in_channels=n_channels,
            length=seq_len,
            latent_size=latent_dim,
            encoder_out_channels=128,
        ).to(device)
        model.apply(initialize_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        #checking for checkpoints
        start_epoch = 1
        if resume:
            ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth"))
            if ckpts:
                latest = max(ckpts, key=lambda fn: int(fn.rstrip(".pth").split("_")[-1]))
                cp = torch.load(latest, map_location=device)
                model.load_state_dict(cp["model_state"])
                optimizer.load_state_dict(cp["opt_state"])
                start_epoch = cp["epoch"] + 1
                print(f"Resumed from {latest}, starting at epoch {start_epoch}")

        # 4) Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                x = batch.to(device)            # (B, seq_len, C)
                x2 = x.permute(0, 2, 1)     # (B, C, seq_len)

                recon, mean, logvar = model(x)

                # pad/truncate to match x2
                if recon.size(2) < x2.size(2):
                    recon = F.pad(recon, (0, x2.size(2) - recon.size(2)))
                elif recon.size(2) > x2.size(2):
                    recon = recon[:, :, : x2.size(2)]

                recon_loss = torch.nn.functional.mse_loss(recon, x2, reduction='mean')
                kld        = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                loss       = recon_loss + kld

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)

            avg_loss = total_loss / len(dataset)
            mlflow.log_metric(f"{meas}_train_loss", avg_loss, step=epoch)
            print(f"[{meas}] Epoch {epoch}/{epochs} loss={avg_loss:.4f}")
            

            #saving checkpoint
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"→ Saved checkpoint: {ckpt_path}")

            if epoch % plot_interval == 0:
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(test_loader))      # shape (1, seq_len, n_lines)
                    orig = sample_batch.to(device)              # still (1, seq_len, n_lines)

                    recon_batch, _, _ = model(orig)             # (1, n_lines, T_out)

                    T_in  = orig.size(1)       # time length
                    T_out = recon_batch.size(2)
                    if T_out < T_in:
                        recon_batch = F.pad(recon_batch, (0, T_in - T_out))
                    elif T_out > T_in:
                        recon_batch = recon_batch[:, :, :T_in]

                    orig_np  = orig.cpu().numpy()                              # (1, T, C)
                    recon_np = recon_batch.permute(0, 2, 1).cpu().numpy()       # (1, T, C)

                    out_dir = f"{PROJECT_ROOT}/clustering_img"
                    plot_reconstruction(
                        orig_np,
                        recon_np,
                        indices=[0],
                        out_path=out_dir,
                        meas=meas.replace(" ", "_")    # match your scaler filename
                    )
                    mlflow.log_artifact(out_dir)

                #deleting old checkpoints
                old = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth")))
                for fn in old[:-3]:
                    os.remove(fn)
                model.train()

        # 5) Save final model
        torch.save(model.state_dict(), f"{PROJECT_ROOT}/{meas.replace(' ','_')}_vae.pth")

        # 6) Extract & save latent features
        model.eval()
        all_latents = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=256):
                mu, logvar, _ = model.encoder(batch.cuda())
                z = model.reparameterize(mu, logvar)
                all_latents.append(z.cpu().numpy())
        latent_features = np.concatenate(all_latents, axis=0)
        np.save(f"{PROJECT_ROOT}/{meas.replace(' ','_')}_latents.npy", latent_features)
        mlflow.log_artifact(f"{PROJECT_ROOT}/{meas.replace(' ','_')}_latents.npy")

        # 7) Final clustering plot
        labels = cluster_latent_features(
            latent_features,
            min_cluster_size=min_cluster_size,
            out_path=f"{PROJECT_ROOT}/clusters.png",
        )
        mlflow.log_artifact(f"{PROJECT_ROOT}/clusters.png")

    mlflow.end_run()

if __name__ == "__main__":

    with open(f'{PROJECT_ROOT}/hyperParameters.json', 'r') as f:
        best_params = json.load(f)

    # Extract best parameters
    best_sample_rate = best_params['sample_rate']
    best_window_ms = best_params['window_ms']
    best_stride_ms = best_params['stride_ms']
    best_latent_dim = best_params['latent_dim']
    best_batch_size = best_params['batch_size']
    best_epochs = best_params['epochs']
    best_learning_rate = best_params['learning_rate']
    best_min_cluster_size = best_params['min_cluster_size']
    main(sample_rate=best_sample_rate, window_ms=best_window_ms, stride_ms=best_stride_ms, latent_dim=best_latent_dim, batch_size=best_batch_size, epochs=best_epochs, learning_rate=best_learning_rate, min_cluster_size=best_min_cluster_size)
