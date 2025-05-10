import os
import numpy as np
import torch
import mlflow
import json
from torch.utils.data import DataLoader
from data_processing import StreamingWindowDataset
from model import VAE, initialize_weights
from ploting_util import plot_reconstruction
from clustering_util import cluster_latent_features
from config import PROJECT_ROOT, DATA_PATH


def main(sample_rate: int, window_ms: int, stride_ms: int, latent_dim: int, batch_size: int, epochs: int, learning_rate: float, min_cluster_size:int):
    # 1) Hyperparameters & MLflow setup
    mlflow.set_experiment("VAE_Fault_Detection")
    mlflow.start_run()

    # 2) DataLoader
    dataset = StreamingWindowDataset(
        pattern=f"{DATA_PATH}/replica_*.parquet",
        sample_rate=sample_rate,
        window_ms=window_ms,
        stride_ms=stride_ms,
        feature_min=0.0,
        feature_max=1.0
    )
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    # 3) Model & optimizer
    seq_len = int(sample_rate * (window_ms / 1000.0))
    _, n_channels = dataset[0].shape
    model = VAE(in_channels=n_channels,
                length=seq_len,
                latent_size=latent_dim,
                encoder_out_channels=128).cuda()
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    # 4) Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            x = batch.cuda()       # shape: (B, seq_len, 72)
            x2 = x.permute(0,2,1)
            recon, mean, logvar = model(x)

            recon_loss = torch.nn.functional.mse_loss(
                recon, x2, reduction='mean')
            kld = -0.5 * torch.mean(
                1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

    # 5) Extract & save latent features
    model.eval()
    all_latents = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=256):
            mu, logvar, _, _ = model.encoder(batch.cuda())
            z = model.reparameterize(mu, logvar)
            all_latents.append(z.cpu().numpy())
    latent_features = np.concatenate(all_latents, axis=0)
    np.save(f"{PROJECT_ROOT}/latent_features.npy", latent_features)
    mlflow.log_artifact(f"{PROJECT_ROOT}/latent_features.npy")

    # 6) Plot reconstructions for a few examples
    orig_batch, = next(iter(loader)), 
    orig = orig_batch.cuda()
    recon = model(orig)[0].cpu().numpy()
    plot_reconstruction(orig_batch.numpy(), recon,
                        indices=[0,1,2], out_path=f"{PROJECT_ROOT}/recon_plot.png")
    mlflow.log_artifact(f"{PROJECT_ROOT}/recon_plot.png")

    # 7) Cluster latent space with HDBSCAN
    # labels = cluster_latent_features(latent_features,
    #                                  min_cluster_size=min_cluster_size,
    #                                  out_path=f"{PROJECT_ROOT}/clusters.png")
    # mlflow.log_artifact(f"{PROJECT_ROOT}/clusters.png")

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
