import os
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

def plot_reconstruction(
    orig: torch.Tensor,
    recon: torch.Tensor,
    scaler: MinMaxScaler,
    epoch: int,
    feature_idx: int = 0,
    sample_idx: int = 0,
    output_dir: str = ".",
    prefix: str = "recon"
):
    """
    Plots one features true vs. reconstructed curve (denormalized) and saves to disk.

    Args:
      orig         Tensor of shape (batch, seq_len, d_in): scaled ground-truth window
      recon        Tensor of same shape: scaled model output
      scaler       Fitted MinMaxScaler (with .data_min_, .data_max_, etc.)
      epoch        current epoch number
      feature_idx  which feature/channel to plot (default 0)
      sample_idx   which element in the batch to plot (default 0)
      output_dir   directory to save the plot
      prefix       filename/title prefix (e.g. 'recon' or 'forecast')
    """
    os.makedirs(output_dir, exist_ok=True)

    # pull out the one sample: shape (seq_len, d_in)
    orig_np  = orig[sample_idx].cpu().numpy()
    recon_np = recon[sample_idx].cpu().numpy()

    # denormalize: scaler expects shape (n_samples, n_features)
    orig_denorm  = scaler.inverse_transform(orig_np)
    recon_denorm = scaler.inverse_transform(recon_np)

    timesteps = range(orig_denorm.shape[0])

    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, orig_denorm[:, feature_idx],  label="True")
    plt.plot(timesteps, recon_denorm[:, feature_idx], "--",  label="Reconstructed")
    plt.title(f"{prefix.capitalize()} @ Epoch {epoch} (feat {feature_idx})")
    plt.xlabel("Time step")
    plt.ylabel(f"Sensor value (feature {feature_idx})")
    plt.legend()
    plt.tight_layout()

    fname = f"{prefix}_epoch{epoch:03d}.png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path)
    plt.close()
    return path

def plot_loss_curve(
    train_losses: list,
    val_losses: list,
    start_epoch: int,
    output_dir: str,
    filename: str = "loss_curve.png",
    title: str = "Training & Validation Loss"
):
    """
    Plots training and validation loss curves over epochs and saves to disk.

    Args:
      train_losses   List of training losses per epoch.
      val_losses     List of validation losses per epoch.
      start_epoch    Epoch number corresponding to train_losses[0].
      output_dir     Directory to save the plot.
      filename       Name of the output image file.
      title          Plot title.
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(start_epoch, start_epoch + len(train_losses))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved loss curve to {path}")
