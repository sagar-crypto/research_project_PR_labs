import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(orig: np.ndarray,
                        recon: np.ndarray,
                        indices: list,
                        out_path: str):
    """
    orig, recon: arrays of shape (batch_size, seq_len, channels)
    indices: list of batch indices to plot
    """
    for i in indices:
        plt.figure()
        plt.plot(orig[i,:,0], label='orig ch1')
        plt.plot(recon[i,:,0], '--', label='recon ch1')
        plt.title(f'Reconstruction Example #{i}')
        plt.legend()
        plt.xlabel('time (samples)')
        plt.ylabel('value')
        plt.tight_layout()
        plt.savefig(f"{out_path.rstrip('.png')}_{i}.png")
