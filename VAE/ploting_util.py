import joblib
import numpy as np
import matplotlib.pyplot as plt
from config import PROJECT_ROOT, SCALER_VAE_DIR

def plot_reconstruction(orig: np.ndarray,
                        recon: np.ndarray,
                        indices: list,
                        out_path: str,
                        epoch: int,
                        meas: str):
    """
    Plots and saves reconstruction snapshots for selected channels.

    Key properties:
    - Expects `orig` and `recon` as (B, T, C) in scaled [0, 1] units
    - Inverse-transforms with per-measurement MinMaxScaler: scalers/minmax_scaler_{meas}.pkl
    - For each channel index in `indices`, overlays orig vs recon and saves PNG to `out_path`
    """
    scaler = joblib.load(f"{SCALER_VAE_DIR}/minmax_scaler_{meas}.pkl")

    #    both orig & recon are (B, T, C)
    orig_raw  = scaler.inverse_transform(orig[0])   # shape (T, C)
    recon_raw = scaler.inverse_transform(recon[0])

    for i in indices:
        # pick channel 0 out of C (or loop channels if you like)
        o = orig_raw[:,  i]     # or orig_raw[:, channel]
        r = recon_raw[:,  i]

        plt.figure()
        plt.plot(o, label='orig ch1')
        plt.plot(r, '--', label='recon ch1')
        plt.title(f'Reconstruction Example #{i} ({meas})')
        plt.legend()
        plt.xlabel('time (samples)')
        plt.ylabel('value')
        plt.tight_layout()

        # ensure out_path ends with .png or a filename
        fname = out_path.rstrip('/') + f"/recon_{meas}_idx{i}_epoch_{epoch}.png"
        plt.savefig(fname)
        plt.close()
