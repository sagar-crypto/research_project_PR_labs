import glob, os
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from config import SCALER_TRANSFORMERS_DIR

class ParquetTimeSeriesDataset(Dataset):
    """
    - Discovers all Parquet files matching `pattern`
    - Flattens MultiIndex headers to single-level names
    - Selects only the "Cub" + "Line" sensor columns
    - Fits a MinMaxScaler over all features
    - Dumps that scaler to SCALER_TRANSFORMERS_DIR
    - Streams windows on-the-fly without caching all in memory
    """

    def __init__(
        self,
        pattern: str,
        sample_rate: float,
        window_ms: float,
        pred_ms: float,
        stride_ms: float,
        feature_range=(0.0, 1.0),
    ):
        # 1) Discover files & compute window sizes
        self.files    = sorted(glob.glob(pattern))
        self.seq_len  = int(window_ms  / 1000.0 * sample_rate)
        self.pred_len = int(pred_ms    / 1000.0 * sample_rate)
        self.stride   = int(stride_ms  / 1000.0 * sample_rate)

        # 2) Read first file to determine columns
        df0      = pd.read_parquet(self.files[0])
        orig_cols = df0.columns
        if not isinstance(orig_cols, pd.MultiIndex):
            raise ValueError("Expected MultiIndex in first parquet")

        # locate time column in the ORIGINAL two‐level index
        lvl1       = orig_cols.get_level_values(1)
        time_mask  = lvl1.str.contains("Zeitpunkt", na=False)
        if not time_mask.any():
            raise ValueError(f"No timestamp column (level1 containing 'Zeitpunkt') in {self.files[0]}")
        time_pair  = orig_cols[time_mask][0]               # e.g. ('ResultsRepl','Zeitpunkt in s')
        self.time_col = f"{time_pair[0]}_{time_pair[1]}"   # flattened name

        # now mask for only your Cub+Line sensors (still in orig_cols)
        lvl0 = orig_cols.get_level_values(0)
        sensor_pairs = orig_cols[lvl0.str.contains("Cub") & lvl0.str.contains("Line")]

        # flatten ALL columns
        df0.columns = [f"{a}_{b}" for a, b in orig_cols]

        # map those sensor_pairs into flattened names → feat_cols
        self.feat_cols  = [f"{a}_{b}" for (a, b) in sensor_pairs]
        self.n_features = len(self.feat_cols)
        # 3) Fit scaler over all files
        self.scaler = MinMaxScaler(feature_range=feature_range)
        for fn in self.files:
            df = pd.read_parquet(fn)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in df.columns]
            arr = df[self.feat_cols].values.astype(np.float32)
            self.scaler.partial_fit(arr)

        # save scaler
        os.makedirs(SCALER_TRANSFORMERS_DIR, exist_ok=True)
        self.scaler_path = str(Path(SCALER_TRANSFORMERS_DIR) / "minmax_scaler_ts_transformer.pkl")
        joblib.dump(self.scaler, self.scaler_path)

        # 4) Precompute window counts for __len__
        counts = []
        for fn in self.files:
            df = pd.read_parquet(fn)
            n = len(df)
            counts.append(max(0, (n - self.seq_len - self.pred_len) // self.stride + 1))
        self.cum_counts = np.concatenate(([0], np.cumsum(counts))).astype(int)

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, idx):
        # locate file and local window index
        file_i = int(np.searchsorted(self.cum_counts, idx, side="right") - 1)
        local  = idx - self.cum_counts[file_i]
        start  = local * self.stride
        fn     = self.files[file_i]

        # load file slice
        df = pd.read_parquet(fn)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in df.columns]

        # extract time & features
        time_arr = df[self.time_col].values.astype(np.float32)
        arr      = self.scaler.transform(df[self.feat_cols].values.astype(np.float32))

        # slice windows
        ctx   = arr[start : start + self.seq_len]
        tgt   = arr[start + self.seq_len : start + self.seq_len + self.pred_len]
        times = time_arr[start : start + self.seq_len]

        return (
            torch.from_numpy(ctx).float(),            # (seq_len, n_features)
            torch.from_numpy(tgt).float(),            # (pred_len, n_features)
            torch.from_numpy(times).float().unsqueeze(-1),  # (seq_len,1)
        )
