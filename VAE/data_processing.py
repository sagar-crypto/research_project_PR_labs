import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import PROJECT_ROOT, SCALER_VAE_DIR

class StreamingWindowDataset(Dataset):
    def __init__(
        self,
        pattern: str,
        sample_rate: float,
        window_ms: float = 50.0,
        stride_ms: float = 0.0,
        feature_min: float = 0.0,
        feature_max: float = 1.0,
        level1_filter: str = ""
    ):
        
        """
        Streams scaled fixed-length windows from Parquet files for a measurement.

        Key properties:
        - Discovers files via glob `pattern`; computes `seq_len` and `stride` in samples
        - Selects canonical CubLine columns, optionally filtered by `level1_filter`
        - Fits one MinMaxScaler across all files; saves per-measurement scaler to disk
        - Uses cumulative window counts for lazy indexing (no preloading)
        - __getitem__ returns a torch Tensor of shape (T, C) in [feature_min, feature_max]
        """
        self.paths = glob.glob(pattern)
        if not self.paths:
            raise FileNotFoundError(f"No files match: {pattern}")

        self.seq_len = int(sample_rate * (window_ms / 1000.0))
        self.stride  = (
            self.seq_len if stride_ms is None
            else int(sample_rate * (stride_ms / 1000.0))
        )

        sample_df = pd.read_parquet(self.paths[0])
        cols = sample_df.columns  # MultiIndex
        if level1_filter != "":
            lvl1 = cols.get_level_values(1)
            mask1 = lvl1.str.contains(level1_filter)
        else:
            mask1 = [True] * len(cols)

        lvl0 = cols.get_level_values(0)
        mask0 = lvl0.str.contains("Cub") & lvl0.str.contains("Line")

        keep = mask0 & mask1
        self.keep_cols = cols[keep]
        if len(self.keep_cols) == 0:
            raise ValueError(f"No columns match measurement '{level1_filter}'")

        self.scaler = MinMaxScaler(feature_range=(int(feature_min), int(feature_max)))
        window_counts = []

        for p in self.paths:
            # load & select
            df = pd.read_parquet(p).loc[:, self.keep_cols.to_list()]

            # fit scaler
            arr = df.values.astype(np.float32)
            self.scaler.partial_fit(arr)

            # compute how many windows in this file
            n = len(df)
            w = max(0, (n - self.seq_len) // self.stride + 1)
            window_counts.append(w)

        # build cumulative window indices
        self.cum_windows = np.concatenate(([0], np.cumsum(window_counts))).astype(int)
        self.level1_filter = level1_filter

    def __len__(self):
        return int(self.cum_windows[-1])

    def __getitem__(self, idx):
        # find file index & window offset
        file_i = int(np.searchsorted(self.cum_windows, idx, side='right') - 1)
        local  = idx - self.cum_windows[file_i]
        start  = local * self.stride
        p      = self.paths[file_i]
        meas = self.level1_filter.replace(" ", "_") if self.level1_filter else "all"
        joblib.dump(self.scaler, f"{SCALER_VAE_DIR}/minmax_scaler_{meas}.pkl")

        # read, select columns, slice window
        df = pd.read_parquet(p).loc[:, self.keep_cols.to_list()]
        window = df.iloc[start : start + self.seq_len].values.astype(np.float32)

        # normalize & to torch
        scaled = self.scaler.transform(window)
        return torch.from_numpy(scaled)
