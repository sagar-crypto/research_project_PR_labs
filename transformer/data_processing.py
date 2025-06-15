import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import PROJECT_ROOT

class TransformerWindowDataset(Dataset):
    """
    - Reads all files matching `pattern`.
    - Fits a MinMaxScaler over all data in __init__, then dumps it once.
    - Slices each file into overlapping windows of length `window_ms` ms.
    - For each window returns either:
      * Forecasting mode: (src, tgt), where src is the first (w - p) samples
        and tgt is the last p samples.
      * Reconstruction mode: (window, window).
    """

    def __init__(
        self,
        pattern: str,
        sample_rate: float,
        window_ms: float = 50.0,
        pred_ms: float = 0.0,
        stride_ms: float = 0.0,
        feature_range=(0.0, 1.0),
        level1_filter: str = "",
        mode: str = "forecast"  # one of ["forecast", "reconstruct"]
    ):
        # 1) Discover files
        self.paths = glob.glob(pattern)
        if not self.paths:
            raise FileNotFoundError(f"No files match: {pattern}")

        # 2) Convert window & stride from ms → samples
        self.window_len = int(sample_rate * (window_ms / 1000.0))
        self.pred_len   = int(sample_rate * (pred_ms    / 1000.0))
        if stride_ms is None:
            self.stride = self.window_len
        else:
            self.stride = int(sample_rate * (stride_ms / 1000.0))

        if self.pred_len >= self.window_len:
            raise ValueError("pred_ms must be smaller than window_ms")

        self.mode = mode
        if mode not in {"forecast", "reconstruct"}:
            raise ValueError("mode must be 'forecast' or 'reconstruct'")

        # 3) Pick columns once
        sample_df = pd.read_parquet(self.paths[0])
        cols = sample_df.columns  # MultiIndex
        if level1_filter:
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

        # 4) Fit scaler over all data in one pass
        self.scaler = MinMaxScaler(feature_range=feature_range)
        window_counts = []
        for p in self.paths:
            df = pd.read_parquet(p).loc[:, self.keep_cols.to_list()]
            arr = df.values.astype(np.float32)
            self.scaler.partial_fit(arr)
            n = len(df)
            count = max(0, (n - self.window_len) // self.stride + 1)
            window_counts.append(count)

        # Dump scaler once
        meas = level1_filter.replace(" ", "_") if level1_filter else "all"
        scaler_path = f"{PROJECT_ROOT}/scalers/minmax_scaler_{meas}.pkl"
        joblib.dump(self.scaler, scaler_path)

        # 5) Build cumulative window index for __getitem__ lookup
        self.cum_counts = np.concatenate(([0], np.cumsum(window_counts))).astype(int)

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, idx):
        # 1) Find which file & local window index
        file_i = int(np.searchsorted(self.cum_counts, idx, side="right") - 1)
        local  = idx - self.cum_counts[file_i]
        start  = local * self.stride
        path   = self.paths[file_i]

        # 2) Load, select, slice
        df = pd.read_parquet(path).loc[:, self.keep_cols.to_list()]
        window = df.iloc[start : start + self.window_len].values.astype(np.float32)

        # 3) Normalize
        window = self.scaler.transform(window)

        # 4) Split into src/tgt or full/full
        if self.mode == "forecast":
            src = window[: self.window_len - self.pred_len]
            tgt = window[self.window_len - self.pred_len :]
            return (
                torch.from_numpy(src),  # shape: ((w-p), d)
                torch.from_numpy(tgt)   # shape: (p, d)
            )
        else:  # reconstruct
            return (
                torch.from_numpy(window),  # shape: (w, d)
                torch.from_numpy(window)   # same for target
            )
