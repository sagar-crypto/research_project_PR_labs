import glob
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import PROJECT_ROOT, SCALER_TRANSFORMERS_DIR

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
        mode: str = "forecast",  # one of ["forecast", "reconstruct"]
        events_map: dict | None = None,
        label_scope: str = "future"
    ):
        self.paths = glob.glob(pattern)
        if not self.paths:
            raise FileNotFoundError(f"No files match: {pattern}")
        self.events_map = events_map
        self.label_scope = label_scope

        self.window_len = int(sample_rate * (window_ms / 1000.0))
        self.pred_len   = int(sample_rate * (pred_ms    / 1000.0))
        if stride_ms is None:
            self.stride = self.window_len
        else:
            self.stride = int(sample_rate * (stride_ms / 1000.0))

        if self.pred_len >= self.window_len:
            raise ValueError("pred_ms must be smaller than window_ms")

        self.mode = mode
        if self.mode not in {"forecast", "reconstruct", "classify"}:
            raise ValueError("mode must be 'forecast', 'reconstruct', or 'classify'")
        if self.mode == "classify" and self.events_map is None:
            raise ValueError("events_map is required in 'classify' mode")

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
        
        lvl1 = sample_df.columns.get_level_values(1)
        time_mask = lvl1.str.contains("Zeitpunkt", na=False)
        if not time_mask.any():
            raise ValueError("No timestamp column found (level-1 contains 'Zeitpunkt')")
        self.time_col = sample_df.columns[time_mask][0]  # e.g. ('ResultsRepl','Zeitpunkt in s')

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
        scaler_path = f"{SCALER_TRANSFORMERS_DIR}/minmax_scaler_{meas}.pkl"
        print(f"Scaler saved to path {scaler_path}")
        joblib.dump(self.scaler, scaler_path)
        self.scaler_path = scaler_path

        self.cum_counts = np.concatenate(([0], np.cumsum(window_counts))).astype(int)

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, idx):
        file_i = int(np.searchsorted(self.cum_counts, idx, side="right") - 1)
        local  = idx - self.cum_counts[file_i]
        start  = local * self.stride
        path   = self.paths[file_i]

        df = pd.read_parquet(path)

        # --- IMPORTANT: slice rows by position with .iloc to get EXACT window_len rows ---
        row_slice = slice(start, start + self.window_len)
        df_rows = df.iloc[row_slice]

        # features (use loc for columns on the already row-sliced frame)
        feat_df = df_rows.loc[:, self.keep_cols.to_list()]
        feat = feat_df.to_numpy(dtype=np.float32)

        # timestamps (seconds); self.time_col can be a MultiIndex tuple or a flat name
        times = df_rows.loc[:, self.time_col].to_numpy(dtype=np.float32)

        feat = self.scaler.transform(feat)

        if self.mode == "forecast":
            src = feat[: self.window_len - self.pred_len]
            tgt = feat[self.window_len - self.pred_len :]
            # return owning, contiguous tensors
            return (torch.tensor(src, dtype=torch.float32),
                    torch.tensor(tgt, dtype=torch.float32))

        elif self.mode == "reconstruct":
            window = feat
            return (torch.tensor(window, dtype=torch.float32),
                    torch.tensor(window, dtype=torch.float32))

        else:  # "classify"
            # classifier uses encoder context as input
            src = feat[: self.window_len - self.pred_len]

            # choose which part of the window to label
            if self.label_scope == "future":
                times_for_label = times[-self.pred_len:]   # prediction horizon
            else:
                times_for_label = times                    # whole window

            base = os.path.splitext(os.path.basename(path))[0]
            t_ev = self.events_map.get(base, None)

            if t_ev is None:
                y = 0
            else:
                t0, t1 = t_ev
                in_evt = (times_for_label >= t0) & (times_for_label <= t1)
                y = int(in_evt.any())

            return torch.tensor(src, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

