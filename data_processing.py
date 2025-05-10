import glob, numpy as np, pandas as pd, pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class StreamingWindowDataset(Dataset):
    def __init__(
        self,
        pattern: str,
        sample_rate: float,
        window_ms: float = 50.0,    # now in ms
        stride_ms: float = 0.0,    # e.g. 50 for non-overlap, 25 for 50% overlap
        feature_min: float = 0.0,
        feature_max: float = 1.0,
    ):
        # discover your Parquet files
        self.paths = glob.glob(pattern)
        if not self.paths:
            raise FileNotFoundError(f"No files match: {pattern}")

        # compute lengths in samples
        self.seq_len = int(sample_rate * (window_ms / 1000.0))
        self.stride  = (
            self.seq_len if stride_ms is None
            else int(sample_rate * (stride_ms / 1000.0))
        )

        # fit MinMaxScaler across all data
        self.scaler = MinMaxScaler(feature_range=(int(feature_min), int(feature_max)))
        for p in self.paths:
            df = pd.read_parquet(p)
            arr = df.iloc[:, 1:].values.astype(np.float32)
            self.scaler.partial_fit(arr)

        # figure out total windows per file
        self.cum_windows = [0]
        for p in self.paths:
            num_rows = pq.ParquetFile(p).metadata.num_rows
            w = max(0, (num_rows - self.seq_len)//self.stride + 1)
            self.cum_windows.append(self.cum_windows[-1] + w)
        self.cum_windows = np.array(self.cum_windows, dtype=int)

    def __len__(self):
        return int(self.cum_windows[-1])

    def __getitem__(self, idx):
        # locate the file & window
        file_i = int(np.searchsorted(self.cum_windows, idx, side='right') - 1)
        local  = idx - self.cum_windows[file_i]
        start  = local * self.stride
        p      = self.paths[file_i]

        # read & slice exactly seq_len rows
        df = pd.read_parquet(p)
        window = df.iloc[start : start + self.seq_len, 1:].values.astype(np.float32)

        # normalize & return
        scaled = self.scaler.transform(window)
        return torch.from_numpy(scaled)
