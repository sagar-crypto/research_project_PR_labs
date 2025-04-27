import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class StreamingWindowDataset(Dataset):
    def __init__(
        self,
        pattern: str,
        seq_len: int = 128,
        stride: int = 16,
        feature_min: float = 0.0,
        feature_max: float = 1.0,
        sep: str = ';',
        header: int = 1,
        encoding: str = 'latin1',
        chunksize: int = 50_000
    ):
        # 1) discover files
        self.paths = glob.glob(pattern)
        if not self.paths:
            raise FileNotFoundError(f"No files match: {pattern}")
        self.seq_len   = seq_len
        self.stride    = stride
        self.feature_min, self.feature_max = feature_min, feature_max
        self.sep, self.header, self.encoding = sep, header, encoding

        # 2) compute per-column global min/max by streaming through each file in chunks
        mins, maxs = None, None
        for p in self.paths:
            for chunk in pd.read_csv(p,
                                     sep=sep,
                                     header=header,
                                     encoding=encoding,
                                     engine='python',
                                     chunksize=chunksize):
                arr = chunk.values.astype(np.float32)
                if mins is None:
                    mins, maxs = arr.min(axis=0), arr.max(axis=0)
                else:
                    mins, maxs = np.minimum(mins, arr.min(axis=0)), \
                                 np.maximum(maxs, arr.max(axis=0))
        self.data_min = mins
        self.data_max = maxs

        # 3) figure out how many windows each file will yield
        self.windows_per_file = []
        self.cum_windows = [0]
        for p in self.paths:
            # total rows in file = lines − header
            total_rows = sum(1 for _ in open(p, 'r', encoding=encoding)) - header
            w = max(0, (total_rows - seq_len) // stride + 1)
            self.windows_per_file.append(w)
            self.cum_windows.append(self.cum_windows[-1] + w)
        self.cum_windows = np.array(self.cum_windows)

    def __len__(self):
        return int(self.cum_windows[-1])

    def __getitem__(self, idx):
        # 1) map global idx → which file and which window within it
        file_i = np.searchsorted(self.cum_windows, idx, side='right') - 1
        local_idx = idx - self.cum_windows[file_i]
        start_row = local_idx * self.stride

        # 2) read exactly seq_len rows from that file
        df = pd.read_csv(self.paths[file_i],
                         sep=self.sep,
                         header=self.header,
                         encoding=self.encoding,
                         engine='python',
                         skiprows=self.header + start_row,
                         nrows=self.seq_len)
        arr = df.values.astype(np.float32)

        # 3) min-max scale into [feature_min, feature_max]
        #    avoid div by zero
        denom = (self.data_max - self.data_min + 1e-6)
        scaled = (arr - self.data_min) / denom
        scaled = scaled * (self.feature_max - self.feature_min) + self.feature_min

        return torch.from_numpy(scaled)  # shape: (seq_len, n_channels)

