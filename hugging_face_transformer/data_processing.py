import glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq  # for fast row counts
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import hashlib

from config import SCALER_TRANSFORMERS_DIR


class ParquetTimeSeriesDataset(Dataset):
    """
    Streams (ctx, tgt, times) windows from Parquet files efficiently.

    Key properties:
      - Uses a canonical, ordered feature list common to ALL files (Cub+Line only, not time)
      - Detects the time column per file by searching for 'Zeitpunkt' (case-insensitive)
      - Scales features once per file and caches (feat_mat, time_arr) to avoid re-reading
      - seq_len = window_len - pred_len (ctx is encoder length; tgt is decoder length)
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
        # 1) Discover files & window sizes
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files match: {pattern}")

        self.sample_rate = float(sample_rate)
        self.pred_len = int(pred_ms / 1000.0 * sample_rate)
        ctx_total = int(window_ms / 1000.0 * sample_rate)
        self.seq_len = ctx_total - self.pred_len
        if self.seq_len <= 0:
            raise ValueError("window_ms must be > pred_ms")
        self.stride = int(stride_ms / 1000.0 * sample_rate)
        if self.stride <= 0:
            raise ValueError("stride_ms must be > 0")

        # small per-process LRU cache {file_i: (feat_mat_scaled, time_arr)}
        self._cache = {}
        self._cache_order = []
        self._max_cache_files = max(16, min(len(self.files), 64))   # CHANGED: was 2

        # 2) Build a CANONICAL feature list common to ALL files (flattened)
        #    Rule: keep columns whose flat name contains 'Cub' and 'Line', and NOT 'Zeitpunkt'
        common = None
        order_ref = None
        time_pat = "zeitpunkt"  # used for detection later

        for i, fn in enumerate(self.files):
            df = pd.read_parquet(fn)
            if isinstance(df.columns, pd.MultiIndex):
                flat_cols = pd.Index([f"{a}_{b}" for a, b in df.columns])
            else:
                flat_cols = pd.Index([str(c) for c in df.columns])

            tmask = flat_cols.str.contains(time_pat, case=False, na=False)
            fmask = flat_cols.str.contains("Cub") & flat_cols.str.contains("Line") & (~tmask)
            feats_i = flat_cols[fmask]

            if order_ref is None:
                order_ref = list(feats_i)      # preserve first-file order
                common = set(feats_i)
            else:
                common &= set(feats_i)

        self.feat_cols = [c for c in (order_ref or []) if c in (common or set())]
        self.n_features = len(self.feat_cols)
        if self.n_features == 0:
            raise ValueError("No common 'Cub'+'Line' feature columns across files.")

        # CHANGED: load-or-fit scaler ONCE; cache keyed by feat_cols + feature_range
        sig = hashlib.sha1((','.join(self.feat_cols) + repr(feature_range)).encode()).hexdigest()[:8]
        os.makedirs(SCALER_TRANSFORMERS_DIR, exist_ok=True)
        self.scaler_path = str(
            Path(SCALER_TRANSFORMERS_DIR) / f"minmax_scaler_ts_transformer_n{self.n_features}_{sig}.pkl"
        )

        self.scaler = None
        if os.path.exists(self.scaler_path):
            try:
                payload = joblib.load(self.scaler_path)
                if (isinstance(payload, dict)
                    and payload.get("feat_cols") == self.feat_cols
                    and payload.get("feature_range") == feature_range):
                    self.scaler = payload["scaler"]
            except Exception:
                self.scaler = None

        if self.scaler is None:
            # Fit ONCE across all files, then cache {scaler, feat_cols, feature_range}
            self.scaler = MinMaxScaler(feature_range=feature_range)
            for fn in self.files:
                df = pd.read_parquet(fn)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [f"{a}_{b}" for a, b in df.columns]
                # defensive check (shouldn’t trigger because we intersected)
                missing = [c for c in self.feat_cols if c not in df.columns]
                if missing:
                    raise KeyError(f"{fn} missing expected features: {missing[:5]}...")
                arr = df.loc[:, self.feat_cols].to_numpy(dtype=np.float32)
                self.scaler.partial_fit(arr)

            payload = {
                "scaler": self.scaler,
                "feat_cols": self.feat_cols,
                "feature_range": feature_range,
            }
            tmp = self.scaler_path + ".tmp"
            joblib.dump(payload, tmp)
            os.replace(tmp, self.scaler_path)  # atomic rename

        # store time detection pattern
        self._time_pat = time_pat

        # 4) Precompute window counts for __len__ using parquet metadata (no data read)
        counts = []
        for fn in self.files:
            n = pq.ParquetFile(fn).metadata.num_rows
            counts.append(max(0, (n - self.seq_len - self.pred_len) // self.stride + 1))  # lags=[0]
        self.cum_counts = np.concatenate(([0], np.cumsum(counts))).astype(int)

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, idx):
        # figure out which file and where inside it
        file_i = int(np.searchsorted(self.cum_counts, idx, side="right") - 1)
        local = idx - self.cum_counts[file_i]
        start = local * self.stride

        feat_mat, time_arr = self._get_file_arrays(file_i)

        ctx = feat_mat[start : start + self.seq_len]
        tgt = feat_mat[start + self.seq_len : start + self.seq_len + self.pred_len]
        times = time_arr[start : start + self.seq_len]

        return (
            torch.from_numpy(ctx).float(),                    # (seq_len, n_features)
            torch.from_numpy(tgt).float(),                    # (pred_len, n_features)
            torch.from_numpy(times).float().unsqueeze(-1),    # (seq_len, 1)
        )

    # ---------- internals ----------

    def _get_file_arrays(self, file_i: int):
        """Return (scaled_feat_mat, time_arr) for file_i, using tiny LRU cache."""
        if file_i in self._cache:
            return self._cache[file_i]

        fn = self.files[file_i]
        df = pd.read_parquet(fn)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{a}_{b}" for a, b in df.columns]

        cols_idx = pd.Index([str(c) for c in df.columns])

        # detect a time column per file (fallback to synthetic time if absent)
        tmask = cols_idx.str.contains(self._time_pat, case=False, na=False)
        if tmask.any():
            time_col = cols_idx[tmask][0]
            time_arr = df[time_col].to_numpy(dtype=np.float32)
        else:
            # synthetic time in seconds, evenly spaced by sample_rate
            time_arr = (np.arange(len(df), dtype=np.float32) / self.sample_rate)

        # strictly select the canonical features in the canonical order
        missing = [c for c in self.feat_cols if c not in df.columns]
        if missing:
            # should not happen because we intersected; but be explicit if a file deviates
            raise KeyError(f"File {fn} is missing expected features: {missing[:8]}...")

        feat_mat = df.loc[:, self.feat_cols].to_numpy(dtype=np.float32)

        # scale once per file
        feat_mat = self.scaler.transform(feat_mat)

        # cache (tiny LRU)
        self._cache[file_i] = (feat_mat, time_arr)
        self._cache_order.append(file_i)
        if len(self._cache_order) > self._max_cache_files:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return feat_mat, time_arr
