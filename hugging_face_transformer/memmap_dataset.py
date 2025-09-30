# hugging_face_transformer/memmap_dataset.py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy.lib.format as npfmt

def _read_npy_shape(fp: str):
    with open(fp, "rb") as f:
        version = npfmt.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = npfmt.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = npfmt.read_array_header_2_0(f)
        else:
            # fallback (rare)
            shape, fortran, dtype = npfmt.read_array_header_1_0(f)
    return shape  # e.g., (T, C)

class NpyTimeSeriesSimple(Dataset):
    """
    Ultra-simple .npy dataset: never keeps files open.
    __getitem__ opens the file, slices, copies to RAM, and closes.
    """
    def __init__(self, npy_dir: str, window: int, pred: int, stride: int,
                 events_map: dict | None = None, label_scope: str = "context"):
        self.dir = Path(npy_dir)
        self.files = sorted(str(p) for p in self.dir.glob("*.npy"))
        assert self.files, f"No .npy files in {npy_dir}"
        self.window = int(window); self.pred = int(pred); self.stride = int(stride)
        self.events_map = events_map or {}; self.label_scope = label_scope

        self.indices = []  # (fid, start)
        self.cum_counts = [0]

        # read shapes without opening memmaps
        self.n_features = None
        for fid, fp in enumerate(self.files):
            T, C = _read_npy_shape(fp)
            if self.n_features is None: self.n_features = int(C)
            last_start = max(0, T - self.window - self.pred + 1)
            for s in range(0, last_start, self.stride):
                self.indices.append((fid, s))
            self.cum_counts.append(len(self.indices))

        self.seq_len = self.window
        self.pred_len = self.pred

    def __len__(self): return len(self.indices)

    def _label_for(self, stem: str, s: int, e_ctx: int, e_pred: int) -> int:
        # events_map must be in *index units* (start_idx, end_idx)
        evs = self.events_map.get(stem, [])
        if not evs: return 0
        for a, b in evs:
            if self.label_scope == "context":
                if not (b <= s or a >= e_ctx): return 1
            elif self.label_scope == "future":
                if not (b <= e_ctx or a >= e_pred): return 1
            elif self.label_scope == "any_overlap":
                if not (b <= s or a >= e_pred): return 1
        return 0

    def __getitem__(self, i: int):
        fid, s = self.indices[i]
        fp = self.files[fid]
        # open once, slice, copy, close
        arr = np.load(fp, mmap_mode=None)  # loads into memory; fast for slices when files are on SSD
        x_ctx = np.array(arr[s : s + self.window, :], dtype=np.float32, copy=True)
        x_tgt = np.array(arr[s + self.window : s + self.window + self.pred, :], dtype=np.float32, copy=True)
        del arr  # ensure file is closed promptly

        stem = Path(fp).stem
        y = self._label_for(stem, s, s + self.window, s + self.window + self.pred)

        return (
            torch.from_numpy(x_ctx),                       # (L, C)
            torch.from_numpy(x_tgt),                       # (pred, C)
            torch.arange(self.window + self.pred, dtype=torch.float32),  # dummy times
            torch.tensor([y], dtype=torch.float32),
        )
