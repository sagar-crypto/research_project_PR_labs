# transformer/utils_label.py
import os
import pandas as pd
from typing import Dict, Tuple

def make_event_map(labels_csv_path: str) -> Dict[str, Tuple[float, float]]:
    # 1) Read as comma-separated (default); don't force semicolons
    df = pd.read_csv(labels_csv_path)

    # 2) Normalize/validate columns we actually have: filename, start_s, end_s
    required = {"filename", "start_s", "end_s"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Labels CSV missing columns {missing}. Found: {list(df.columns)}")

    # 3) Base name without extension (e.g., "replica_0.parquet" -> "replica_0")
    def base(p: str) -> str:
        p = str(p).replace("\\", "/")
        return os.path.splitext(os.path.basename(p))[0]

    df["base"] = df["filename"].map(base)

    # 4) Ensure numeric seconds
    df["start_s"] = pd.to_numeric(df["start_s"], errors="coerce")
    df["end_s"]   = pd.to_numeric(df["end_s"],   errors="coerce")
    df = df.dropna(subset=["start_s", "end_s"])

    # 5) If multiple intervals per file exist, merge into [min_start, max_end]
    events = {}
    for k, g in df.groupby("base", sort=False):
        s = float(g["start_s"].min())
        e = float(g["end_s"].max())
        events[k] = (s, e)

    return events

