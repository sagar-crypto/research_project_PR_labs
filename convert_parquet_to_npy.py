# tools/convert_parquet_to_npy.py
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="e.g. /data/replica_*.parquet")
    ap.add_argument("--outdir", required=True, help="where to write .npy files")
    ap.add_argument("--columns", nargs="*", default=None,
                    help="optional list of columns to keep; if omitted, keep all numeric")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(args.pattern))
    assert files, f"No files match: {args.pattern}"

    for fp in files:
        stem = Path(fp).stem
        out = outdir / f"{stem}.npy"
        if out.exists():
            print(f"[skip] {out} exists"); continue

        df = pd.read_parquet(fp)
        # If your Parquet has MultiIndex columns, flatten or select here:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["__".join(map(str, c)) for c in df.columns]

        if args.columns:
            keep = [c for c in args.columns if c in df.columns]
            df = df[keep]
        else:
            # keep only numeric columns
            df = df.select_dtypes(include=[np.number])

        arr = df.to_numpy(dtype=np.float32, copy=False)
        np.save(out, arr)
        print(f"[ok] {stem}: {arr.shape} -> {out}")

if __name__ == "__main__":
    main()
