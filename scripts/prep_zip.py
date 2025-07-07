import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


import argparse
import glob
import os
import shutil
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATH, SCALER_PREDTRAD_DIR, DATA_PATH_PREDTRAD


def collect_and_scale(
    paths,
    level1_filter,
    feature_range,
    save_scaler_path=None,
    sample_frac: float = 1.0,   # <--- new: fraction of rows to keep (0 < sample_frac ≤ 1)
    random_state: int = 42
):
    """
    Reads all parquet files in `paths`, optionally samples a fraction of each,
    fits a MinMaxScaler, then returns the stacked scaled data and columns.

    Args:
      paths            – list of file paths (parquet)
      level1_filter    – string filter for level-1 column names
      feature_range    – tuple(min, max) for MinMaxScaler
      save_scaler_path – optional path to dump the fitted scaler
      sample_frac      – fraction of rows to sample from each file (1.0 = all)
      random_state     – seed for reproducible sampling
    Returns:
      all_data (np.ndarray of shape [total_rows * sample_frac, n_features]),
      keep_cols (list of tuples for MultiIndex)
    """
    # 1) Determine keep_cols based on MultiIndex filtering
    df0 = pd.read_parquet(paths[0])
    cols = df0.columns  # MultiIndex

    lvl0 = cols.get_level_values(0)
    mask0 = lvl0.str.contains("Cub") & lvl0.str.contains("Line")
    if level1_filter:
        lvl1 = cols.get_level_values(1)
        mask1 = lvl1.str.contains(level1_filter)
    else:
        mask1 = np.ones(len(cols), dtype=bool)
    keep = mask0 & mask1
    keep_cols = [c for i,c in enumerate(cols) if keep[i]]
    if not keep_cols:
        raise ValueError(f"No columns match filters: level1_filter='{level1_filter}'")

    # 2) Fit scaler over sampled data
    scaler = MinMaxScaler(feature_range=feature_range)
    for path in paths:
        df = pd.read_parquet(path)[keep_cols]
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
        scaler.partial_fit(df.values.astype(np.float32))

    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)

    # 3) Collect & transform sampled data
    all_data = []
    for path in paths:
        df = pd.read_parquet(path)[keep_cols]
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
        arr = scaler.transform(df.values.astype(np.float32))
        all_data.append(arr)

    return np.vstack(all_data), keep_cols

def main(args):
    # 1) Discover parquet files

    pattern     = os.path.join(DATA_PATH, "*.parquet")
    save_scaler = os.path.join(
        SCALER_PREDTRAD_DIR,
        f"minmax_scaler_{args.entity}.pkl"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match pattern: {pattern}")

    # 2) Filter, scale, and collect data
    feature_range = (args.fr_min, args.fr_max)
    data, keep_cols = collect_and_scale(
        paths,
        args.level1_filter,
        feature_range,
        save_scaler_path=save_scaler,
        sample_frac=0.03,
        random_state=123
    )
    print(f"Using columns ({len(keep_cols)}): {keep_cols}")

    # 3) Split into train/test
    X_train, X_test = train_test_split(
        data,
        test_size=args.test_size,
        shuffle=args.shuffle,
        random_state=args.random_state
    )
    # Dummy labels (zeros) for unsupervised training
    y_test = np.zeros((X_test.shape[0],), dtype=np.int32)

    # 4) Prepare temporary directory for split files
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
    os.makedirs(args.temp_dir, exist_ok=True)

    # 5) Save .npy files with entity prefix
    entity = args.entity
    np.save(os.path.join(args.temp_dir, f"{entity}_train.npy"),  X_train)
    np.save(os.path.join(args.temp_dir, f"{entity}_test.npy"),   X_test)
    np.save(os.path.join(args.temp_dir, f"{entity}_labels.npy"), y_test)

    # 6) Zip into data/<dataset>.zip (or custom output-dir)
    os.makedirs(DATA_PATH_PREDTRAD, exist_ok=True)
    zip_path = os.path.join(DATA_PATH_PREDTRAD, f"{args.dataset}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(args.temp_dir)):
            zf.write(os.path.join(args.temp_dir, fname), arcname=fname)

    # 7) Clean up temporary directory
    shutil.rmtree(args.temp_dir)

    print(f"✅ Archive created: {zip_path}")
    print("Contents:", sorted(zf.namelist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Parquet data for PredTrAD using config.py"
    )
    parser.add_argument(
        "--entity", required=True,
        help="Entity name prefix (e.g. sensor1)"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset zip name without .zip (e.g. mydata)"
    )
    parser.add_argument(
        "--level1-filter", default="",
        help="Substring to filter level-1 column names (optional)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for the test set (default: 0.2)"
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle before splitting"
    )
    parser.add_argument(
        "--random-state", type=int, default=0,
        help="Random seed for splitting"
    )
    parser.add_argument(
        "--fr-min", type=float, default=0.0,
        help="Min for MinMaxScaler feature_range"
    )
    parser.add_argument(
        "--fr-max", type=float, default=1.0,
        help="Max for MinMaxScaler feature_range"
    )
    parser.add_argument(
        "--temp-dir", default="temp_split",
        help="Temporary directory for split files"
    )
    args = parser.parse_args()
    main(args)
