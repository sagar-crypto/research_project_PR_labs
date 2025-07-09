import sys
from pathlib import Path

# ensure project_root on sys.path so we can import config.py
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

def build_row_labels(n_rows: int, events: pd.DataFrame, sample_rate: float) -> np.ndarray:
    """
    Given the number of rows in a parquet file, and a DataFrame of events
    with columns ['t_evnt_start','t_evnt_end'] in seconds, produce a
    0/1 array of length n_rows marking anomaly intervals.
    """
    labels = np.zeros(n_rows, dtype=np.int32)
    for _, ev in events.iterrows():
        start = int(ev.t_evnt_start * sample_rate)
        end   = int(ev.t_evnt_end   * sample_rate)
        start = max(0, min(start, n_rows))
        end   = max(0, min(end,   n_rows))
        labels[start:end] = 1
    return labels

def collect_scale_label(
    paths: list,
    labels_df: pd.DataFrame,
    sample_rate: float,
    level1_filter: str,
    feature_range: tuple,
    save_scaler_path: str,
    sample_frac: float = 1.0,
    random_state: int = 42
):
    """
    Reads all parquet files in `paths`, filters columns, builds per-row labels,
    fits a MinMaxScaler on sampled data, applies scaling, and stacks data & labels.
    Returns:
      data_all (np.ndarray): shape [total_rows, n_features]
      labels_all (np.ndarray): shape [total_rows,]
      keep_cols (list): list of column tuples kept
    """
    # 1) Determine which columns to keep
    df0 = pd.read_parquet(paths[0])
    cols = df0.columns
    lvl0 = cols.get_level_values(0)
    mask0 = lvl0.str.contains("Cub") & lvl0.str.contains("Line")
    if level1_filter:
        lvl1 = cols.get_level_values(1)
        mask1 = lvl1.str.contains(level1_filter)
    else:
        mask1 = np.ones(len(cols), dtype=bool)
    keep = mask0 & mask1
    keep_cols = [c for i, c in enumerate(cols) if keep[i]]
    if not keep_cols:
        raise ValueError(f"No columns match level1_filter='{level1_filter}'")

    # 2) Fit scaler on sampled data
    scaler = MinMaxScaler(feature_range=feature_range)
    for p in paths:
        df = pd.read_parquet(p)[keep_cols]
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
        scaler.partial_fit(df.values.astype(np.float32))
    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)

    # 3) Transform data and build labels
    data_list = []
    label_list = []
    for p in paths:
        df = pd.read_parquet(p)[keep_cols]
        stem = Path(p).stem
        events = labels_df[labels_df["file_stem"] == stem]
        labels_full = build_row_labels(len(df), events, sample_rate)
        n = len(df)
        if sample_frac < 1.0:
            k   = int(n * sample_frac)
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n, size=k, replace=False)   # positions 0…n-1
            df  = df.iloc[idx].reset_index(drop=True)
            labels_full = labels_full[idx]
        else:
            df = df.reset_index(drop=True)
        arr = scaler.transform(df.values.astype(np.float32))
        data_list.append(arr)
        label_list.append(labels_full)

    data_all = np.vstack(data_list)
    labels_all = np.concatenate(label_list)
    return data_all, labels_all, keep_cols

def main():
    parser = argparse.ArgumentParser(description="Prepare Parquet+Labels for PredTrAD")
    parser.add_argument("--entity",       required=True, help="Entity prefix (e.g. sensor1)")
    parser.add_argument("--dataset",      required=True, help="Dataset name (zip without .zip)")
    parser.add_argument("--labels-csv",   required=True, help="Path to labels.csv (semicolon-separated)")
    parser.add_argument("--sample-rate",  type=float, required=True, help="Sampling rate in Hz")
    parser.add_argument("--level1-filter", default="",   help="Filter substring for level-1 columns")
    parser.add_argument("--fr-min",       type=float, default=0.0, help="Min for MinMaxScaler")
    parser.add_argument("--fr-max",       type=float, default=1.0, help="Max for MinMaxScaler")
    parser.add_argument("--sample-frac",  type=float, default=1.0, help="Fraction of rows to sample")
    parser.add_argument("--test-size",    type=float, default=0.2, help="Fraction for test split")
    parser.add_argument("--shuffle",      action="store_true", help="Shuffle before split")
    parser.add_argument("--random-state", type=int,   default=0,   help="Random seed")
    parser.add_argument("--temp-dir",     default="temp_split", help="Temporary directory for .npy files")
    args = parser.parse_args()

    # 1) Load and parse labels.csv
    labels_df = pd.read_csv(args.labels_csv, sep=";")
    labels_df["file_base"] = labels_df["export_file"].apply(lambda p: Path(p).name)
    labels_df["file_stem"] = labels_df["file_base"].apply(lambda n: Path(n).stem)

    # 2) Discover Parquet files
    pattern = os.path.join(DATA_PATH, "*.parquet")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match pattern: {pattern}")

    # 3) Prepare scaler path
    save_scaler = os.path.join(SCALER_PREDTRAD_DIR, f"minmax_scaler_{args.entity}.pkl")

    # 4) Collect data, scale, label
    data_all, labels_all, keep_cols = collect_scale_label(
        paths,
        labels_df,
        args.sample_rate,
        args.level1_filter,
        (args.fr_min, args.fr_max),
        save_scaler,
        sample_frac=args.sample_frac,
        random_state=args.random_state
    )
    print(f"Using columns ({len(keep_cols)}): {keep_cols}")

    # 5) Split into train/test (features & labels together)
    X_train, X_test, y_train, y_test = train_test_split(
        data_all, labels_all,
        test_size=args.test_size,
        shuffle=args.shuffle,
        random_state=args.random_state
    )

    # 6) Save .npy files
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
    os.makedirs(args.temp_dir, exist_ok=True)
    e = args.entity
    np.save(os.path.join(args.temp_dir, f"{e}_train.npy"),  X_train)
    np.save(os.path.join(args.temp_dir, f"{e}_test.npy"),   X_test)
    np.save(os.path.join(args.temp_dir, f"{e}_labels.npy"), y_test)

    # 7) Zip into DATA_PATH_PREDTRAD
    os.makedirs(DATA_PATH_PREDTRAD, exist_ok=True)
    zip_path = os.path.join(DATA_PATH_PREDTRAD, f"{args.dataset}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(args.temp_dir)):
            zf.write(os.path.join(args.temp_dir, fname), arcname=fname)
    shutil.rmtree(args.temp_dir)

    print(f"✅ Archive created: {zip_path}")
    print("Contents:", sorted(zf.namelist()))

if __name__ == "__main__":
    main()
