#!/usr/bin/env python3
"""
make_predtrad_npy_zip.py

Read train/test/labels CSVs from an *input zip*, convert to .npy, and pack them
into an *output zip* with names that PredTrAD's customized utils.py expects:

  <entity>_train.npy
  <entity>_test.npy
  <entity>_labels.npy

Usage (example from your paths):
  python make_predtrad_npy_zip.py \
    --in-zip   /home/vault/iwi5/iwi5305h/data_predtrad/myData.zip \
    --entity   sensor1 \
    --out-zip  /home/vault/iwi5/iwi5305h/data_predtrad/myData.npy.zip

IMPORTANT: Your utils.py loads from
  f"/home/vault/iwi5/iwi5305h/data_predtrad/{dataset}.zip"
So either:
  - set out-zip name to match your config's dataset, OR
  - change your config's "dataset" to match the out-zip basename.

By default this script looks for zip members ending with:
  train.csv, test.csv, labels.csv
It auto-detects delimiter (',' vs ';') and takes *numeric* columns only for X.
"""

import argparse
import io
import os
import sys
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _find_member(zf: zipfile.ZipFile, target_names):
    """
    Return the FIRST member whose lowercased path ends with one of target_names.
    """
    names = zf.namelist()
    low = [n.lower() for n in names]
    for t in target_names:
        t = t.lower()
        for n, ln in zip(names, low):
            if ln.endswith("/" + t) or ln == t:
                return n
    return None


def _read_csv_auto(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """
    Try to read a CSV from a zip member with delimiter auto-detection.
    If the first parse yields a single wide column containing ';', retry with sep=';'.
    """
    with zf.open(member, "r") as fh:
        data = fh.read()
    # try default (comma)
    bio = io.BytesIO(data)
    df = pd.read_csv(bio)
    if df.shape[1] == 1:
        # maybe semicolon
        try:
            bio2 = io.BytesIO(data)
            df2 = pd.read_csv(bio2, sep=";")
            if df2.shape[1] > 1:
                return df2
        except Exception:
            pass
    return df


def _ensure_numeric_matrix(df: pd.DataFrame, where: str) -> np.ndarray:
    """
    Keep numeric columns; coerce to float32. Raise if no numeric columns left.
    """
    # Coerce mixed/extension dtypes
    df = df.apply(pd.to_numeric, errors="coerce")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise SystemExit(f"[{where}] No numeric columns found after coercion.")
    X = num.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(X).any():
        raise SystemExit(f"[{where}] All values are non-finite after coercion.")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-zip", required=True, help="Path to input CSV zip (contains train.csv/test.csv/labels.csv)")
    ap.add_argument("--entity", required=True, help="Entity name to use in output filenames, e.g., sensor1")
    ap.add_argument("--out-zip", required=True, help="Path to output NPY zip (this should match your PredTrAD dataset name)")
    ap.add_argument("--train-name", default="train.csv", help="Member name (suffix) for train CSV")
    ap.add_argument("--test-name",  default="test.csv",  help="Member name (suffix) for test CSV")
    ap.add_argument("--labels-name",default="labels.csv",help="Member name (suffix) for labels CSV")
    ap.add_argument("--align", default="strict",
                choices=["strict","min","trim-labels","trim-test","pad-labels-zero"],
                help="How to handle length mismatches between test.csv and labels.csv")
    args = ap.parse_args()

    in_zip  = Path(args.in_zip)
    out_zip = Path(args.out_zip)
    entity  = args.entity

    if not in_zip.exists():
        raise SystemExit(f"Input zip not found: {in_zip}")

    with zipfile.ZipFile(in_zip, "r") as z:
        m_train  = _find_member(z, [args.train_name])
        m_test   = _find_member(z, [args.test_name])
        m_labels = _find_member(z, [args.labels_name])

        if not m_train:
            raise SystemExit(f"Could not find '{args.train_name}' inside {in_zip.name}")
        if not m_test:
            raise SystemExit(f"Could not find '{args.test_name}' inside {in_zip.name}")
        if not m_labels:
            raise SystemExit(f"Could not find '{args.labels_name}' inside {in_zip.name}")

        print(f"Found in zip -> train: {m_train} | test: {m_test} | labels: {m_labels}")

        df_train  = _read_csv_auto(z, m_train)
        df_test   = _read_csv_auto(z, m_test)
        df_labels = _read_csv_auto(z, m_labels)

    # Build X matrices (numeric only)
    X_train = _ensure_numeric_matrix(df_train,  "train.csv")
    X_test  = _ensure_numeric_matrix(df_test,   "test.csv")

    # Build y vector (labels can be headerless or have 1st col = y)
    if df_labels.shape[1] == 1:
        y = df_labels.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)
    else:
        # If a headered file slipped in, try first column and warn
        print("[labels.csv] Detected multiple columns; using the first column as labels.")
        y = df_labels.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)

    n_test = X_test.shape[0]
    n_labels = len(y)

    if n_labels != n_test:
        print(f"[warn] Label length ({n_labels}) != test rows ({n_test}). align={args.align}")

        if args.align == "strict":
            raise SystemExit(f"Label length ({n_labels}) != test rows ({n_test}). "
                            f"Ensure labels.csv aligns 1:1 with test.csv rows.")

        # drop any NaNs from labels and re-count
        y = y[~np.isnan(y)]
        n_labels = len(y)

        if args.align in ("min",):
            n = min(n_test, n_labels)
            X_test = X_test[:n]
            y = y[:n]

        elif args.align == "trim-labels":
            if n_labels < n_test:
                # pad not allowed in this mode
                raise SystemExit("labels shorter than test; use --align min or --align pad-labels-zero")
            y = y[:n_test]

        elif args.align == "trim-test":
            if n_test < n_labels:
                raise SystemExit("test shorter than labels; use --align min")
            X_test = X_test[:n_labels]

        elif args.align == "pad-labels-zero":
            if n_labels > n_test:
                y = y[:n_test]
            elif n_labels < n_test:
                pad = np.zeros((n_test - n_labels,), dtype=y.dtype)
                y = np.concatenate([y, pad], axis=0)

    # final safety
    if len(y) != X_test.shape[0]:
        raise SystemExit(f"After align='{args.align}', lengths still differ: labels={len(y)}, test={X_test.shape[0]}")

    # Write to output zip as <entity>_*.npy
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="npy_zip_") as td:
        td = Path(td)
        np.save(td / f"{entity}_train.npy",  X_train)
        np.save(td / f"{entity}_test.npy",   X_test)
        np.save(td / f"{entity}_labels.npy", y)

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z2:
            z2.write(td / f"{entity}_train.npy",  arcname=f"{entity}_train.npy")
            z2.write(td / f"{entity}_test.npy",   arcname=f"{entity}_test.npy")
            z2.write(td / f"{entity}_labels.npy", arcname=f"{entity}_labels.npy")

    print("Wrote:", out_zip)
    print("Shapes -> train:", X_train.shape, "| test:", X_test.shape, "| labels:", y.shape)


if __name__ == "__main__":
    main()
