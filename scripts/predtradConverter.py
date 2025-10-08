from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("ERROR: pyarrow is required. Install with: pip install pyarrow", file=sys.stderr)
    raise

# --------------------------- Helpers ---------------------------

def flatten_cols(cols):
    """Flatten potential MultiIndex column names."""
    if isinstance(cols, pd.MultiIndex):
        return ["_".join([str(x) for x in c if str(x) != "None"]).strip() for c in cols]
    return [str(c) for c in cols]

def guess_time_col(colnames, hint=None):
    """Heuristically find a time column name. Respects a --time_col_hint if given."""
    if hint:
        for c in colnames:
            if c == hint or hint.lower() in str(c).lower():
                return c
    # heuristic patterns
    low = [str(c).lower() for c in colnames]
    cand = []
    for i, c in enumerate(low):
        if ("zeitpunkt" in c and "s" in c) or c == "zeitpunkt in s":
            cand.append(i)
        elif re.search(r"\btime\b|\btimestamp\b|\bzeit\b", c):
            cand.append(i)
    if cand:
        def score(idx):
            c = low[idx]
            if "zeitpunkt in s" in c: return 3
            if "zeitpunkt" in c and "s" in c: return 2
            if "zeitpunkt" in c: return 1
            return 0
        cand.sort(key=score, reverse=True)
        return colnames[cand[0]]
    return None

def load_labels_map(labels_csv):
    """
    Read labels.csv -> dict: filename -> ndarray[[start_raw, end_raw], ...]
    Auto-detects column names similar to [filename,start,end].
    """
    df = pd.read_csv(labels_csv)
    def find(name_options):
        for opt in name_options:
            for c in df.columns:
                if c.lower() == opt:
                    return c
        for opt in name_options:
            for c in df.columns:
                if opt in str(c).lower():
                    return c
        return None
    file_col  = find(["file", "filename", "parquet", "fname"])
    start_col = find(["start", "start_s", "fault_start", "start_time", "start_ms", "start_us", "begin"])
    end_col   = find(["end", "end_s", "fault_end", "end_time", "end_ms", "end_us", "finish"])
    if any(x is None for x in [file_col, start_col, end_col]):
        raise ValueError(f"labels.csv must include filename/start/end-like columns. Found: {list(df.columns)}")
    df = df[[file_col, start_col, end_col]].copy()
    df.columns = ["filename", "start_raw", "end_raw"]
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(str(x)))
    labels = {}
    for fname, sub in df.groupby("filename"):
        s = sub["start_raw"].to_numpy(dtype=float)
        e = sub["end_raw"].to_numpy(dtype=float)
        labels[fname] = np.stack([s, e], axis=1)
    return labels

def scale_labels_to_seconds(arr2, unit="auto"):
    """
    Convert raw label times to seconds.
    unit in {"s","ms","us","ns","auto"}.
    """
    if unit == "s":
        scale = 1.0
    elif unit == "ms":
        scale = 1e-3
    elif unit == "us":
        scale = 1e-6
    elif unit == "ns":
        scale = 1e-9
    elif unit == "auto":
        med = float(np.nanmedian(arr2))
        if med > 1e13:   # nanoseconds-ish
            scale = 1e-9
        elif med > 1e10: # microseconds-ish
            scale = 1e-6
        elif med > 1e7:  # milliseconds-ish
            scale = 1e-3
        else:
            scale = 1.0
    else:
        raise ValueError("labels_unit must be one of s|ms|us|ns|auto")
    return arr2 * scale

def select_numeric_features(df, time_col=None, include_regex=None):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if time_col and (time_col not in num_cols) and (time_col in df.columns):
        num_cols = [time_col] + num_cols
    if include_regex:
        r = re.compile(include_regex)
        num_cols = [c for c in num_cols if r.search(c)]
    return df[num_cols]

@contextmanager
def zip_writer(path: Path):
    import zipfile
    zf = None
    try:
        zf = zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED)
        yield zf
    finally:
        if zf is not None:
            zf.close()

# --------------------------- Core ---------------------------

def convert_streaming(
    data_dir,
    labels_csv,
    out_zip,
    train_frac_healthy=0.7,
    sample_rate=None,
    include_regex=None,
    dropna=False,
    batch_size=200_000,
    time_col_hint=None,
    labels_unit="auto",
    labels_relative="none",   # none | minus_first_time | custom
    labels_custom_offset=0.0,
    seed=42,
    max_files=None,
    dry_run=False,
):
    data_dir = Path(data_dir)
    files = sorted([p for p in data_dir.glob("*.parquet")])
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No .parquet files in {data_dir}")

    labels_map = load_labels_map(labels_csv)
    set_lbl = set(labels_map.keys())
    set_parq = set(p.name for p in files)

    missing_in_labels = sorted(set_parq - set_lbl)
    missing_on_disk   = sorted(set_lbl - set_parq)

    print(f"Parquet files: {len(set_parq)} | Label entries: {len(set_lbl)}")
    print(f"Files present as parquet but missing from labels.csv: {len(missing_in_labels)}")
    print(f"Files referenced in labels.csv but not found as parquet: {len(missing_on_disk)}")
    if missing_in_labels:
        print("  e.g.,", missing_in_labels[:5])
    if missing_on_disk:
        print("  e.g.,", missing_on_disk[:5])

    # temp dir for incremental CSVs
    workdir = Path(tempfile.mkdtemp(prefix="predtrad_"))
    train_csv = workdir / "train.csv"
    test_csv  = workdir / "test.csv"
    lab_csv   = workdir / "labels.csv"

    rng = np.random.default_rng(seed)

    header_written = False
    n_train = n_test = n_test_faults = 0
    feat_names = None
    used_files = []
    files_without_intervals = []

    f_train = open(train_csv, "w", encoding="utf-8", newline="")
    f_test  = open(test_csv, "w", encoding="utf-8", newline="")
    f_lab   = open(lab_csv,  "w", encoding="utf-8", newline="")

    try:
        for i, fp in enumerate(files, start=1):
            base = fp.name
            used_files.append(base)

            intervals_raw = labels_map.get(base, np.empty((0,2), dtype=float))
            if intervals_raw.size == 0:
                files_without_intervals.append(base)

            # Stream batches
            pf = pq.ParquetFile(str(fp))

            # Determine time column name from small probe batch
            tcol_name = None
            try:
                probe = next(pf.iter_batches(batch_size=min(10_000, batch_size)))
                tb = pa.Table.from_batches([probe])
                df_probe = tb.to_pandas(types_mapper=pd.ArrowDtype)
                # flatten potential grouped/multiindex headers like ('ResultsRepl','Zeitpunkt in s')
                df_probe.columns = flatten_cols(df_probe.columns)

                # Prefer explicit substring hint first (case-insensitive)
                if time_col_hint:
                    for c in df_probe.columns:
                        if time_col_hint.lower() in str(c).lower():
                            tcol_name = c
                            break

                # Fallback heuristic on flattened names
                if tcol_name is None:
                    tcol_name = guess_time_col(df_probe.columns, hint=time_col_hint)
            except StopIteration:
                continue  # empty file? skip

            if tcol_name is None and sample_rate is None:
                raise ValueError(f"{base}: missing time column (after flatten) & sample_rate")

            first_time_of_file = None
            intervals_sec = scale_labels_to_seconds(intervals_raw.astype(float), unit=labels_unit)

            row_offset = 0
            for batch in pf.iter_batches(batch_size=batch_size):
                tb = pa.Table.from_batches([batch])
                df = tb.to_pandas(types_mapper=pd.ArrowDtype)  # fast

                df.columns = flatten_cols(df.columns)

                # build times
                if tcol_name and (tcol_name in df.columns):
                    times = df[tcol_name].to_numpy()
                    if times.size >= 2 and np.any(np.diff(times) < 0):
                        order = np.argsort(times)
                        df    = df.iloc[order]
                        times = times[order]
                else:
                    if sample_rate is None:
                        raise ValueError(f"{base}: missing time column & sample_rate")
                    start_t = row_offset / float(sample_rate)
                    times = start_t + np.arange(len(df), dtype=np.float64) / float(sample_rate)

                if first_time_of_file is None and len(times) > 0:
                    first_time_of_file = float(times[0])

                feat_df = select_numeric_features(df, time_col=tcol_name, include_regex=include_regex)
                if tcol_name and (tcol_name in feat_df.columns):
                    feat_df = feat_df.drop(columns=[tcol_name])
                feat_df = feat_df.apply(pd.to_numeric, errors="coerce")

                arr = feat_df.to_numpy(dtype="float64", copy=False)

                nan_mask = np.isnan(arr)
                inf_mask = np.isinf(arr)
                if nan_mask.any() or inf_mask.any():
                    if dropna:
                        good = ~(nan_mask.any(axis=1) | inf_mask.any(axis=1))
                        feat_df = feat_df.loc[good]
                        times   = times[good]
                        arr     = feat_df.to_numpy(dtype="float64", copy=False)
                    else:
                        bad_cols = feat_df.columns[(np.isnan(arr).any(axis=0) | np.isinf(arr).any(axis=0))]
                        raise ValueError(f"{base}: found NaN/Inf in columns {list(bad_cols)}; re-run with --dropna or clean data")


                # labels for this batch
                y = np.zeros(len(times), dtype=np.int8)
                if intervals_sec.size > 0 and len(times) > 0:
                    if labels_relative == "minus_first_time":
                        start = intervals_sec[:,0] - first_time_of_file
                        end   = intervals_sec[:,1] - first_time_of_file
                    elif labels_relative == "custom":
                        start = intervals_sec[:,0] + float(labels_custom_offset)
                        end   = intervals_sec[:,1] + float(labels_custom_offset)
                    else:
                        start = intervals_sec[:,0].copy()
                        end   = intervals_sec[:,1].copy()

                    s = np.minimum(start, end)
                    e = np.maximum(start, end)
                    # OR masks per interval
                    for s_i, e_i in zip(s, e):
                        if np.isnan(s_i) or np.isnan(e_i):
                            continue
                        m = (times >= s_i) & (times <= e_i)
                        y[m] = 1

                # header/schema
                if not header_written:
                    feat_names = feat_df.columns.tolist()
                    f_train.write(",".join(feat_names) + "\n")
                    f_test.write(",".join(feat_names) + "\n")
                    header_written = True
                else:
                    if feat_df.columns.tolist() != feat_names:
                        raise RuntimeError(f"{base}: Feature columns differ from previous files.")

                healthy = (y == 0)
                # Bernoulli split for healthy rows only; all faults -> test
                to_train = healthy & (rng.random(size=healthy.size) < float(train_frac_healthy))
                to_test  = ~to_train  # includes all faults + remaining healthy

                if np.any(to_train):
                    feat_df.loc[to_train].to_csv(f_train, index=False, header=False)
                    n_train += int(np.sum(to_train))
                if np.any(to_test):
                    feat_df.loc[to_test].to_csv(f_test, index=False, header=False)
                    pd.DataFrame(y[to_test]).to_csv(f_lab, index=False, header=False)
                    n_test += int(np.sum(to_test))
                    n_test_faults += int(np.sum(y[to_test] == 1))

                row_offset += len(df)

            if (i % 50 == 0) or (i == len(files)):
                print(f"[{i}/{len(files)}] processed: {base}")

    finally:
        f_train.close()
        f_test.close()
        f_lab.close()

    meta = {
        "features": feat_names or [],
        "n_features": int(len(feat_names or [])),
        "n_rows_train": int(n_train),
        "n_rows_test": int(n_test),
        "n_fault_test": int(n_test_faults),
        "train_frac_healthy": float(train_frac_healthy),
        "assembled_from_files": used_files,
        "files_without_intervals_in_labels_csv": files_without_intervals,
        "notes": {
            "split": "Train has only healthy rows; Test has faults + remaining healthy rows.",
            "labels_unit": labels_unit,
            "labels_relative": labels_relative,
        }
    }

    if dry_run:
        print("DRY-RUN complete (no ZIP written). Summary:")
        print(json.dumps(meta, indent=2))
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
        return

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="predtrad_zip_") as td:
        meta_path = Path(td) / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        with zip_writer(out_zip) as z:
            z.write(str(train_csv), arcname="train.csv")
            z.write(str(test_csv),  arcname="test.csv")
            z.write(str(Path(lab_csv)),   arcname="labels.csv")
            z.write(str(meta_path), arcname="meta.json")

    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass

    print("ZIP:", str(out_zip))
    print(f"Train rows: {n_train:,} | Test rows: {n_test:,} | Test faults: {n_test_faults:,}")
    print(f"Files with no label intervals: {len(files_without_intervals)} (treated as all-healthy)")
    if files_without_intervals:
        print("  e.g.,", files_without_intervals[:3])

# --------------------------- Diagnostics ---------------------------

def diagnose(data_dir, labels_csv, max_files=20, time_col_hint=None, sample_rate=None, labels_unit="auto"):
    data_dir = Path(data_dir)
    files = sorted([p for p in data_dir.glob("*.parquet")])
    if max_files:
        files = files[:max_files]

    labels_map = load_labels_map(labels_csv)

    set_parq = set(p.name for p in files)
    set_lbl  = set(labels_map.keys())

    print("Parquet files (scoped):", len(set_parq))
    print("Label entries (all):   ", len(set_lbl))
    print("Missing in labels:", len(set_parq - set_lbl))
    print("Missing on disk:", len(set_lbl - set_parq))

    matched = sorted(list(set_parq & set_lbl))[:min(5, len(set_parq & set_lbl))]
    print("Checking samples:", matched)

    for name in matched:
        fp = data_dir / name
        pf = pq.ParquetFile(str(fp))
        try:
            tb0 = next(pf.iter_batches(batch_size=50_000))
        except StopIteration:
            print(f"{name}: empty parquet?")
            continue
        cols = [str(n) for n in pa.Table.from_batches([tb0]).schema.names]
        tcol = guess_time_col(cols, hint=time_col_hint)
        if tcol is None and sample_rate is None:
            print(f"{name}: ⚠ no time col; provide --sample_rate")
            continue
        tb = pa.Table.from_batches([tb0]).to_pandas()
        tb.columns = flatten_cols(tb.columns)
        if tcol is not None:
            times = tb[tcol].to_numpy()
        else:
            times = np.arange(len(tb), dtype=float) / float(sample_rate)
        if len(times) < 2:
            print(f"{name}: too few rows")
            continue
        dt = np.diff(times)
        dt_med = float(np.median(dt))
        if dt_med > 0.5: tun = "seconds"
        elif dt_med > 5e-4: tun = "milliseconds-ish"
        elif dt_med > 5e-7: tun = "microseconds-ish"
        else: tun = "nanoseconds-ish"
        its = labels_map[name].astype(float)
        its_sec = scale_labels_to_seconds(its, unit=labels_unit)
        tmin, tmax = float(times.min()), float(times.max())
        hits = 0
        for s,e in its_sec:
            s,e = (min(s,e), max(s,e))
            if e >= tmin and s <= tmax:
                hits += 1
        print(f"{name}: time unit ~{tun}, time range [{tmin:.3g},{tmax:.3g}], "
              f"intervals in labels:{len(its)}, overlapping intervals (no offset): {hits}")

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert parquet + interval labels to PredTrAD ZIP (streaming).")
    ap.add_argument("--data_dir", required=True, help="Folder with .parquet files")
    ap.add_argument("--labels_csv", required=True, help="CSV with filename,start,end")
    ap.add_argument("--out_zip", required=False, default="mydata.zip")

    ap.add_argument("--train_frac_healthy", type=float, default=0.7, help="Fraction of healthy rows to send to train")
    ap.add_argument("--sample_rate", type=float, default=None, help="Hz, used only if no time column")
    ap.add_argument("--include_regex", type=str, default=None, help="Regex to keep only some feature columns")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with NaN/Inf")
    ap.add_argument("--batch_size", type=int, default=200_000, help="Parquet batch size")
    ap.add_argument("--time_col_hint", type=str, default=None, help="Exact or substring hint for time column")
    ap.add_argument("--labels_unit", type=str, default="auto", choices=["s","ms","us","ns","auto"])
    ap.add_argument("--labels_relative", type=str, default="none", choices=["none","minus_first_time","custom"])
    ap.add_argument("--labels_custom_offset", type=float, default=0.0, help="Seconds to add if labels_relative=custom")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of files for testing")
    ap.add_argument("--dry_run", action="store_true", help="Traverse and summarize without writing ZIP")
    ap.add_argument("--diagnose", action="store_true", help="Run filename/unit diagnostics and exit")

    args = ap.parse_args()

    if args.diagnose:
        diagnose(args.data_dir, args.labels_csv, max_files=args.max_files,
                 time_col_hint=args.time_col_hint, sample_rate=args.sample_rate,
                 labels_unit=args.labels_unit)
        return

    convert_streaming(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        out_zip=args.out_zip,
        train_frac_healthy=args.train_frac_healthy,
        sample_rate=args.sample_rate,
        include_regex=args.include_regex,
        dropna=args.dropna,
        batch_size=args.batch_size,
        time_col_hint=args.time_col_hint,
        labels_unit=args.labels_unit,
        labels_relative=args.labels_relative,
        labels_custom_offset=args.labels_custom_offset,
        seed=args.seed,
        max_files=args.max_files,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
