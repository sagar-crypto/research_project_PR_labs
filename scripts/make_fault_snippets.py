import os
import argparse
import glob
import pandas as pd
from pathlib import Path


def main(args):
    labels = pd.read_csv(args.labels, sep=';')
    os.makedirs(args.output_dir, exist_ok=True)
    count = 0

    for idx, row in labels.iterrows():
        # parse event times (in seconds)
        t0 = float(row['t_evnt_start'])
        t1 = float(row['t_evnt_end'])
        # pre-/post-event margins (now in ms)
        pre_ms  = args.pre_sec
        post_ms = args.post_sec
        # assumes 'export_file' column holds the filename or relative path
        stem  = Path(row['export_file']).stem
        fname = f"{stem}.parquet"
        # search under data_dir
        matches = glob.glob(os.path.join(args.data_dir, '**', fname), recursive=True)
        if not matches:
            print(f"Warning: no file found for {fname}, skipping event {idx}")
            continue
        data_path = matches[0]

        # load raw data
        df = pd.read_parquet(data_path).reset_index(drop=True)
        if isinstance(df.columns, pd.MultiIndex):
            # pick the second level
            df.columns = [
                f"{str(l0).strip()}_{str(l1).strip()}"
                for l0, l1 in df.columns
            ]
        time_col = [c for c in df.columns if "Zeitpunkt in s" in c][0]

        #Mask ±5 ms around t0…t1 (both in ms)
        t0, t1 = float(row["t_evnt_start"]), float(row["t_evnt_end"])
        pre_ms, post_ms = args.pre_sec/ 1000.0, args.post_sec/ 1000.0

        mask = (
            (df[time_col] >= (t0 - pre_ms)) &
            (df[time_col] <= (t1 + post_ms))
        )
        snippet = df.loc[mask]

        #Write out only those filtered rows
        out_name = f"fault_{count}ms.parquet"
        snippet.to_parquet(os.path.join(args.output_dir, out_name), index=False)
        count+=1
        print(f"Wrote {out_name}: {len(snippet)} rows spanning {snippet[time_col].min():.3f}-{snippet[time_col].max():.3f} ms")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Make fault-centered data snippets")
    p.add_argument('--labels',      required=True, help='CSV with columns t_evnt_start,t_evnt_end,export_file')
    p.add_argument('--data_dir',    required=True, help='Base dir of raw parquet files')
    p.add_argument('--output_dir',  required=True, help='Where to write snippet parquet files')
    p.add_argument('--sample_rate', type=float, required=True, help='Samples per second of the data')
    p.add_argument('--pre_sec',     type=float, default=5.0, help='milliseconds before fault to include')
    p.add_argument('--post_sec',    type=float, default=5.0, help='Milliseconds after  fault to include')
    args = p.parse_args()
    main(args)
