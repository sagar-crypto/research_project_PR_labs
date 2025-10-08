import argparse
import sys
import re
import pandas as pd
from pathlib import Path

def autodetect_sep(in_path):
    # Simple heuristic: check first line for common separators
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
        line = f.readline()
    if line.count(';') > line.count(','):
        return ';'
    return ','

def load_df_any(in_path, sep=None):
    if sep is None:
        sep = autodetect_sep(in_path)
    try:
        df = pd.read_csv(in_path, sep=sep)
    except Exception:
        # fallback: try the other one
        alt = ';' if sep == ',' else ','
        df = pd.read_csv(in_path, sep=alt)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to original labels.csv")
    ap.add_argument("--out", dest="out_path", required=True, help="Path to write cleaned labels csv")
    ap.add_argument("--sep", dest="sep", default=None, help="CSV delimiter (default: auto)")
    ap.add_argument("--filecol", default=None, help="Column with file path (default: auto detect export_file/filename)")
    ap.add_argument("--startcol", default=None, help="Start time column (default: auto detect t_evnt_start/start...)")
    ap.add_argument("--endcol", default=None, help="End time column (default: auto detect t_evnt_end/end...)")
    ap.add_argument("--force_ext", default="parquet", help="Force extension for output names (default: parquet)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    df = load_df_any(str(in_path), sep=args.sep)

    # auto-detect columns if not provided
    def find_col(options):
        # exact lower match
        lower_map = {c.lower(): c for c in df.columns}
        for opt in options:
            if opt in lower_map:
                return lower_map[opt]
        # substring
        for opt in options:
            for c in df.columns:
                if opt in c.lower():
                    return c
        return None

    file_col  = args.filecol  or find_col(["export_file","file","filename","parquet","fname"])
    start_col = args.startcol or find_col(["t_evnt_start","start","start_s","fault_start","start_time","start_ms"])
    end_col   = args.endcol   or find_col(["t_evnt_end","end","end_s","fault_end","end_time","end_ms"])

    if any(x is None for x in [file_col, start_col, end_col]):
        print("ERROR: Could not detect file/start/end columns.", file=sys.stderr)
        print("Columns present:", list(df.columns), file=sys.stderr)
        sys.exit(2)

    # build cleaned filename
    fn = df[file_col].astype(str)\
        .str.replace("\\\\","/", regex=False)\
        .str.split("/").str[-1]   # basename

    # replace extension to desired
    ext = args.force_ext.strip(".")
    fn = fn.str.replace(r"\.[^.]+$", f".{ext}", regex=True)

    clean = df[[start_col, end_col]].copy()
    clean.insert(0, "filename", fn)
    clean.columns = ["filename", "start_s", "end_s"]

    # coerce to numeric seconds (assume already seconds)
    clean["start_s"] = pd.to_numeric(clean["start_s"], errors="coerce")
    clean["end_s"]   = pd.to_numeric(clean["end_s"], errors="coerce")

    clean.to_csv(out_path, index=False)
    print(f"Wrote cleaned labels to: {out_path} (rows={len(clean)})")
    # small summary
    print("Sample rows:")
    print(clean.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
