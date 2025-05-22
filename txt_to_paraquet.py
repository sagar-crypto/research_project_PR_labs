import glob
import pandas as pd

for txt in glob.glob("/home/vault/iwi5/iwi5305h/new_dataset_90kv/all/replica_*.txt"):
    # 1) Read the first two lines as a two-level header
    df = pd.read_csv(
        txt,
        sep=";",
        header=[0, 1],            # <- preserve both header rows
        encoding="latin1",
        engine="python",
    )

    # 2) Find the MultiIndex column whose **second** level is "Zeitpunkt in s"
    time_col = [col for col in df.columns if col[1] == "Zeitpunkt in s"][0]

    # 3) Drop any row with a negative time stamp
    df = df[df[time_col] >= 0]

    # 4) Write out to Parquet
    df.to_parquet(
        txt.replace(".txt", ".parquet"),
        compression="snappy",
    )
