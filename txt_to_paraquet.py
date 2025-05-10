import glob, pandas as pd

for txt in glob.glob("/home/vault/iwi5/iwi5305h/new_dataset_90kv/all/replica_*.txt"):
    df = pd.read_csv(txt, sep=';', header=1, encoding='latin1', engine='python')
    # drop negative-time rows
    df = df[df.iloc[:,0] >= 0]
    # write same basename but .parquet
    df.to_parquet(txt.replace('.txt','.parquet'), compression='snappy')
