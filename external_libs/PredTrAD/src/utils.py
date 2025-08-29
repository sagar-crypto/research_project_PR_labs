import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


import os
import zipfile
import numpy as np
import joblib
import torch
from torch import nn
import subprocess
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import ndcg_score
from src import models
from src.PredTrAD_v2 import PredTrAD_v2
from src.PredTrAD_v1 import PredTrAD_v1
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import SCALER_PREDTRAD_DIR



def _default_scaler_dir_from_zip(zip_path: str) -> str:
    base = os.path.dirname(zip_path)
    d = os.path.join(base, "scalers")
    os.makedirs(d, exist_ok=True)
    return d

def apply_sklearn_scaler(train_dict: dict,
                         test_dict: dict,
                         dataset_name: str,
                         scaler_choice: str,
                         scaler_dir: str):
    """
    Fits a sklearn scaler on TRAIN only (per entity), applies to TRAIN and TEST,
    and saves the scaler to <scaler_dir>/<dataset>_<entity>_<scaler>.joblib
    """
    scaler_choice = (scaler_choice or "").lower()
    for ent, Xtr in list(train_dict.items()):
        # to numpy, flatten time: [N,T,D] -> [N*T, D]
        X = Xtr.detach().cpu().numpy()
        if X.ndim == 3:
            N, T, D = X.shape
            Xf = X.reshape(-1, X.shape[-1])
        else:  # [N, D]
            D = X.shape[-1]
            Xf = X.reshape(-1, D)

        if scaler_choice in ("minmax", "min_max", "sk_minmax"):
            scaler = MinMaxScaler()
            scaler_tag = "minmax"
        elif scaler_choice in ("standard", "zscore", "sk_standard"):
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler_tag = "standard"
        else:
            # no scaling
            continue

        # fit on TRAIN (flattened)
        scaler.fit(Xf)

        # transform TRAIN
        X_scaled = scaler.transform(Xf).astype("float32").reshape(X.shape)
        train_dict[ent] = torch.from_numpy(X_scaled)

        # transform TEST (if present)
        if ent in test_dict:
            Xt = test_dict[ent].detach().cpu().numpy()
            if Xt.ndim == 3:
                Xt_scaled = scaler.transform(Xt.reshape(-1, D)).astype("float32").reshape(Xt.shape)
            else:
                Xt_scaled = scaler.transform(Xt.reshape(-1, D)).astype("float32").reshape(Xt.shape)
            test_dict[ent] = torch.from_numpy(Xt_scaled)

        # save scaler
        out_path = os.path.join(scaler_dir, f"{dataset_name}_{ent}_{scaler_tag}.joblib")
        joblib.dump(scaler, out_path)



class LogCoshLoss(torch.nn.Module):
    """
    Logarithm of the hyperbolic cosine of the prediction error.

    Args:
        y_t: torch.Tensor of shape (batch_size, n_features)
            The true values.
        y_prime_t: torch.Tensor of shape (batch_size, n_features)
            The predicted values.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = y_t - y_prime_t
        return torch.log(torch.cosh(ey_t + 1e-12))
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def hit_att(ascore, labels, ps = [100, 150]):
    res = {}
    for p in ps:
        hit_score = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
            if l:
                size = round(p * len(l) / 100)
                a_p = set(a[:size])
                intersect = a_p.intersection(l)
                hit = len(intersect) / len(l)
                hit_score.append(hit)
        res[f'Hit@{p}%'] = np.mean(hit_score)
    return res

def ndcg(ascore, labels, ps = [100, 150]):
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try:
                    hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
                except Exception as e:
                    return {}
                ndcg_scores.append(hit)
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
    return res

def execute_command(command):
    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Decode and return the standard output
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        # If the command fails, capture and return the standard error
        return e.stderr.decode('utf-8')

def cut_tensor(percentage: float, tensor: torch.Tensor):
    """
    Cut the tensor by percentage from the middle in the first dimension.

    """
    mid = round(tensor.size(0) / 2)
    window = round(tensor.size(0) * percentage * 0.5)
    return tensor[mid - window: mid + window, :]

def train_val_split(data : torch.Tensor, train_size : float = 0.8, shuffle : bool = False, percentage : float = 1.0):
    """
    Split the data into training and validation sets among the first dimension.
    """

    # set random state for reproducibility
    torch.manual_seed(0)

    if shuffle:
        data = data[torch.randperm(data.size(0))]

    if percentage < 1.0 and percentage > 0.0:
        data = cut_tensor(percentage, data)

    split = int(data.size(0) * train_size)
    return data[:split], data[split:]

def convert_to_windows(data, window_size):
    windows = []
    for i, g in enumerate(data):
        if i >= window_size:
            w = data[(i + 1) - window_size:(i + 1)]  # cut
        else:
            w = torch.cat([data[0].repeat(window_size - (i+1), 1), data[0:(i+1)]])  # pad (added (i+1), because without that, windows[0] and windows[1] are the same)
        windows.append(w)
    return torch.stack(windows)

def load_dataset(device, dataset="", id="", scaler=""):
    MAX_FILES   = int(os.getenv("MAX_FILES", "0"))    # 0 = no limit
    MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))  # 0 = no limit

    dict_train_tensor = {}
    dict_test_tensor  = {}
    dict_test_labels  = {}

    seen = {"train": 0, "test": 0, "val": 0}

    zip_path = f'/home/vault/iwi5/iwi5305h/data_predtrad/{dataset}.zip'
    with zipfile.ZipFile(zip_path) as z:
        for file in z.namelist():
            if ((scaler in file and dataset == "TIKI") or (id in file and dataset != "TIKI")) and file.endswith('.npy'):
                filename = os.path.basename(file)
                entity, split = filename.split('_')

                # ---- TRAIN ----
                if "train" in split:
                    if MAX_FILES and seen["train"] >= MAX_FILES:
                        continue
                    arr = np.load(z.open(file), allow_pickle=False).astype(np.float32, copy=False)
                    if MAX_SAMPLES and arr.shape[0] > MAX_SAMPLES:
                        arr = arr[:MAX_SAMPLES]
                    x = torch.from_numpy(arr).contiguous()      # CPU float32
                    dict_train_tensor[entity] = x
                    seen["train"] += 1

                # ---- TEST ----
                elif "test" in split:
                    if MAX_FILES and seen["test"] >= MAX_FILES:
                        continue
                    arr = np.load(z.open(file), allow_pickle=False).astype(np.float32, copy=False)
                    if MAX_SAMPLES and arr.shape[0] > MAX_SAMPLES:
                        arr = arr[:MAX_SAMPLES]
                    x = torch.from_numpy(arr).contiguous()      # CPU float32
                    dict_test_tensor[entity] = x
                    seen["test"] += 1

                # ---- LABELS ----
                elif "labels" in split:
                    dict_test_labels[entity] = np.load(z.open(file), allow_pickle=False)

    scaler_choice = (scaler or "").lower()
    if scaler_choice not in ("", "none"):
        scaler_dir = os.environ.get(
            "PREDTRAD_SCALER_DIR",
            _default_scaler_dir_from_zip(zip_path)
        )
        apply_sklearn_scaler(
            dict_train_tensor, dict_test_tensor,
            dataset_name=dataset,
            scaler_choice=scaler_choice,  # "sk_standard" or "sk_minmax"
            scaler_dir=scaler_dir
        )

    # mock timestamps: start at 2000-06-17 00:00:00, 30s interval
    start_time = datetime.strptime("2000-06-17 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
    dict_test_timestamps = {
        k: np.arange(start_time, start_time + len(v) * 30, 30, dtype=np.float64)
        for k, v in dict_test_tensor.items()
    }

    # robust column inference (works even if train is empty)
    n_columns = None
    if dict_train_tensor:
        first_key = next(iter(dict_train_tensor))
        n_columns = dict_train_tensor[first_key].shape[1]
    elif dict_test_tensor:
        first_key = next(iter(dict_test_tensor))
        n_columns = dict_test_tensor[first_key].shape[1]
    else:
        n_columns = 0  # no data; caller should handle

    columns = [f"feature_{i}" for i in range(n_columns)]
    return dict_train_tensor, dict_test_tensor, dict_test_labels, dict_test_timestamps, columns


def load_model(modelname, dims, device, lr_d = 0.0001):
    if ("PredTrAD_v1" in modelname):
        # PARAMETERS
        input_dim = dims
        d_model = 256
        n_heads = 8
        num_encoder_layers = 1
        num_decoder_layers = 1
        ff_hidden_dim = 1024
        
        model = PredTrAD_v1(input_dim, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim)
    elif ("PredTrAD_v2" in modelname):
        # PARAMETERS
        input_dim = dims
        d_model = 512
        n_heads = 8
        num_encoder_layers = 1
        num_decoder_layers = 1
        ff_hidden_dim = 1024

        model = PredTrAD_v2(input_dim, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim)
    else:
        model_class = getattr(models, modelname)
        model = model_class(dims)
        
    model.double().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    epoch = -1
    return model, optimizer, scheduler, epoch


class WindowDataset(torch.utils.data.Dataset):
    """
    Streams sliding windows from a dict of time-series tensors
    without concatenating them all in memory.
    """
    def __init__(self, series_dict, window_size, stride):
        self.window_size = window_size
        self.stride      = stride
        self.series_dict = series_dict

        # build index: list of (key, start_idx)
        self.index_map = []
        for key, tensor in series_dict.items():
            length = tensor.shape[0]
            for start in range(0, length - window_size + 1, stride):
                self.index_map.append((key, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        key, start = self.index_map[idx]
        ts = self.series_dict[key]   # torch.Tensor shape (T, feats)
        window = ts[start : start + self.window_size]
        return window
