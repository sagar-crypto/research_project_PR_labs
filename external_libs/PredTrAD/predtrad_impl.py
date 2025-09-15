import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
import click
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import mlflow.pytorch
from src.models import *
from src.utils import *
from src.pot import *
from config import CHECKPOINT_PREDTRAD_DIR


def print_cuda_mem(tag=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated()/1e9
        r = torch.cuda.memory_reserved()/1e9
        m = torch.cuda.max_memory_allocated()/1e9
        print(f"[MEM {tag}] alloc={a:.2f}G reserved={r:.2f}G max_alloc={m:.2f}G")

def _raw_len(t):  # dict_testD[k] is a tensor
    try:
        return int(t.shape[0])
    except Exception:
        return len(t)


def _last_step(x: torch.Tensor) -> torch.Tensor:
            """
            Accepts either [B, D] or [B, 1, D] or [B, L_dec, D].
            Returns [B, D] (last time step).
            """
            if x.dim() == 2:
                return x                               # [B, D]
            elif x.dim() == 3:
                return x[:, -1, :]                     # [B, D]
            else:
                raise RuntimeError(f"Unexpected output dim {x.shape}")



def labels_to_window_labels(lbls_1d: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Any fault inside a window => label 1; else 0."""
    out = []
    n = len(lbls_1d)
    for start in range(0, n - win + 1, stride):
        w = lbls_1d[start : start + win]
        out.append(1 if np.any(w) else 0)
    return np.array(out, dtype=int)

def backprop(epoch, model, data, feats, optimizer, criterion, scheduler, training=True, _shuffle=False):
    if 'TranAD' in model.name:
        dataset = CustomDataset(data)
        shuffle = True if training and _shuffle else False
        dataloader = DataLoader(dataset, batch_size=model.batch, shuffle=shuffle)
        n = epoch + 1
        l1s = []
        if training:
            model.train()
            for d in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = criterion(z, elem) if not isinstance(z, tuple) else (1 / n) * criterion(z[0], elem) + (1 - 1 / n) * criterion(z[1], elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Training at epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.eval()
            test_losses = []
            test_preds = []
            for i, d in enumerate(dataloader):
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                if i == 0:
                    mlflow.log_text(str(summary(model, input_data=(window, elem), verbose=0)), "model_summary.txt")
                z = model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]

                test_loss = criterion(z, elem).detach().cpu().numpy()
                test_pred = z.detach().cpu().numpy()

                test_losses.append(test_loss.squeeze(0))
                test_preds.append(test_pred.squeeze(0))
            return np.concatenate(test_losses, axis=0), np.concatenate(test_preds, axis=0)
    elif 'DTAAD' in model.name:
        _lambda = 0.8
        dataset = CustomDataset(data)
        shuffle = True if training and _shuffle else False
        dataloader = DataLoader(dataset, batch_size=model.batch, shuffle=shuffle)
        l1s = []
        if training:
            for d in dataloader:
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                l1 = _lambda * criterion(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * criterion(z[1].permute(1, 0, 2), elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.eval()
            test_losses = []
            test_preds = []
            for i, d in enumerate(dataloader):
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)

                if i == 0:
                    mlflow.log_text(str(summary(model, input_data=window, verbose=0)), "model_summary.txt")
                z = model(window)
                z = z[1].permute(1, 0, 2)

                test_loss = criterion(z, elem).detach().cpu().numpy()
                test_pred = z.detach().cpu().numpy()

                test_losses.append(test_loss.squeeze(0))
                test_preds.append(test_pred.squeeze(0))
            return np.concatenate(test_losses, axis=0), np.concatenate(test_preds, axis=0)
    elif 'PredTrAD_v1' in model.name:
        model_device = next(model.parameters()).device
        model_dtype  = next(model.parameters()).dtype
        # data is now a WindowDataset of raw windows
        shuffle = True if training and _shuffle else False
        dataset = data
        dataloader = DataLoader(dataset,
                                batch_size=model.batch,
                                shuffle=shuffle)

        if training:
            model.train()
            train_losses = []
            for window in dataloader:
                optimizer.zero_grad()
                # split each window into encoder input, decoder input, and labels
                window = window.to(device=model_device, dtype=torch.float32, non_blocking=True)
                enc    = window[:, :model.n_enc, :]
                dec    = window[:, model.n_enc-1 : model.n_window-1, :]
                labels = window[:, model.n_enc : model.n_window, :]
                outputs = model(enc, dec)
                loss = torch.mean(criterion(outputs, labels))
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f"Epoch {epoch},\tloss= {np.mean(train_losses)}")
            return np.mean(train_losses), optimizer.param_groups[0]['lr']

        else:
            with torch.no_grad():
                model.eval()
                test_losses = []
                test_preds  = []
                for i, window in enumerate(dataloader):
                    if i == 0:
                        # log model summary once
                        samp = window[:1].to(device=model_device, dtype=torch.float32, non_blocking=True)
                        sample_enc = samp[:, :model.n_enc, :]
                        sample_dec = samp[:, model.n_enc-1 : model.n_window-1, :]
                        mlflow.log_text(str(summary(model,
                                                    input_data=(sample_enc, sample_dec),
                                                    verbose=0)),
                                        "model_summary.txt")

                    window = window.to(device=model_device, dtype=torch.float32, non_blocking=True)
                    enc    = window[:, :model.n_enc, :]
                    dec    = window[:, model.n_enc-1 : model.n_window-1, :]
                    labels = window[:, model.n_enc : model.n_window, :]

                    outputs = model(enc, dec)
                    # get last timestep’s loss & prediction like before
                    test_loss = criterion(outputs, labels)\
                                    .detach()\
                                    .cpu()\
                                    .numpy()[:, -1, :]
                    test_pred = outputs.detach().cpu().numpy()[:, -1, :]

                    test_losses.append(test_loss)
                    test_preds.append(test_pred)

                return (np.concatenate(test_losses, axis=0),
                        np.concatenate(test_preds,  axis=0))

    elif 'PredTrAD_v2' in model.name:
        model_device = next(model.parameters()).device
        shuffle = True if training and _shuffle else False
        dataset = data
        dataloader = DataLoader(dataset, batch_size=model.batch, shuffle=shuffle)

        if training:
            model.train()
            train_losses = []
            for window in dataloader:
                optimizer.zero_grad()
                window = window.to(device=model_device, dtype=torch.float32, non_blocking=True)
                enc    = window[:, :model.n_enc, :]
                dec    = window[:, model.n_enc-1 : model.n_window-1, :]
                labels = window[:, model.n_enc : model.n_window, :]   # future targets [B, L_dec, D]

                output = model(enc, dec)
                out_last = _last_step(output)                         # [B, D]
                tgt_last = labels[:, -1, :]                           # [B, D]

                # quick shape sanity:
                if __debug__ and (out_last.shape != tgt_last.shape):
                    print(f"[v2/train] out={tuple(out_last.shape)} tgt={tuple(tgt_last.shape)}")

                loss = torch.mean(criterion(out_last, tgt_last))      # per-feature MSE, then mean
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tloss= {np.mean(train_losses)}')
            return np.mean(train_losses), optimizer.param_groups[0]['lr']
        else:
            with torch.no_grad():
                model.eval()
                test_losses = []
                test_preds  = []
                for i, window in enumerate(dataloader):
                    if i == 0:
                        samp = window[:1].to(device=model_device, dtype=torch.float32, non_blocking=True)
                        sample_enc = samp[:, :model.n_enc, :]
                        sample_dec = samp[:, model.n_enc-1 : model.n_window-1, :]
                        mlflow.log_text(str(summary(model, input_data=(sample_enc, sample_dec), verbose=0)),
                                        "model_summary.txt")

                    window = window.to(device=model_device, dtype=torch.float32, non_blocking=True)
                    enc    = window[:, :model.n_enc, :]
                    dec    = window[:, model.n_enc-1 : model.n_window-1, :]
                    labels = window[:, model.n_enc : model.n_window, :]

                    output   = model(enc, dec)
                    out_last = _last_step(output)                     # [B, D]
                    tgt_last = labels[:, -1, :]                       # [B, D]

                    # loss per feature, keep it as [B, D] to match v1 downstream
                    test_loss = criterion(out_last, tgt_last).detach().cpu().numpy()
                    test_pred = out_last.detach().cpu().numpy()

                    test_losses.append(test_loss)
                    test_preds.append(test_pred)

                return np.concatenate(test_losses, axis=0), np.concatenate(test_preds, axis=0)



def experiment_common(
        model_name: str,
        dataset: str,
        entity: str,
        retrain: bool,
        shuffle: bool,
        val: int,
        mlflow_experiment: str,
        n_epochs: int,
        hyp_lr: float,
        hyp_criterion: str,
        hyp_percentage: float,
        eval_fn: callable,
        params: dict
):
    """Common function for running an experiment.

    Args:
        model_name (str): Model name.
        dataset (str): Dataset name.
        entity (str): Entity to train on.
        retrain (bool): Whether to retrain the model.
        shuffle (bool): Whether to shuffle the data.
        val (int): Frequency of validation.
        mlflow_experiment (str): MLflow experiment name.
        n_epochs (int): Number of epochs.
        hyp_lr (float): Learning rate.
        hyp_criterion (str): Loss function to use.
        hyp_percentage (callable): Evaluation function.
        eval_fn (callable): Evaluation function.
        params (dict): Additional experiment-specific parameters.
    """

    # Early Stopping
    COUNTER = 0
    PATIENCE = 3

    # For TIKI dataset, Validation score is set to infinity as val_score is the loss
    # For other datasets, Validation score is set to 0 as val_score is the F1 score
    BEST_VAL_SCORE = float('inf') if dataset == "TIKI" else 0

    # Path to save the model
    os.makedirs(CHECKPOINT_PREDTRAD_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(CHECKPOINT_PREDTRAD_DIR, f"{model_name}_{dataset}_{entity}.pt")
    

    # Criterion selection
    if hyp_criterion == "LogCosh":
        criterion = LogCoshLoss()
    elif hyp_criterion == "MSE":
        criterion = nn.MSELoss(reduction='none')
    elif hyp_criterion == "Huber":
        criterion = nn.HuberLoss(reduction='none', delta=params.get('delta', 1.0))
    else:
        raise ValueError("Criterion not found")

    mlflow.set_experiment(mlflow_experiment)
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_name=f"{dataset}_{entity}_{model_name}"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_trainD, dict_testD, raw_test_labels, dict_test_timestamps, test_columns = load_dataset(device, dataset=dataset, id=entity, scaler=params["scaler"])

        dims = len(test_columns)
        model, optimizer, scheduler, epoch = load_model(model_name, dims, device=device, lr_d=hyp_lr)
        model = model.to(device=device, dtype=torch.float32)
        model.batch = params.get("batch_size", model.batch)
        if "window_size" in params and params["window_size"]:
            model.n_window = int(params["window_size"])
        if "n_enc" in params and params["n_enc"]:
            model.n_enc = int(params["n_enc"])
        testO = dict_testD.copy()

        window_size = model.n_window
        trainD = WindowDataset(dict_trainD, window_size, params.get('stride_size'))
        stride_size = params.get("stride_size", window_size)

        #For the fault detection dataset
        dict_windowed_valD  = {}
        dict_val_labels     = {}
        dict_windowed_testD = {}
        dict_test_labels    = {}
        if mlflow_experiment == "Experiment_4":
            # --- Build VAL and TEST from dict_testD (both have labels) --

            # how many test files to hold out for validation
            all_items = sorted(dict_testD.items(), key=lambda kv: kv[0])     # deterministic order
            n_files   = len(all_items)
            vsf       = int(params.get("val_split_files", 1))
            vsfrac    = float(params.get("val_split_frac", 0.0))
            if 0.0 < vsfrac < 1.0:
                vsf = max(1, int(round(n_files * vsfrac)))
            vsf = max(0, min(vsf, max(0, n_files - 1)))  # clamp

            val_items  = all_items[:vsf]
            test_items = all_items[vsf:] if vsf > 0 else all_items
            print(f"[SPLIT] n_files={n_files}, vsf={vsf}, val_files={len(val_items)}, test_files={len(test_items)}")

            # Build VAL
            for k, v in val_items:
                dict_windowed_valD[k] = WindowDataset({k: v}, window_size, stride_size)
                full_lbls = raw_test_labels[k]
                dict_val_labels[k] = labels_to_window_labels(full_lbls, window_size, stride_size)

            # Build TEST
            for k, v in test_items:
                dict_windowed_testD[k] = WindowDataset({k: v}, window_size, stride_size)
                full_lbls = raw_test_labels[k]
                dict_test_labels[k] = labels_to_window_labels(full_lbls, window_size, stride_size)
            n_val  = len(dict_windowed_valD)
            n_test = len(dict_windowed_testD)
            print(f"[SPLIT] val_files={n_val}, test_files={n_test}")
            try:
                mlflow.log_param("n_val_files",  n_val)
                mlflow.log_param("n_test_files", n_test)
            except Exception:
                pass

            if n_test == 0:
                raise RuntimeError(
                    "No TEST files. Set 'val_split_files': 0 (and/or lower 'val_split_frac')."
                )

            for k, ds in dict_windowed_testD.items():
                nw = len(ds) if hasattr(ds, "__len__") else None
                print(f"[TESTREADY] {k}: raw_len={_raw_len(dict_testD[k])}, "
                    f"windows={nw} (win={window_size}, stride={stride_size})")
                if nw is not None and nw == 0:
                    print(f"[WARN] {k} has zero windows → it will be skipped in testing.")


        # Experiment 2
        elif dataset == "TIKI" and mlflow_experiment == "Experiment_2":
            trainD, valD = train_val_split(data=trainD, train_size=0.8, shuffle=shuffle, percentage=hyp_percentage)
            dict_windowed_valD = {"val": valD} if val > 0 else {}
            dict_windowed_testD = {k: convert_to_windows(v, window_size) for k, v in dict_testD.items()}

        # Experiment 1 and 3
        elif dataset in ["SMD", "SMAP", "MSL"]:
            trainD, _ = train_val_split(data=trainD, train_size=1.0, shuffle=shuffle, percentage=hyp_percentage)

            dict_windowed_testD = {}
            dict_windowed_valD = {}
            dict_val_labels = {}

            #Experiment 3
            if mlflow_experiment == "Experiment_3":
                val_entities = ["machine-1-1", "machine-1-2", "machine-1-3"]
                for k, v in dict_testD.items():
                    windowed_data = convert_to_windows(v, window_size)
                    if k in val_entities:
                        dict_windowed_valD[k] = windowed_data
                        dict_val_labels[k] = raw_test_labels[k]
                        raw_test_labels.pop(k)
                    else:
                        dict_windowed_testD[k] = windowed_data
            # Experiment 1 
            else:
                dict_windowed_testD = {k: convert_to_windows(v, window_size) for k, v in dict_testD.items()}

        # save parameters to mlflow
        mlflow.log_param("model", model_name)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("entity", entity)
        mlflow.log_param("device", device)
        mlflow.log_param("epochs", n_epochs)
        mlflow.log_param("lr", hyp_lr)

        # Additional parameters passed in the dictionary
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        val_every = int(params.get("val_every", 1))           # run validation every N epochs
        have_val  = len(dict_windowed_valD) > 0               # we actually built a val set
        BEST_VAL_SCORE = float('inf') if dataset == "TIKI" else -float('inf')
        COUNTER, PATIENCE = 0, 3

        if retrain:
            for e in tqdm(range(n_epochs)):
                lossT, lr = backprop(e, model, trainD, dims, optimizer, criterion, scheduler,
                                    training=True, _shuffle=shuffle)
                mlflow.log_metric("train_loss", float(lossT), step=e)
                mlflow.log_metric("train_lr",   float(lr),    step=e)

                # If no validation set exists (e.g., you set val_split_files=0), still save at epoch 0 & last
                if not have_val:
                    if e == 0 or e == n_epochs - 1:
                        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                        torch.save(model.state_dict(), SAVE_PATH)
                        try: mlflow.log_artifact(SAVE_PATH)
                        except Exception: pass
                    continue

                # Validate every `val_every` epochs
                do_val = (val_every > 0) and ((e + 1) % val_every == 0)
                if not do_val:
                    continue

                model.eval()
                with torch.no_grad():
                    # Stable POT baseline: no shuffle here
                    lossTrain,_ = backprop(e, model, trainD, dims, optimizer, criterion, scheduler,
                                        training=False, _shuffle=False)
                    params["init_score"] = np.mean(lossTrain, axis=1)

                    val_score = 0.0
                    for val_file, valD in dict_windowed_valD.items():
                        lossV,_ = backprop(e, model, valD, dims, optimizer, criterion, scheduler,
                                        training=False, _shuffle=False)
                        if dataset != "TIKI":
                            mean_lossV = (np.mean(lossV, axis=1) if getattr(lossV, "ndim", 1) == 2
                                        else np.asarray(lossV).ravel())

                            lbl = np.asarray(dict_val_labels[val_file])
                            labelsFinal = (lbl.any(axis=1) if lbl.ndim >= 2 else (lbl != 0)).astype(np.int64).ravel()

                            # align lengths (and skip if empty)
                            m = min(len(mean_lossV), len(labelsFinal))
                            if m == 0:
                                print(f"[VAL] Skipping {val_file}: no windows after windowing "
                                    f"(loss_n={len(mean_lossV)}, labels_n={len(labelsFinal)})")
                                continue
                            mean_lossV  = mean_lossV[:m]
                            labelsFinal = labelsFinal[:m]

                            result = eval_fn(mean_lossV, labelsFinal, params)
                            f1 = float(result['final_result']['f1'])
                            mlflow.log_metric(f"val_f1_{val_file}", f1, step=e)
                            val_score += f1
                        else:
                            v = float(np.mean(lossV))
                            mlflow.log_metric(f"val_loss_{val_file}", v, step=e)
                            val_score  += v

                    val_score /= max(1, len(dict_windowed_valD))
                    mlflow.log_metric("val_score", val_score, step=e)

                    improved = (val_score < BEST_VAL_SCORE) if dataset == "TIKI" else (val_score > BEST_VAL_SCORE)
                    if improved or (e == 0 and not os.path.exists(SAVE_PATH)):
                        BEST_VAL_SCORE = val_score
                        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                        torch.save(model.state_dict(), SAVE_PATH)
                        try: mlflow.log_artifact(SAVE_PATH)
                        except Exception: pass
                        COUNTER = 0
                    else:
                        COUNTER += 1
                        print(f"Early stopping: {COUNTER}/{PATIENCE}")
                        if COUNTER >= PATIENCE:
                            print(f"Early stopping at epoch {e + 1}")
                            break
                model.train()


        # Load the best model from the training phase for testing
        if n_epochs > 1 and val > 0:
            model.load_state_dict(torch.load(SAVE_PATH))
            model.to(device)
        model.eval()

        lossT,_ = backprop(0, model, trainD, dims, optimizer, criterion, scheduler, training=False, _shuffle=shuffle)

        if all((len(ds) == 0) for ds in dict_windowed_testD.values()):
            raise RuntimeError(
                "All TEST files produce zero windows. Increase data or reduce 'window_size'/'stride_size'."
        )

        print("Testing phase")
        for i, (test_file, testD) in tqdm(enumerate(dict_windowed_testD.items())):
            test_labels = dict_test_labels[test_file]
            loss, y_pred = backprop(0, model, testD, dims, optimizer, criterion, scheduler, training=False, _shuffle=shuffle)
            if test_labels.ndim == 1:
                test_labels = test_labels.reshape(-1, 1)

            ### Scores
            lbl = np.asarray(test_labels)
            if lbl.ndim == 1:
                lbl = lbl.reshape(-1, 1)
            if loss.ndim == 2 and lbl.shape[1] == 1 and loss.shape[1] > 1:
                # repeat a single label column across all feature dims
                lbl = np.repeat(lbl, loss.shape[1], axis=1)
            test_labels = lbl

            df = pd.DataFrame()
            test_pot_predictions = []
            test_pot_thresholds  = []

            num_dims = loss.shape[1] if loss.ndim == 2 else 1
            for j in range(num_dims):
                test_col_name = test_columns[j] if j < len(test_columns) else f"dim{j}"

                # per-dim score (1-D) and label (binarized 1-D)
                score_1d = (loss[:, j] if loss.ndim == 2 else np.asarray(loss)).ravel()
                lbl_col  = test_labels[:, j] if getattr(test_labels, "ndim", 1) >= 2 else test_labels
                label_1d = (np.asarray(lbl_col) != 0).astype(np.int64).ravel()

                # per-dim init_score (from train lossT)
                init_1d = (lossT[:, j] if getattr(lossT, "ndim", 1) == 2 else np.asarray(lossT)).ravel()

                # align lengths and skip empty
                m = min(len(score_1d), len(label_1d), len(init_1d))
                if m == 0:
                    print(f"[TEST] Skipping col {j} ({test_col_name}): empty series "
                        f"(score={len(score_1d)}, labels={len(label_1d)}, init={len(init_1d)})")
                    continue

                score_1d = score_1d[:m]
                label_1d = label_1d[:m]
                init_1d  = init_1d[:m]
                params["init_score"] = init_1d  # POT reference for this dim

                score_1d = np.asarray(score_1d, dtype=np.float64)
                init_1d  = np.asarray(init_1d,  dtype=np.float64)
                score_1d = np.nan_to_num(score_1d, nan=0.0, posinf=None, neginf=0.0)
                init_1d  = np.nan_to_num(init_1d,  nan=0.0, posinf=None, neginf=0.0)
                score_1d = np.where(score_1d <= 0, 1e-12, score_1d)
                init_1d  = np.where(init_1d  <= 0, 1e-12, init_1d)
                params["init_score"] = init_1d  # POT reference for this dim

                # compute per-dim metrics (guard against POT errors)
                try:
                    result = eval_fn(score_1d, label_1d, params)
                except ValueError as e:
                    print(f"[TEST] Skipping col {j} ({test_col_name}) due to POT error: {e}")
                    continue

                # collect outputs safely
                test_pot_predictions.append(result.get("final_pot_predictions", None))
                test_pot_thresholds.append(result.get("final_pot_thresholds", None))

                result_df = pd.DataFrame([result.get("final_result", {})])
                result_df["column"] = test_col_name
                df = pd.concat([df, result_df], ignore_index=True)

            mlflow.log_metric(f"test_loss_{test_file}", np.mean(loss))

            # Final results
            init = np.asarray(np.mean(lossT, axis=1), dtype=float).ravel()

            score = (np.mean(loss, axis=1) if getattr(loss, "ndim", 1) == 2
                    else np.asarray(loss)).ravel()

            lbl_all = np.asarray(test_labels)
            labelsFinal = (lbl_all.any(axis=1) if lbl_all.ndim >= 2 else (lbl_all != 0)) \
                            .astype(np.int64).ravel()

            # Align lengths and skip if any is empty
            m = min(len(score), len(labelsFinal), len(init))
            if m == 0:
                print(f"[TEST] Skipping {test_file}: empty series "
                    f"(score={len(score)}, labels={len(labelsFinal)}, init={len(init)})")
                continue

            score        = score[:m]
            labelsFinal  = labelsFinal[:m]
            params["init_score"] = init[:m]   # POT reference series

            score = np.asarray(score, dtype=np.float64)
            init  = np.asarray(init,  dtype=np.float64)
            score = np.nan_to_num(score, nan=0.0, posinf=None, neginf=0.0)
            init  = np.nan_to_num(init,  nan=0.0, posinf=None, neginf=0.0)
            score = np.where(score <= 0, 1e-12, score)
            init  = np.where(init  <= 0, 1e-12, init)
            params["init_score"] = init  # reassign sanitized init

            # Now call the evaluator with the aligned vectors
            try:
                test_result = eval_fn(score, labelsFinal, params)
            except ValueError as e:
                print(f"[TEST] Skipping {test_file} due to POT error: {e}")
                continue

            final_result = test_result["final_result"]
            final_result.update(hit_att(loss, test_labels))
            final_result.update(ndcg(loss, test_labels))

            final_pot_predictions = test_result.get("final_pot_predictions", None)
            final_pot_thresholds = test_result.get("final_pot_thresholds", None)

            # make second result dataframe for Final results
            final_result_df = pd.DataFrame([final_result])

            np_pot_predictions = np.array(test_pot_predictions).T
            np_pot_thresholds = np.array(test_pot_thresholds).T
            np_test_data = testO[test_file].cpu().numpy()

            with tempfile.TemporaryDirectory() as tmpdirname:

                # make folder for test file and model name
                save_folder = f"{tmpdirname}/{model_name}_{test_file}"
                os.makedirs(save_folder, exist_ok=True)

                # save df to csv
                df.to_csv(f'{save_folder}/test_results.csv', index=False)

                # save model
                torch.save(model.state_dict(), f"{save_folder}/model.pt")

                # save final result df to csv
                final_result_df.to_csv(f'{save_folder}/final_results.csv', index=False)

                # save pot_predictions to npy
                np.save(f"{save_folder}/pot_prediction.npy", np_pot_predictions)

                # save pot_thresholds to npy
                np.save(f"{save_folder}/pot_threshold.npy", np_pot_thresholds)

                # save final_pot_predictions to npy
                np.save(f"{save_folder}/final_pot_predictions.npy", final_pot_predictions)

                # save final_pot_thresholds to npy
                np.save(f"{save_folder}/final_pot_thresholds.npy", final_pot_thresholds)

                # save y_pred to npy
                np.save(f"{save_folder}/y_pred.npy", y_pred)

                # save loss to npy
                np.save(f"{save_folder}/loss.npy", loss)

                # save timestamps to npy
                np.save(f"{save_folder}/timestamps.npy", dict_test_timestamps[test_file])

                # save test data to npy
                np.save(f"{save_folder}/test_data.npy", np_test_data)

                # save the labels to npy
                np.save(f"{save_folder}/labels.npy", dict_test_labels[test_file])

                # make zip folder of "test_file" folder
                shutil.make_archive(save_folder, 'zip', save_folder)

                # log zip folder to mlflow
                mlflow.log_artifact(f"{save_folder}.zip")

            pre_metric_name = f"test_{test_file}" if len(dict_testD) > 1 else "test_"
            for key, value in final_result.items():
                metric_name = f"{pre_metric_name}_{key}".replace("@", "_at_").replace("%", "pct")
                mlflow.log_metric(metric_name, value)

            mlflow.log_metric(f"{pre_metric_name}_f1_new", calculate_f1_score(df))
        mlflow.end_run()


def eval_fn_exp_1_2(loss, labels, params):
    """Evaluates the model using the POT-based evaluation and additional params .

    Args:
        loss (np.ndarray): The loss values from the model.
        labels (np.ndarray): The true labels.
        params (dict): Additional parameters for the evaluation.

    Returns:
        dict: The evaluation results including metrics and predictions.
    """
    # Perform POT-based evaluation
    final_result, final_pot_predictions = pot_eval(params.get('init_score', np.array([])), loss, labels, lm=(params.get('lm_d0', None), params.get('lm_d1', None)))

    return {
        "final_result": final_result,
        "final_pot_predictions": final_pot_predictions
    }
def eval_fn_exp_3(loss, labels, params):
    """Evaluates the model using the POT-based evaluation and additional metrics.

    Args:
        loss (np.ndarray): The loss values from the model.
        labels (np.ndarray): The true labels.
        params (dict): Additional parameters for the evaluation.

    Returns:
        dict: The evaluation results including metrics and predictions.
    """
    # Perform POT-based evaluation
    final_result, final_pot_predictions, final_pot_thresholds = pot_eval_dynamic(loss[:1000], loss[1000:], labels[1000:], q=params.get('q', None), level=params.get('level', None))

    return {
        "final_result": final_result,
        "final_pot_predictions": final_pot_predictions,
        "final_pot_thresholds": final_pot_thresholds
    }

def eval_fn_exp_4(loss: np.ndarray, labels: np.ndarray, params: dict) -> dict:
    """
    Simple POT-based evaluation for Experiment 4 (dry-run).
    """
    final_result, final_pot_predictions = pot_eval(
        params.get('init_score', np.array([])),
        loss,
        labels,
        lm=(params.get('lm_d0', None), params.get('lm_d1', None))
    )
    return {
        "final_result": final_result,
        "final_pot_predictions": final_pot_predictions
    }

# Main CLI
@click.group()
def cli():
    """Main CLI for running experiments."""
    pass

@cli.command('experiment1_2')
@click.option('--config', type=click.Path(exists=True), is_eager=True, help="Path to a JSON config file with parameters.")
def experiment1_2(config):
    """
    A function to run the 3_1 experiment with the given model and dataset configuration.
    Args are dynamically passed to the `experiment_common` function.
    """

    # Load the parameters from the config file
    params = json.load(open(config))

    # Set up additional parameters as a dictionary
    additional_params = {
        'lm_d0': params.get("hyp_lm_d0"),
        'lm_d1': params.get("hyp_lm_d1"),
        'delta': params.get("hyp_delta"),
        'scaler': params.get("scaler", "min_max")
    }
    print(params)

    # Call the common experiment function with evaluation and saving functions
    experiment_common(params.get("model_name"), params.get("dataset"), params.get("entity"), params.get("retrain"), params.get("shuffle"), 
                      params.get("val"), params.get("mlflow_experiment"), params.get("n_epochs"), params.get("hyp_lr"), params.get("hyp_criterion"), 
                      params.get("hyp_percentage"), eval_fn_exp_1_2, additional_params)

@cli.command('experiment3')
@click.option('--config', type=click.Path(exists=True), is_eager=True, help="Path to a JSON config file with parameters.")
def experiment3_1(config):
    """
    A function to run the experiment 3 with the given model and dataset configuration.
    Args are dynamically passed to the `experiment_common` function.
    """

    # Load the parameters from the config file
    params = json.load(open(config))

    # Set up additional parameters as a dictionary
    additional_params = {
        'q': params.get('hyp_q'),
        'level': params.get('hyp_level'),
        'delta': params.get('hyp_delta'),
        'scaler': params.get("scaler", "min_max")
    }

    # Call the common experiment function with evaluation and saving functions
    experiment_common(params.get("model_name"), params.get("dataset"), params.get("entity"), params.get("retrain"), params.get("shuffle"), 
                      params.get("val"), params.get("mlflow_experiment"), params.get("n_epochs"), params.get("hyp_lr"), params.get("hyp_criterion"), 
                      params.get("hyp_percentage"), eval_fn_exp_3, additional_params)

@cli.command('experiment4')
@click.option('--config', type=click.Path(exists=True), required=True)
def experiment4(config):
    """
    Run custom Experiment 4 (dry-run on 3% subset).
    """
    params = json.load(open(config))
    additional_params = {
        'init_score': np.array(params.get('init_score', [])),
        'lm_d0':       params.get('lm_d0'),
        'lm_d1':       params.get('lm_d1'),
        'scaler':      params.get('scaler', 'min_max'),
        'stride_size': params.get('stride_size')
    }
    experiment_common(
        params["model_name"],
        params["dataset"],
        params["entity"],
        params["retrain"],
        params["shuffle"],
        params["val"],
        params["mlflow_experiment"],
        params["n_epochs"],
        params["hyp_lr"],
        params["hyp_criterion"],
        params["hyp_percentage"],
        eval_fn_exp_4,
        additional_params
    )

if __name__ == '__main__':
    cli()
