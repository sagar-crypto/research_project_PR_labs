# ⚡ PR-Labs Research Project — Time-Series Fault Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🚀 Overview

This repository contains the complete deep learning pipeline for **time-series fault detection and anomaly analysis** developed at **PR-Labs (FAU Erlangen-Nürnberg)**.

The project combines:

* 🧠 **Custom Transformer Autoencoder (PredTrAD-inspired)**
* 🤖 **Hugging Face TimeSeriesTransformer fine-tuning**
* 🔍 **Variational Autoencoder (VAE) for latent representation**
* 📊 **Unsupervised clustering and evaluation**

---

## 🗂 Project Structure

```
research_project_PR_labs/
├── transformer/                    # Custom Transformer Autoencoder + Classifier
│   ├── train.py                    # Train transformer autoencoder
│   ├── train_classification_head.py# Train classification head
│   ├── model.py                    # Model definitions
│   ├── data_processing.py          # Dataset and loader
│   ├── utils.py                    # Helper functions
│   ├── utils_label.py              # Helper functions to map the labels to data
│   ├── clasify_config.json         # config for classification head
│   └── hyperParameters.json        # Transformer config
│
├── hugging_face_transformer/       # Hugging Face TimeSeriesTransformer head training
│   ├── train.py                    # Train transformer autoencoder
│   ├── utils.py                    # Helper functions
│   ├── train_linear_head.py        # Train classification head
│   ├── memmap_dataset.py           # Another data loader when a memmap dataset is passed
│   ├── linear_head.py              # linear head model
│   ├── data_processing.py          # Dataset and loader
│   ├── classify_config.json        # config for classification head
│   └── hyper_parameter.json        # Transformer config
│
├── VAE/                            # Variational Autoencoder implementation
│   ├── train.py                    # Train autoencoder
│   ├── clustering_util.py          # Helper function for Clustering
│   ├── model.py                    # Model definitions
│   ├── ploting_util.py             # Helper function for ploting
│   ├── 6_clusters.py               # Clustering Helper
│   ├── data_processing.py          # Dataset and loader
│   └── hyper_parameter.json        # Train transformer autoencoder
│
├── external_libs/                  # External cloned dependencies
│   ├── PredTrADv1/
│   └── PredTrADv2/
│
├── scripts/                        # Data and label processing scripts
├── checkpoints/                    # Saved model checkpoints
├── config.py                       # Global path configuration
├── pyproject.toml                  # Package definition (pip installable)
├── environment.lock.yml            # Fully reproducible Conda environment
├── requirements.lock.txt           # Pinned pip requirements
└── README.md
```

---

## ⚙️ Environment Setup

### 🧠 Option 1 — Reproduce the Exact Working Environment

```bash
# Clone repository
git clone https://github.com/<your-username>/research_project_PR_labs.git
cd research_project_PR_labs

# Create & activate environment
mamba env create -f environment.lock.yml
mamba activate prlabs-ts

# Install correct PyTorch build for your hardware (choose one)
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# or CPU-only
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Install project
pip install -e .
```

### 🧬 Option 2 — Lightweight Setup (Flexible Versions)

```bash
conda create -n prlabs-ts python=3.10
conda activate prlabs-ts

pip install -e .
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

---

## ⚡ Quick Start Commands

### 🔹 Train Transformer Autoencoder

```bash
python -m transformer.train
```

### 🔹 Train Linear Classification Head

```bash
python -m transformer.train_classification_head
```

### 🔹 Train Hugging Face Transformer

```bash
python -m hugging_face_transformer.train
```

### 🔹 Fine-Tune Hugging Face Transformer Head

```bash
python -m hugging_face_transformer.train_linear_head
```

### 🔹 Train PredTrad V1 Transformer

```bash
# paths
PREDTRAD_ROOT="$HOME/research_project_PR_labs/external_libs/PredTrAD"
CONFIG_PATH="$PREDTRAD_ROOT/config/epx4/myconfig.json"

# (optional) limit input for smoke tests (0 = no limit)
export MAX_FILES=0
export MAX_SAMPLES=0

# (optional) pick a GPU
# export CUDA_VISIBLE_DEVICES=0

# launch training (experiment4 entrypoint)
python "$PREDTRAD_ROOT/predtrad_impl.py" experiment4 \
  --config "$CONFIG_PATH"
```

### 🔹 Train Variational Autoencoder

```bash
python -m VAE.train
```

🗁 All training outputs (checkpoints, logs, and metrics) are saved under:

```
checkpoints/
mlruns/
```

---

## 📊 Pipeline Overview

| Stage                       | Description                                  | Script                                      |
| --------------------------- | -------------------------------------------- | ------------------------------------------- |
| **Data Preparation**        | Preprocess and label parquet/CSV sensor data | `scripts/make_clean_labels.py`              |
| **Transformer AE Training** | Unsupervised feature learning                | `transformer/train.py`                      |
| **Fault Head Training**     | Linear classification head                   | `transformer/train_classification_head.py`  |
| **Hugging Face Head**       | Linear head fine-tuning                      | `hugging_face_transformer/train_hf_head.py` |
| **VAE Training**            | Latent representation learning               | `VAE/train_vae.py`                          |
| **Clustering & Evaluation** | Latent space visualization & metrics         | `clustering_utils.py`                       |

---

## ⚙️ Key Configuration Files

| File                                            | Purpose                            |
| ----------------------------------------------- | ---------------------------------- |
| `transformer/hyperParameters.json`              | Transformer AE hyperparameters     |
| `hugging_face_transformer/hyper_parameter.json` | HF transformer configuration       |
| `config.py`                                     | Global paths and directories       |
| `pyproject.toml`                                | Project metadata (for pip install) |
| `environment.lock.yml`                          | Exact dependency snapshot          |

---

## 📈 Outputs & Artifacts

| Artifact               | Description                       | Location                        |
| ---------------------- | --------------------------------- | ------------------------------- |
| 🤩 **Model Weights**   | Trained checkpoints (.pth)        | `checkpoints/`                  |
| 🔢 **Latent Features** | Encoded representations (.npy)    | `latent_features.npy`           |
| 📊 **Metrics JSON**    | Precision, Recall, F1, AUC        | `fault_classifier_metrics.json` |
| 📉 **Loss Curves**     | Training & validation loss        | `loss_validation_plot.png`      |
| 🔎 **Reconstructions** | Original vs reconstructed signals | `clustering_img/`               |

---

## 🧠 Reproducibility Tips

* Ensure identical preprocessing during training and inference
* Match normalization bounds (`feature_min`, `feature_max`)
* Verify checkpoint paths in `config.py`
* Use `environment.lock.yml` for exact dependencies

---

## 🌐 External Dependencies

This project integrates:

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [PredTrAD (Predictive Transformer for Anomaly Detection)](https://github.com/your-external-lib)
* [MLflow](https://mlflow.org/) for experiment tracking

---

## 🤝 Contribution

Contributions are welcome!
Please open an issue for bugs, ideas, or feature requests before submitting a pull request.

---

## 📞 Contact

**Author:** Sagar Sikdar
📧 Email: [sagar.sikdar@fau.de](mailto:sagar.sikdar@fau.de)
🌐 GitHub: [github.com/sagar-crypto](https://github.com/sagar-crypto)

---

## 🔮 Future Improvements

* 🧱 Unified CLI (`train --model transformer/vae/hf`)
* 🐳 Docker image for full reproducibility
* 📊 Visualization dashboards for clustering metrics
* ⚙️ Add Lightning or Hydra config management

---

## 📓 Table of Contents

* [Overview](#-overview)
* [Project Structure](#-project-structure)
* [Environment Setup](#%EF%B8%8F-environment-setup)
* [Quick Start](#-quick-start-commands)
* [Pipeline Overview](#-pipeline-overview)
* [Key Configuration Files](#-key-configuration-files)
* [Outputs](#-outputs--artifacts)
* [Reproducibility Tips](#-reproducibility-tips)
* [External Dependencies](#-external-dependencies)
* [Contribution](#-contribution)
* [Contact](#-contact)
* [Future Improvements](#-future-improvements)

---

### 🌟 TL;DR — 1-Line Setup

```bash
mamba env create -f environment.lock.yml && mamba activate prlabs-ts && pip install -e .
```

🚀 *You’re ready to train, cluster, and detect faults!*
