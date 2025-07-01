from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = "/home/vault/iwi5/iwi5305h/new_dataset_90kv/all"
CHECKPOINT_VAE_DIR = f"{PROJECT_ROOT}/checkpoints/VAE"
CHECKPOINT_TRANSFORMERS_DIR = f"{PROJECT_ROOT}/checkpoints/transformers"
MODELS_DIR = f"{PROJECT_ROOT}/models"
VAE_DIR = f"{PROJECT_ROOT}/VAE"
TRANSFORMERS_DIR = f"{PROJECT_ROOT}/transformer"
SCALER_TRANSFORMERS_DIR = f"{PROJECT_ROOT}/scalers/transformer"
SCALER_VAE_DIR = f"{PROJECT_ROOT}/scalers/VAE"
CLUSTERING_TRANSFORMERS_DIR = f"{PROJECT_ROOT}/clustering_img/Transformer_Cluster_Img"
CLUSTERING_VAE_DIR = f"{PROJECT_ROOT}/clustering_img/VAE_Cluster_img"
