#!/bin/bash -l
#
# SLURM Job Script for Optuna Hyperparameter Optimization of FixMatch Model
#
# ===============================
# SLURM Directives
# ===============================
#SBATCH --gres=gpu:v100:1               # Request 1 NVIDIA A100 GPU
#SBATCH --partition=v100                # Specify the GPU partition
#SBATCH --time=24:00:00                 # Maximum runtime of 24 hours
#SBATCH --export=NONE                   # Do not export current environment variables
#SBATCH --job-name=v100-train-code         # Job name
#SBATCH --output=v100-train-code.out      # Standard output log file (%j expands to job ID)
#SBATCH --error=v100-train-code.err       # Standard error log file (%j expands to job ID)
 
# ===============================
# Environment Configuration
# ===============================
 
# Set HTTP and HTTPS proxies if required (uncomment and modify if needed)
# export HTTP_PROXY=http://proxy:80
# export HTTPS_PROXY=http://proxy:80
 
# Unset SLURM_EXPORT_ENV to prevent SLURM from exporting environment variables
unset SLURM_EXPORT_ENV
 
# Load the necessary modules
module load python/3.12-conda        # Load Python Anaconda module
 
# Activate the Conda environment
conda activate /home/hpc/iwi5/iwi5305h/miniconda3/envs/PR_LAB_PROJECT  # Replace with your Conda environment path
 
# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5305h/research_project_PR_labs  # Replace with your script directory path
 
# ===============================
# Execute the Python Training Script
# ===============================
 
# Run the Optuna-based FixMatch HPO Python script with necessary arguments
python3 train.py
