#!/bin/bash
#SBATCH --job-name=sae
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgzzsn2025-gpu-a100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out

pwd;hostname;date

mkdir -p logs

source .env
eval "$(conda shell.bash hook)"
conda activate sae-diff-autointerpr
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# CUDA
module load CUDA/12.4.0
echo $CUDA_HOME

HOSTNAME=$(hostname)
PORT=8888
echo "Jupyter Notebook server starting on $HOSTNAME:$PORT"
jupyter notebook --no-browser --ip=$HOSTNAME --port=$PORT