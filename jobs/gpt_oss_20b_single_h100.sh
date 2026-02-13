#!/bin/bash
#SBATCH --job-name=gpt-oss-20b-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=${VLLM_CONFIG_SLURM_TIME:-24:00:00}
#SBATCH --partition=${VLLM_CONFIG_SLURM_PARTITION:-capella}
#SBATCH --mem=${VLLM_CONFIG_SLURM_MEM:-48G}
#SBATCH --output=logs/gpt_oss_20b_%j.out
#SBATCH --error=logs/gpt_oss_20b_%j.err

# Create logs directory
mkdir -p logs

# Load modules / set environment
module load CUDA
export XDG_CACHE_HOME="${VLLM_CONFIG_CACHE_DIR:-/tmp}"
export TRITON_CACHE_DIR="${VLLM_CONFIG_CACHE_DIR:-/tmp}/triton"
mkdir -p "$TRITON_CACHE_DIR"

# Activate virtual environment
source "${VLLM_CONFIG_VENV_DIR:-$HOME/host_vllm/.venv}/bin/activate"
ulimit -n 16384

# Model to serve - GPT-OSS 20B (24GB VRAM requirement with MXFP4 quantization)
# The model uses built-in MXFP4 quantization from post-training
VLLM_MODEL="openai/gpt-oss-20b"
VLLM_PORT=8000

echo "Starting GPT-OSS 20B on port $VLLM_PORT..."
echo "Model: $VLLM_MODEL"

# Use vllm serve command (recommended for GPT-OSS)
vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  --reasoning_parser openai_gptoss \
  --trust-remote-code
