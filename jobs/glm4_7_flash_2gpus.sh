#!/bin/bash
#SBATCH --job-name=glm4.7-flash-2gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --time=${VLLM_CONFIG_SLURM_TIME:-12:00:00}
#SBATCH --partition=${VLLM_CONFIG_SLURM_PARTITION:-capella}
#SBATCH --mem=${VLLM_CONFIG_SLURM_MEM:-128G}
#SBATCH --output=logs/glm4_7_flash_2gpus_%j.out
#SBATCH --error=logs/glm4_7_flash_2gpus_%j.err

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

# Model to serve - GLM-4.7-Flash (30B-A3B MoE model, fits in 2 GPUs)
# Uses FP8 quantization for 8-bit inference
VLLM_MODEL="zai-org/GLM-4.7-Flash"
VLLM_PORT=8001

echo "Starting GLM-4.7-Flash on port $VLLM_PORT with 2 GPUs..."
echo "Model: $VLLM_MODEL"

# Use vLLM serve with FP8 quantization
# --quantization fp8 enables 8-bit quantization
vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice
  # --quantization fp8 \
