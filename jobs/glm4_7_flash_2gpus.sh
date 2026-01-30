#!/bin/bash
#SBATCH --job-name=glm4.7-flash-2gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=capella
#SBATCH --output=logs/glm4_7_flash_2gpus_%j.out
#SBATCH --error=logs/glm4_7_flash_2gpus_%j.err

# Create logs directory
mkdir -p logs

# Load modules / set environment
module load CUDA
export XDG_CACHE_HOME=/data/horse/ws/s1787956-Cache
export TRITON_CACHE_DIR=/data/horse/ws/s1787956-Cache/triton
mkdir -p $TRITON_CACHE_DIR

# Activate virtual environment (adjust path if needed)
# NOTE: Requires vLLM nightly with GLM-4.7 support:
# pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly
source $HOME/host_vllm/.venv/bin/activate
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
  --quantization fp8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice
