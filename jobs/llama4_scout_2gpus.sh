#!/bin/bash
#SBATCH --job-name=llama4-scout-2gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --time=${VLLM_CONFIG_SLURM_TIME:-12:00:00}
#SBATCH --partition=${VLLM_CONFIG_SLURM_PARTITION:-capella}
#SBATCH --mem=${VLLM_CONFIG_SLURM_MEM:-256G}
#SBATCH --output=logs/llama4_scout_2gpus_%j.out
#SBATCH --error=logs/llama4_scout_2gpus_%j.err

# Create logs directory
mkdir -p logs

# Load CUDA module
module load CUDA

export XDG_CACHE_HOME="${VLLM_CONFIG_CACHE_DIR:-/tmp}"
export TRITON_CACHE_DIR="${VLLM_CONFIG_CACHE_DIR:-/tmp}/triton"
mkdir -p "$TRITON_CACHE_DIR"

# Activate virtual environment
source "${VLLM_CONFIG_VENV_DIR:-$HOME/host_vllm/.venv}/bin/activate"
ulimit -n 16384

# Model to serve - Llama-4-Scout-17B-16E-Instruct (17B activated params, 109B total)
# - Mixture-of-Experts with 16 experts
# - 10M token context window
# - Native multimodal (vision and text support)
# - FP8 quantization for 8-bit inference on 2 GPUs
VLLM_MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct-FP8"
VLLM_PORT=8005

echo "Starting Llama-4-Scout-17B-16E-Instruct on port $VLLM_PORT with 2 GPUs (FP8 quantization)..."
echo "Model: $VLLM_MODEL"

# Use vLLM serve with FP8 quantization
vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 8 \
  --enable-auto-tool-choice \
  --tool-call-parser llama4_pythonic \
  --trust-remote-code