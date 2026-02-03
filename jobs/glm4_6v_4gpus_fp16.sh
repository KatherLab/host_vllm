#!/bin/bash
#SBATCH --job-name=glm4.6v-4gpus-fp16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=${VLLM_CONFIG_SLURM_TIME:-12:00:00}
#SBATCH --partition=${VLLM_CONFIG_SLURM_PARTITION:-capella}
#SBATCH --mem=${VLLM_CONFIG_SLURM_MEM:-256G}
#SBATCH --output=logs/glm4_6v_4gpus_fp16_%j.out
#SBATCH --error=logs/glm4_6v_4gpus_fp16_%j.err

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

# Model to serve - GLM-4.6V (108B parameter vision-language MoE model)
# This is the FULL PRECISION (16-bit) version - requires 4x H100 GPUs
# Use glm4_6v_2gpus_fp8.sh for 8-bit (2 GPUs) which is more efficient
VLLM_MODEL="zai-org/GLM-4.6V"
VLLM_PORT=8004

echo "Starting GLM-4.6V (FULL PRECISION) on port $VLLM_PORT with 4 GPUs..."
echo "Model: $VLLM_MODEL"
echo "WARNING: This uses 4x H100 GPUs. Consider using FP8 version with 2 GPUs instead."

# Use vLLM serve without quantization (full BF16 precision)
vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --trust-remote-code
