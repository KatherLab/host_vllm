#!/bin/bash
#SBATCH --job-name=glm4.6v-2gpus-fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=capella
#SBATCH --output=logs/glm4_6v_2gpus_fp8_%j.out
#SBATCH --error=logs/glm4_6v_2gpus_fp8_%j.err
# Note: --account, --time, --cpus-per-task, --mem-per-gpu set by run.sh via sbatch options

# Create logs directory
mkdir -p logs

# Load CUDA module
module load CUDA

# Set cache directories from config or use defaults
if [[ -n "$VLLM_CONFIG_CACHE_XDG" && "$VLLM_CONFIG_CACHE_XDG" != "null" ]]; then
    export XDG_CACHE_HOME="$VLLM_CONFIG_CACHE_XDG"
fi
if [[ -n "$VLLM_CONFIG_CACHE_TRITON" && "$VLLM_CONFIG_CACHE_TRITON" != "null" ]]; then
    export TRITON_CACHE_DIR="$VLLM_CONFIG_CACHE_TRITON"
elif [[ -n "$XDG_CACHE_HOME" ]]; then
    export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
fi
if [[ -n "$VLLM_CONFIG_CACHE_HF" && "$VLLM_CONFIG_CACHE_HF" != "null" ]]; then
    export HF_HOME="$VLLM_CONFIG_CACHE_HF"
fi
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

# Activate virtual environment from config or detect from current environment
if [[ -n "$VLLM_CONFIG_VENV_DIR" && "$VLLM_CONFIG_VENV_DIR" != "null" ]]; then
    source "$VLLM_CONFIG_VENV_DIR/bin/activate"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    source "$VIRTUAL_ENV/bin/activate"
elif [[ -f "$HOME/host_vllm/.venv/bin/activate" ]]; then
    source "$HOME/host_vllm/.venv/bin/activate"
else
    echo "Warning: No virtual environment found. Using system Python."
fi
ulimit -n 16384

# Model to serve - GLM-4.6V (108B parameter vision-language MoE model)
# - Supports 128K token context
# - Native multimodal function calling
# - Interleaved image-text generation
# - FP8 quantization (2x H100 GPUs with 80GB each)
# Requirements: vllm>=0.15.0, transformers>=4.51.0

# Use config values if available, otherwise use defaults
VLLM_MODEL="${VLLM_CONFIG_HF_ID:-zai-org/GLM-4.6V-FP8}"
VLLM_PORT="${VLLM_CONFIG_PORT:-8004}"
TENSOR_PARALLEL_SIZE="${VLLM_CONFIG_GPUS:-2}"
GPU_MEMORY_UTIL="${VLLM_CONFIG_GPU_MEM:-0.90}"
MAX_MODEL_LEN="${VLLM_CONFIG_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${VLLM_CONFIG_MAX_NUM_SEQS:-16}"
DTYPE="${VLLM_CONFIG_DTYPE:-auto}"

echo "Starting GLM-4.6V on port $VLLM_PORT with $TENSOR_PARALLEL_SIZE GPUs (FP8 quantization)..."
echo "Model: $VLLM_MODEL"

# Use vLLM serve with FP8 quantization
vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTIL \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --enable-auto-tool-choice \
  --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --dtype $DTYPE \
  --trust-remote-code
