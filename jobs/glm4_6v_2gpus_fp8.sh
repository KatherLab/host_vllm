#!/bin/bash
#SBATCH --job-name=glm4.6v-2gpus-fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=capella
#SBATCH --output=logs/glm4_6v_2gpus_fp8_%j.out
#SBATCH --error=logs/glm4_6v_2gpus_fp8_%j.err

# Create logs directory
mkdir -p logs

# Load modules / set environment
module load CUDA
export XDG_CACHE_HOME=/data/horse/ws/s1787956-Cache
export TRITON_CACHE_DIR=/data/horse/ws/s1787956-Cache/triton
mkdir -p $TRITON_CACHE_DIR

# Activate virtual environment (adjust path if needed)
# NOTE: Requires vLLM>=0.12.0 and transformers>=5.0.0rc0 for GLM-4.6V:
# pip install vllm>=0.12.0
# pip install transformers>=5.0.0rc0
source $HOME/host_vllm/.venv/bin/activate
ulimit -n 16384

# Model to serve - GLM-4.6V (108B parameter vision-language MoE model)
# - Supports 128K token context
# - Native multimodal function calling
# - Interleaved image-text generation
# Uses FP8 quantization (2x H100 GPUs with 80GB each)
VLLM_MODEL="zai-org/GLM-4.6V-FP8"
VLLM_PORT=8004

echo "Starting GLM-4.6V on port $VLLM_PORT with 2 GPUs (FP8 quantization)..."
echo "Model: $VLLM_MODEL"

# Use vLLM serve with FP8 quantization
vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --tensor-parallel-size 2 \
  --quantization fp8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --trust-remote-code
