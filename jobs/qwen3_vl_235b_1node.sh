#!/bin/bash
#SBATCH --job-name=qwen3-vl-235b
#SBATCH --nodes=1                     # 1 node with 4 GPUs
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=capella
#SBATCH --output=logs/qwen3_vl_235b_%j.out
#SBATCH --error=logs/qwen3_vl_235b_%j.err

# Create logs directory
mkdir -p logs

# Load CUDA module
module load CUDA

export XDG_CACHE_HOME=/data/horse/ws/s1787956-Cache
export TRITON_CACHE_DIR=/data/horse/ws/s1787956-Cache/triton
mkdir -p $TRITON_CACHE_DIR

# Activate virtual environment (adjust path if needed)
# NOTE: Qwen3-VL requires latest transformers from source:
# pip install git+https://github.com/huggingface/transformers
source $HOME/host_vllm/.venv/bin/activate
ulimit -n 16384

# Model to serve - Qwen3-VL-235B-A22B-Thinking (236B MoE vision-language model)
# - 235B total parameters, 22B active parameters
# - Supports native 256K context, expandable to 1M
# - Visual agent capabilities, advanced spatial perception, video understanding
VLLM_MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
VLLM_PORT=8003

# Determine master address and port for distributed launch
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((12345 + $SLURM_JOB_ID % 1000))

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Starting Qwen3-VL-235B-A22B-Thinking on 1 node (4 GPUs) with FP8 quantization..."
echo "Model: $VLLM_MODEL"

# Better NCCL configuration for multi-node
export NCCL_IB_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_TIMEOUT=1800

# Launch vLLM server across the allocated nodes using torchrun (distributed)
srun \
  --nodes=4 \
  --ntasks-per-node=1 \
  --gpus-per-task=4 \
  --cpus-per-task=16 \
  --exclusive \
  torchrun \
    --nnodes=4 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size 16 \
    --quantization fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --trust-remote-code
