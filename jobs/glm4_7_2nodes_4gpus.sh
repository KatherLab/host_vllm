#!/bin/bash
#SBATCH --job-name=glm4.7-2nodes-4gpus
#SBATCH --nodes=2               # 2 nodes, each with 4 GPUs => 8 GPUs total
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=capella
#SBATCH --output=logs/glm4_7_2nodes_4gpus_%j.out
#SBATCH --error=logs/glm4_7_2nodes_4gpus_%j.err

# Create logs directory
mkdir -p logs

# Load CUDA module
module load CUDA

export XDG_CACHE_HOME=/data/horse/ws/s1787956-Cache
export TRITON_CACHE_DIR=/data/horse/ws/s1787956-Cache/triton
mkdir -p $TRITON_CACHE_DIR

# Activate virtual environment (adjust path if needed)
# NOTE: Requires vLLM nightly with GLM-4.7 support:
# pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly
source $HOME/host_vllm/.venv/bin/activate
ulimit -n 16384

# Model to serve - GLM-4.7 (358B parameters MoE model)
# Uses FP8 quantization for 8-bit inference - requires 8 GPUs (2 nodes x 4 GPUs)
VLLM_MODEL="zai-org/GLM-4.7"
VLLM_PORT=8002

# Determine master address and port for distributed launch
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((12345 + $SLURM_JOB_ID % 1000))

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Starting GLM-4.7 (358B MoE) on 2 nodes with FP8 quantization..."
echo "Model: $VLLM_MODEL"

# Better NCCL configuration for multi-node
export NCCL_IB_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_TIMEOUT=1800

# Launch vLLM server across the allocated nodes using srun and torchrun
srun \
  --nodes=2 \
  --ntasks-per-node=1 \
  --gpus-per-task=4 \
  --cpus-per-task=16 \
  --exclusive \
  torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size 8 \
    --quantization fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice