#!/bin/bash
#SBATCH --job-name=llama4-maverick-2nodes-8gpus
#SBATCH --nodes=2               # 2 nodes, each with 4 GPUs => 8 GPUs total
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=4
#SBATCH --partition=capella
#SBATCH --output=logs/llama4_maverick_2nodes_8gpus_%j.out
#SBATCH --error=logs/llama4_maverick_2nodes_8gpus_%j.err
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

# Model to serve - Llama-4-Maverick-17B-128E-Instruct (17B activated params, 400B total)
# - Mixture-of-Experts with 128 experts
# - 1M token context window (expandable)
# - Native multimodal (vision and text support)
# - FP8 quantization for 8-bit inference across 8 GPUs (2 nodes)
# Requirements: vllm>=0.15.0, transformers>=4.51.0, torch>=2.10.0

# Use config values if available, otherwise use defaults
VLLM_MODEL="${VLLM_CONFIG_HF_ID:-meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8}"
VLLM_PORT="${VLLM_CONFIG_PORT:-8006}"
TENSOR_PARALLEL_SIZE="${VLLM_CONFIG_GPUS:-8}"
GPU_MEMORY_UTIL="${VLLM_CONFIG_GPU_MEM:-0.90}"
MAX_MODEL_LEN="${VLLM_CONFIG_MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${VLLM_CONFIG_MAX_NUM_SEQS:-8}"
DTYPE="${VLLM_CONFIG_DTYPE:-auto}"

# Distributed configuration
RDZV_BACKEND="${VLLM_CONFIG_RDZV_BACKEND:-c10d}"
RDZV_TIMEOUT="${VLLM_CONFIG_RDZV_TIMEOUT:-1200}"
MASTER_PORT_BASE="${VLLM_CONFIG_MASTER_PORT:-29500}"

# Determine master address and port for distributed launch
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((MASTER_PORT_BASE + $SLURM_JOB_ID % 1000))

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Starting Llama-4-Maverick-17B-128E-Instruct on 2 nodes (8 GPUs) with FP8 quantization..."
echo "Model: $VLLM_MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"

# NCCL configuration for multi-node (use config or optimal defaults)
export NCCL_DEBUG="${VLLM_CONFIG_NCCL_DEBUG:-WARN}"
export NCCL_IB_TIMEOUT="${VLLM_CONFIG_NCCL_IB_TIMEOUT:-22}"
export NCCL_BLOCKSIZE="${VLLM_CONFIG_NCCL_BLOCKSIZE:-1048576}"
if [[ -n "$VLLM_CONFIG_NCCL_SOCKET_IFNAME" && "$VLLM_CONFIG_NCCL_SOCKET_IFNAME" != "null" ]]; then
    export NCCL_SOCKET_IFNAME="$VLLM_CONFIG_NCCL_SOCKET_IFNAME"
fi
# Disable IB if not available, optimize for Ethernet
export NCCL_IB_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_TIMEOUT=1800

# Launch vLLM server across the allocated nodes using srun and torchrun
# --tensor-parallel-size distributes model across all GPUs
# --rdzv_backend=c10d uses PyTorch's distributed rendezvous backend
# --rdzv_endpoint ensures all nodes coordinate properly
srun \
  --nodes=2 \
  --ntasks-per-node=1 \
  --gpus-per-task=4 \
  --exclusive \
  torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=$RDZV_BACKEND \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-gb 8 \
    --enable-auto-tool-choice \
    --tool-call-parser llama4_pythonic \
    --trust-remote-code