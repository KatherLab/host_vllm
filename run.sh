#!/bin/bash
# Run script for VLLM-Host
# Automatically submits SLURM jobs based on config.yaml
#
# Usage: ./run.sh [-c <config_file>]
#   -c <config_file>  Use custom config file (default: config.yaml)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# Parse command line arguments
while getopts ":c:h" opt; do
    case $opt in
        c)
            CONFIG_FILE="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [-c <config_file>]"
            echo "  -c <config_file>  Use custom config file (default: config.yaml)"
            echo "  -h                Show this help message"
            exit 0
            ;;
        \?)
            echo -e "${RED}Error: Invalid option -$OPTARG${NC}" >&2
            echo "Usage: $0 [-c <config_file>]"
            exit 1
            ;;
        :)
            echo -e "${RED}Error: Option -$OPTARG requires an argument.${NC}" >&2
            exit 1
            ;;
    esac
done

# Check if config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Parse active_model from config (simple YAML parsing)
# Requires yq or uses grep fallback
if command -v yq &> /dev/null; then
    ACTIVE_MODEL=$(yq -r '.active_model' "$CONFIG_FILE")
else
    # Fallback grep parsing
    ACTIVE_MODEL=$(grep "^active_model:" "$CONFIG_FILE" | sed 's/active_model://' | tr -d ' "')
fi

if [[ -z "$ACTIVE_MODEL" || "$ACTIVE_MODEL" == "null" ]]; then
    echo -e "${RED}Error: No active_model set in config.yaml${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  VLLM-Host Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Active model: ${GREEN}$ACTIVE_MODEL${NC}"

# Function to extract config values
# Supports both yq and a simple fallback for basic YAML parsing
# Usage: get_config ".models.modelname.field" or "active_model"
get_config() {
    local key="$1"
    if command -v yq &> /dev/null; then
        yq -r "$key" "$CONFIG_FILE"
    else
        # Fallback: handle nested keys like .models.<name>.field
        # Parse the key: .models.<model_name>.<field>
        local model_name=$(echo "$key" | sed -n 's/\.models\.\([^\.]*\)\..*/\1/p')
        local field=$(echo "$key" | sed 's/.*\.//')
        
        if [[ -n "$model_name" ]]; then
            # Looking for a field in a model: find model section, then get field
            awk -v model="$model_name" -v field="$field" '
                /^models:/ { in_models=1; next }
                in_models && $0 ~ "^  " model ":" { in_model=1; next }
                in_models && in_model && /^  [a-z]/ { in_model=0; next }
                in_model && $0 ~ "^    " field ":" {
                    $1=""; 
                    sub(/^[ 	:]/, "");
                    gsub(/^[" ]+|[" ]+$/, "");
                    print;
                    exit
                }
            ' "$CONFIG_FILE"
        else
            # Top-level key
            grep -E "^${key}:" "$CONFIG_FILE" | head -1 | sed "s/^${key}://" | tr -d ' "' | sed 's/#.*//'
        fi
    fi
}

# Get model configuration
JOB_SCRIPT=$(get_config ".models.${ACTIVE_MODEL}.job_script")
QUERY_SCRIPT=$(get_config ".models.${ACTIVE_MODEL}.query_script")
MODEL_NAME=$(get_config ".models.${ACTIVE_MODEL}.name")
HF_ID=$(get_config ".models.${ACTIVE_MODEL}.huggingface_id")
PORT=$(get_config ".models.${ACTIVE_MODEL}.port")
GPUS=$(get_config ".models.${ACTIVE_MODEL}.gpus")
NODES=$(get_config ".models.${ACTIVE_MODEL}.nodes")
PRECISION=$(get_config ".models.${ACTIVE_MODEL}.precision")
VISION=$(get_config ".models.${ACTIVE_MODEL}.vision")
DESC=$(get_config ".models.${ACTIVE_MODEL}.description")

# Get optional model-specific SLURM settings (fallback to global slurm settings)
JOB_TIME=$(get_config ".models.${ACTIVE_MODEL}.time")
JOB_PARTITION=$(get_config ".models.${ACTIVE_MODEL}.partition")

# Get global SLURM configuration
SLURM_ACCOUNT=$(get_config ".slurm.account")
SLURM_CPUS_PER_TASK=$(get_config ".slurm.cpus_per_task")
SLURM_MEM_PER_GPU=$(get_config ".slurm.mem_per_gpu")

# Get cache directories
CACHE_XDG=$(get_config ".cache.xdg_cache_home")
CACHE_TRITON=$(get_config ".cache.triton_cache_dir")
CACHE_HF=$(get_config ".cache.huggingface_cache")

# Get venv path
VENV_DIR=$(get_config ".paths.venv_dir")

# Get distributed settings
RDZV_BACKEND=$(get_config ".distributed.rdzv_backend")
RDZV_TIMEOUT=$(get_config ".distributed.rdzv_timeout")
MASTER_PORT_BASE=$(get_config ".distributed.master_port_base")

# Get NCCL settings
NCCL_SOCKET_IFNAME=$(get_config ".nccl.socket_ifname")
NCCL_BLOCKSIZE=$(get_config ".nccl.blocksize")
NCCL_IB_TIMEOUT=$(get_config ".nccl.ib_timeout")
NCCL_DEBUG=$(get_config ".nccl.debug")

# Get vLLM settings
VLLM_GPU_MEM=$(get_config ".vllm.gpu_memory_utilization")
VLLM_MAX_MODEL_LEN=$(get_config ".vllm.max_model_len_default")
VLLM_MAX_NUM_SEQS=$(get_config ".vllm.max_num_seqs")
VLLM_DTYPE=$(get_config ".vllm.dtype")

# Fallback to global SLURM settings if not defined per-model
if [[ -z "$JOB_TIME" || "$JOB_TIME" == "null" ]]; then
    JOB_TIME=$(get_config ".slurm.time_default")
fi
if [[ -z "$JOB_PARTITION" || "$JOB_PARTITION" == "null" ]]; then
    JOB_PARTITION=$(get_config ".slurm.partition")
fi

# Set defaults for optional configs
[[ -z "$RDZV_BACKEND" || "$RDZV_BACKEND" == "null" ]] && RDZV_BACKEND="c10d"
[[ -z "$RDZV_TIMEOUT" || "$RDZV_TIMEOUT" == "null" ]] && RDZV_TIMEOUT="1200"
[[ -z "$MASTER_PORT_BASE" || "$MASTER_PORT_BASE" == "null" ]] && MASTER_PORT_BASE="29500"
[[ -z "$NCCL_BLOCKSIZE" || "$NCCL_BLOCKSIZE" == "null" ]] && NCCL_BLOCKSIZE="1048576"
[[ -z "$NCCL_IB_TIMEOUT" || "$NCCL_IB_TIMEOUT" == "null" ]] && NCCL_IB_TIMEOUT="22"
[[ -z "$NCCL_DEBUG" || "$NCCL_DEBUG" == "null" ]] && NCCL_DEBUG="WARN"
[[ -z "$VLLM_GPU_MEM" || "$VLLM_GPU_MEM" == "null" ]] && VLLM_GPU_MEM="0.90"
[[ -z "$VLLM_MAX_MODEL_LEN" || "$VLLM_MAX_MODEL_LEN" == "null" ]] && VLLM_MAX_MODEL_LEN="32768"
[[ -z "$VLLM_MAX_NUM_SEQS" || "$VLLM_MAX_NUM_SEQS" == "null" ]] && VLLM_MAX_NUM_SEQS="16"
[[ -z "$VLLM_DTYPE" || "$VLLM_DTYPE" == "null" ]] && VLLM_DTYPE="auto"
[[ -z "$SLURM_CPUS_PER_TASK" || "$SLURM_CPUS_PER_TASK" == "null" ]] && SLURM_CPUS_PER_TASK="16"

# Validate
if [[ -z "$JOB_SCRIPT" || "$JOB_SCRIPT" == "null" ]]; then
    echo -e "${RED}Error: Model '$ACTIVE_MODEL' not found in config${NC}"
    echo "Available models:"
    if command -v yq &> /dev/null; then
        yq -r '.models | keys | .[]' "$CONFIG_FILE" | sed 's/^/  - /'
    else
        # List model names (lines with 2-space indent followed by word chars, ending in colon)
        awk '/^models:/{found=1; next} found && /^$/{exit} found && /^  [a-z0-9_-]+:/{print "  - " $1}' "$CONFIG_FILE" | sed 's/://g'
    fi
    exit 1
fi

JOB_SCRIPT="$SCRIPT_DIR/$JOB_SCRIPT"

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo -e "${RED}Error: Job script not found: $JOB_SCRIPT${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Model Details:${NC}"
echo "  Name:        $MODEL_NAME"
echo "  HuggingFace: $HF_ID"
echo "  GPUs:        $GPUS"
echo "  Nodes:       $NODES"
echo "  Port:        $PORT"
echo "  Precision:   $PRECISION"
echo "  Vision:      $VISION"
echo "  Description: $DESC"
echo ""
echo -e "${YELLOW}Job Configuration:${NC}"
echo "  Script:      $JOB_SCRIPT"
if [[ -n "$JOB_TIME" && "$JOB_TIME" != "null" ]]; then
    echo "  Time:        $JOB_TIME"
fi
if [[ -n "$JOB_PARTITION" && "$JOB_PARTITION" != "null" ]]; then
    echo "  Partition:   $JOB_PARTITION"
fi
echo ""

# Check if job is already running
check_running() {
    local pattern="$1"
    squeue -u "$USER" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" | grep -q "$pattern"
}

if check_running "$ACTIVE_MODEL"; then
    echo -e "${YELLOW}Warning: A job for '$ACTIVE_MODEL' is already running${NC}"
    squeue -u "$USER" | grep "$ACTIVE_MODEL"
    echo ""
    read -p "Submit another job? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Canceled."
        exit 0
    fi
fi

# Check if port is already in use on any node
echo -e "${BLUE}Checking port availability...${NC}"

# Build sbatch options for overrides
SBATCH_OPTS=""
if [[ -n "$JOB_TIME" && "$JOB_TIME" != "null" ]]; then
    SBATCH_OPTS="$SBATCH_OPTS --time=$JOB_TIME"
fi
if [[ -n "$JOB_PARTITION" && "$JOB_PARTITION" != "null" ]]; then
    SBATCH_OPTS="$SBATCH_OPTS --partition=$JOB_PARTITION"
fi
if [[ -n "$SLURM_ACCOUNT" && "$SLURM_ACCOUNT" != "null" ]]; then
    SBATCH_OPTS="$SBATCH_OPTS --account=$SLURM_ACCOUNT"
fi
if [[ -n "$SLURM_CPUS_PER_TASK" && "$SLURM_CPUS_PER_TASK" != "null" ]]; then
    SBATCH_OPTS="$SBATCH_OPTS --cpus-per-task=$SLURM_CPUS_PER_TASK"
fi
if [[ -n "$SLURM_MEM_PER_GPU" && "$SLURM_MEM_PER_GPU" != "null" ]]; then
    SBATCH_OPTS="$SBATCH_OPTS --mem-per-gpu=$SLURM_MEM_PER_GPU"
fi

# Export configuration as environment variables for job script
export VLLM_CONFIG_CACHE_XDG="$CACHE_XDG"
export VLLM_CONFIG_CACHE_TRITON="$CACHE_TRITON"
export VLLM_CONFIG_CACHE_HF="$CACHE_HF"
export VLLM_CONFIG_VENV_DIR="$VENV_DIR"
export VLLM_CONFIG_RDZV_BACKEND="$RDZV_BACKEND"
export VLLM_CONFIG_RDZV_TIMEOUT="$RDZV_TIMEOUT"
export VLLM_CONFIG_MASTER_PORT="$MASTER_PORT_BASE"
export VLLM_CONFIG_NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export VLLM_CONFIG_NCCL_BLOCKSIZE="$NCCL_BLOCKSIZE"
export VLLM_CONFIG_NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT"
export VLLM_CONFIG_NCCL_DEBUG="$NCCL_DEBUG"
export VLLM_CONFIG_GPU_MEM="$VLLM_GPU_MEM"
export VLLM_CONFIG_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN"
export VLLM_CONFIG_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS"
export VLLM_CONFIG_DTYPE="$VLLM_DTYPE"
export VLLM_CONFIG_MODEL="$ACTIVE_MODEL"
export VLLM_CONFIG_HF_ID="$HF_ID"
export VLLM_CONFIG_PORT="$PORT"
export VLLM_CONFIG_GPUS="$GPUS"
export VLLM_CONFIG_NODES="$NODES"

# Submit the job
echo ""
echo -e "${GREEN}Submitting SLURM job...${NC}"

if [[ -n "$SBATCH_OPTS" ]]; then
    echo -e "${BLUE}SLURM overrides:${NC}$SBATCH_OPTS"
    JOB_ID=$(sbatch $SBATCH_OPTS "$JOB_SCRIPT" 2>&1 | grep -oP "^Submitted batch job \K[0-9]+")
else
    JOB_ID=$(sbatch "$JOB_SCRIPT" 2>&1 | grep -oP "^Submitted batch job \K[0-9]+")
fi

if [[ -z "$JOB_ID" ]]; then
    echo -e "${RED}Error: Failed to submit job${NC}"
    exit 1
fi

echo -e "${GREEN}Job submitted successfully!${NC}"
echo ""
echo -e "${BLUE}Job Information:${NC}"
echo "  Job ID:       $JOB_ID"
echo "  Job Script:   $JOB_SCRIPT"
echo "  Model:        $ACTIVE_MODEL ($MODEL_NAME)"
echo "  Port:         $PORT"
echo "  Nodes:        $NODES"
echo "  GPUs:         $GPUS"
if [[ -n "$JOB_TIME" && "$JOB_TIME" != "null" ]]; then
    echo "  Time Limit:   $JOB_TIME"
fi
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "  Check status: ${GREEN}squeue -j $JOB_ID${NC}"
echo "  View logs:    ${GREEN}tail -f logs/${ACTIVE_MODEL}_*.out${NC}"
echo "  Cancel job:   ${GREEN}scancel $JOB_ID${NC}"
echo ""
echo -e "${YELLOW}Note: Job may take time to start. Monitor with: squeue -j $JOB_ID${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
