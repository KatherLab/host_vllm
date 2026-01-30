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

# Fallback to global SLURM settings if not defined per-model
if [[ -z "$JOB_TIME" || "$JOB_TIME" == "null" ]]; then
    JOB_TIME=$(get_config ".slurm.time_default")
fi
if [[ -z "$JOB_PARTITION" || "$JOB_PARTITION" == "null" ]]; then
    JOB_PARTITION=$(get_config ".slurm.partition")
fi

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
echo "  Job ID: $JOB_ID"
echo ""
echo -e "${BLUE}Monitoring job...${NC}"
echo "  Command: squeue -j $JOB_ID"
echo ""

# Wait for job to start (optional monitoring)
WAIT_TIMEOUT=300  # 5 minutes
WAITED=0
echo "Waiting for job to start..."
while squeue -j "$JOB_ID" -h | grep -q "PD"; do
    if [[ $WAITED -ge $WAIT_TIMEOUT ]]; then
        echo -e "${YELLOW}Timeout waiting for job to start. Check with: squeue -j $JOB_ID${NC}"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Still pending... ($WAITED s)"
done

# Get job info
if squeue -j "$JOB_ID" -h | grep -q "R"; then
    NODE=$(squeue -j "$JOB_ID" -h -o "%N")
    echo ""
    echo -e "${GREEN}Job is now running!${NC}"
    echo "  Node: $NODE"
    echo "  Port: $PORT"
    echo ""
    echo -e "${BLUE}To query the model:${NC}"
    echo "  export VLLM_HOST=$NODE"
    
    # Use custom query script if defined, otherwise default
    if [[ -n "$QUERY_SCRIPT" && "$QUERY_SCRIPT" != "null" && -f "$SCRIPT_DIR/$QUERY_SCRIPT" ]]; then
        echo "  $SCRIPT_DIR/$QUERY_SCRIPT"
    else
        echo "  python main.py --model $ACTIVE_MODEL --host $NODE --prompt \"Your prompt here\""
    fi
    echo ""
    echo -e "${BLUE}To check logs:${NC}"
    echo "  tail -f logs/${ACTIVE_MODEL}_*.out"
else
    echo ""
    echo -e "${YELLOW}Job status unknown. Check with:${NC}"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/${ACTIVE_MODEL}_*.out"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
