# VLLM Host - HPC Model Serving

This repo provides a complete workflow for hosting large AI models on HPC clusters using vLLM and SLURM.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Enter repo
cd /data/horse/ws/s1787956-host_vllm/code/host_vllm

# Install dependencies
uv sync

# Make run script executable
chmod +x run.sh
```

### 2. Configure

Edit `config.yaml` and set `active_model`:

```yaml
# Options: gpt-oss-20b, gpt-oss-120b, glm-4.7-flash, glm-4.7, glm-4.6v, glm-4.6v-fp16, qwen3-vl-235b
active_model: glm-4.6v  # Default: 8-bit vision model on 2 GPUs
```

### 3. Launch Model Server

```bash
./run.sh                    # Use default config.yaml
./run.sh -c my_config.yaml  # Use custom config file
./run.sh -h                 # Show help

# This will:
# - Read the config file
# - Submit the appropriate SLURM job
# - Show the job ID and node assignment
```

### 4. Query the Model

Once the job is running:

```bash
# Text-only query
python main.py --model glm-4.6v --prompt "Describe the solar system"

# With image (for vision models)
python main.py --model glm-4.6v --prompt "What's in this image?" --image-url "https://..."

# With local image
python main.py --model glm-4.6v --prompt "Analyze this" --image /path/to/photo.jpg

# List all available models
python main.py --list-models
```

## 📁 Directory Structure

```
.
├── config.yaml           # Main configuration (default)
├── run.sh               # Launcher script
├── main.py              # Example client
├── pyproject.toml       # Dependencies
├── jobs/                # SLURM job scripts
│   ├── gpt_oss_20b_single_h100.sh
│   ├── gpt_oss_120b_single_h100.sh
│   ├── glm4_7_flash_2gpus.sh
│   ├── glm4_7_2nodes_4gpus.sh
│   ├── glm4_6v_2gpus_fp8.sh      # GLM-4.6V (8-bit, 2 GPUs)
│   ├── glm4_6v_4gpus_fp16.sh     # GLM-4.6V (16-bit, 4 GPUs)
│   └── qwen3_vl_235b_4nodes.sh
├── scripts/             # Query scripts
│   ├── query_openai_compatible.py  # Default query script
│   └── README.md
└── logs/                # Job output logs
```

## 🎯 Available Models

| Model | HuggingFace ID | GPUs | Vision | Quantization |
|-------|----------------|------|--------|--------------|
| gpt-oss-20b | `openai/gpt-oss-20b` | 1 | ❌ | MXFP4 |
| gpt-oss-120b | `openai/gpt-oss-120b` | 1 | ❌ | MXFP4 |
| glm-4.7-flash | `zai-org/GLM-4.7-Flash` | 2 | ❌ | FP8 |
| glm-4.7 | `zai-org/GLM-4.7` | 8 (2 nodes) | ❌ | FP8 |
| **glm-4.6v** | `zai-org/GLM-4.6V-FP8` | **2** | **✅** | FP8 |
| glm-4.6v-fp16 | `zai-org/GLM-4.6V` | 4 | ✅ | BF16 |
| qwen3-vl-235b | `Qwen/Qwen3-VL-235B-A22B-Thinking` | 16 (4 nodes) | ✅ | FP8 |

## ⚙️ Precision Configuration

8-bit precision (FP8/MXFP4) is used by default for efficiency. For 16-bit precision, use double the GPUs:

```yaml
# 8-bit (default) - 2 GPUs
active_model: glm-4.6v

# 16-bit - 4 GPUs
active_model: glm-4.6v-fp16
```

## 📝 Complete Example

```bash
# 1. Setup
uv sync

# 2. Configure for vision model
# Edit config.yaml: active_model: glm-4.6v

# 3. Launch
./run.sh
# Output: Job 12345 submitted, running on node hpc-node-01

# 4. Query
export VLLM_HOST=hpc-node-01
python main.py --model glm-4.6v --prompt "What do you see?" --image-url "https://example.com/image.jpg"

# 5. Monitor
squeue -u $USER
tail -f logs/glm4_6v_2gpus_fp8_*.out
```

## 📄 Custom Configuration Files

You can use multiple configuration files and select them at runtime:

```bash
# Create multiple configs
cp config.yaml production.yaml
cp config.yaml development.yaml

# Edit production.yaml to use different model or SLURM settings

# Run with custom config
./run.sh -c production.yaml
./run.sh -c development.yaml
```

## 🔧 Configuration Reference

### Model Configuration

Each model in `config.yaml` supports:

```yaml
models:
  my-model:
    name: "Display Name"
    huggingface_id: "org/model-id"
    job_script: "jobs/my_model.sh"           # SLURM job script
    query_script: "scripts/query.py"         # Optional: custom query script
    port: 8000                               # vLLM server port
    gpus: 1                                  # GPUs required
    nodes: 1                                 # Nodes required
    quantization: "fp8"                      # Quantization mode
    precision: "8bit"                        # Precision (8bit/16bit)
    vision: false                            # Vision model support
    # Optional: Override global SLURM settings per model
    time: "24:00:00"                         # Job time limit (default from slurm.time_default)
    partition: "long"                        # SLURM partition (default from slurm.partition)
    description: "Description for display"
```

### SLURM Configuration

```yaml
slurm:
  partition: "capella"     # SLURM partition
  account: ""              # SLURM account (optional)
  time_default: "12:00:00" # Default walltime
  cpus_per_task: 16        # CPUs per task
```

## 🧩 Query Scripts

Models can define custom `query_script` in config. Default is `scripts/query_openai_compatible.py`.

To create a custom query script:
1. Create a script in `scripts/` directory
2. Make it executable: `chmod +x scripts/my_script.py`
3. Reference it in your model config

See `scripts/README.md` for more details.