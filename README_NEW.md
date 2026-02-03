# vLLM Host - Multi-Node LLM Deployment System

A comprehensive SLURM-based deployment system for running large language models with vLLM on HPC clusters, supporting multi-node distributed inference with proper vLLM v0.15.0 configurations.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Multi-Node Deployment](#multi-node-deployment)
- [Available Models](#available-models)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Features

- **Multi-Node Support**: Distributed inference across multiple SLURM nodes using torchrun
- **vLLM v0.15.0**: Latest vLLM features including improved MoE support, FP8 quantization, and multi-modal capabilities
- **Centralized Configuration**: All settings in `config.yaml` - no hardcoded paths in job scripts
- **SLURM Integration**: Automatic job submission with configurable time limits, partitions, and accounts
- **Multiple Models**: Pre-configured support for GPT-OSS, GLM-4, Qwen3-VL, Llama-4 models
- **Quantization Support**: FP8, MXFP4, BF16 quantization for efficient inference
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints

## Requirements

### Software Requirements

```bash
# Python >= 3.10
python --version

# Required Python packages
vllm>=0.15.0           # Latest vLLM with multi-node improvements
transformers>=4.51.0   # For Llama-4 and latest model support
torch>=2.10.0          # PyTorch with distributed improvements
```

### Hardware Requirements

- **GPU**: NVIDIA H100 (80GB) or A100 (40GB/80GB) recommended
- **Network**: InfiniBand or high-speed Ethernet for multi-node deployments
- **Storage**: Shared filesystem accessible from all compute nodes
- **CUDA**: 12.1 or later

### Cluster Requirements

- SLURM workload manager
- NCCL 2.18+ for multi-GPU communication
- Shared cache directories across nodes

## Installation

### 1. Clone Repository

```bash
cd /data/horse/ws/YOUR_USERNAME
git clone <repository_url> host_vllm
cd host_vllm
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install vllm>=0.15.0
pip install transformers>=4.51.0
pip install torch>=2.10.0
pip install pyyaml openai
```

### 3. Configure Environment

Edit `config.yaml` to set your paths and SLURM settings:

```yaml
slurm:
  account: "p_scads_pathology"  # Your SLURM account
  partition: "capella"          # Your partition name
  time_default: "12:00:00"      # Default job time limit

cache:
  xdg_cache_home: "/path/to/your/cache"  # Or leave empty for default
  triton_cache_dir: ""                    # Leave empty for auto

paths:
  venv_dir: "/data/horse/ws/YOUR_USERNAME/host_vllm/.venv"
```

## Configuration

### config.yaml Structure

#### SLURM Configuration

```yaml
slurm:
  partition: "capella"              # SLURM partition
  account: "p_scads_pathology"      # SLURM account (required)
  time_default: "12:00:00"          # Default time limit (HH:MM:SS)
  cpus_per_task: 16                 # CPUs per GPU task
  mem_per_gpu: "64G"                # Memory per GPU
```

#### Cache Directories

```yaml
cache:
  xdg_cache_home: ""        # Leave empty for $HOME/.cache
  triton_cache_dir: ""      # Leave empty for auto ($XDG_CACHE_HOME/triton)
  huggingface_cache: ""     # Optional: HF model cache directory
```

**Important**: Leave cache paths empty to use system defaults. This prevents exposing personal paths in the configuration.

#### Virtual Environment

```yaml
paths:
  venv_dir: ""  # Leave empty to auto-detect from current environment
                # Or set to: /path/to/your/.venv
```

#### vLLM Settings

```yaml
vllm:
  gpu_memory_utilization: 0.90      # Fraction of GPU memory (0.0-1.0)
  max_model_len_default: 32768      # Default context length
  max_num_seqs: 16                  # Parallel sequences
  dtype: "auto"                     # Data type: auto, bfloat16, float16
```

#### Multi-Node Distributed Settings

```yaml
distributed:
  rdzv_backend: "c10d"         # PyTorch rendezvous backend
  rdzv_timeout: 1200           # Timeout in seconds
  master_port_base: 29500      # Base port for communication

nccl:
  socket_ifname: ""            # Network interface (e.g., ib0, eth0)
  blocksize: 1048576           # NCCL block size
  ib_timeout: 22               # InfiniBand timeout
  debug: "WARN"                # Debug level: TRACE, INFO, WARN, ERROR
```

#### Per-Model Configuration

```yaml
models:
  llama-4-maverick:
    name: "Llama-4-Maverick-17B-128E-Instruct"
    huggingface_id: "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    job_script: "jobs/llama4_maverick_2nodes_8gpus.sh"
    port: 8006
    gpus: 8             # Total GPUs across all nodes
    nodes: 2            # Number of SLURM nodes
    quantization: "fp8"
    vision: true
    time: "24:00:00"    # Override default time limit
    # partition: "long" # Optional: override partition
```

## Usage

### Basic Usage

1. **Select Model**: Edit `config.yaml` and set `active_model`:

```yaml
active_model: llama-4-scout  # Choose from available models
```

2. **Submit Job**:

```bash
./run.sh
```

3. **Monitor Job**:

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/llama4_scout_*.out
```

### Custom Configuration

Use a custom config file:

```bash
./run.sh -c my_config.yaml
```

### Querying the Model

Once the job is running, query via OpenAI-compatible API:

```bash
# Set environment variables
export VLLM_HOST=<node_name>  # From job output
export VLLM_PORT=8006

# Query with Python script
python scripts/query_openai_compatible.py \
  --host $VLLM_HOST \
  --port $VLLM_PORT \
  --prompt "Explain quantum computing"

# Or use main.py
python main.py \
  --model llama-4-scout \
  --host $VLLM_HOST \
  --prompt "Your prompt here"
```

## Multi-Node Deployment

### Architecture

vLLM multi-node deployment uses:
- **Tensor Parallelism (TP)**: Model weights distributed across GPUs
- **torchrun**: PyTorch distributed launcher
- **c10d backend**: Reliable rendezvous for node coordination
- **NCCL**: High-performance GPU communication

### Multi-Node Configuration

For models requiring multiple nodes (e.g., Llama-4-Maverick, GLM-4.7):

```yaml
models:
  model-name:
    nodes: 2              # Number of SLURM nodes
    gpus: 8               # Total GPUs (nodes × gpus-per-node)
    time: "24:00:00"      # Longer time for large models
```

### Network Configuration

#### InfiniBand (Recommended)

If you have InfiniBand:

```yaml
nccl:
  socket_ifname: "ib0"   # InfiniBand interface
  ib_timeout: 22         # IB-specific timeout
```

#### Ethernet

For Ethernet-based clusters:

```yaml
nccl:
  socket_ifname: "eth0"  # Ethernet interface
  # Or leave empty for auto-detection
```

### Verifying Multi-Node Setup

After job submission, check logs for:

```
Master address: node001
Master port: 29542
Starting <Model> on 2 nodes (8 GPUs)...
Tensor Parallel Size: 8
```

Successful multi-node communication shows:
```
INFO: Started server process [PID]
INFO: Waiting for application startup.
INFO: Application startup complete.
```

## Available Models

### Single-Node Models

| Model | GPUs | VRAM | Context | Description |
|-------|------|------|---------|-------------|
| **GPT-OSS-20B** | 1 | 24GB | 128K | Fast reasoning with tool use |
| **GPT-OSS-120B** | 1 | 80GB | 128K | High-reasoning production model |
| **GLM-4.7-Flash** | 2 | 160GB | 128K | MoE with fast inference |
| **GLM-4.6V (FP8)** | 2 | 160GB | 128K | Vision-language MoE |
| **GLM-4.6V (FP16)** | 4 | 320GB | 128K | Full precision VLM |
| **Llama-4-Scout** | 2 | 160GB | 10M | Multimodal with 16 experts |

### Multi-Node Models

| Model | Nodes | GPUs | VRAM | Context | Description |
|-------|-------|------|------|---------|-------------|
| **GLM-4.7** | 2 | 8 | 640GB | 131K | 358B MoE language model |
| **Qwen3-VL-235B** | 2 | 8 | 640GB | 256K | Vision-language agent |
| **Llama-4-Maverick** | 2 | 8 | 640GB | 1M | 128 experts, multimodal |

### Quantization Types

- **FP8**: 8-bit floating point, ~2x memory reduction
- **MXFP4**: 4-bit MX format for GPT-OSS models
- **BF16**: 16-bit bfloat, full precision baseline

## Troubleshooting

### Common Issues

#### 1. Job Fails to Start

**Symptom**: Job remains in pending (PD) state

**Solution**:
```bash
# Check job details
scontrol show job <JOB_ID>

# Verify account access
sacctmgr show user $USER

# Check partition availability
sinfo -p capella
```

#### 2. Multi-Node Communication Fails

**Symptom**: Timeout errors, "No available node types" message

**Solution**:
```yaml
# In config.yaml, try different network interface
nccl:
  socket_ifname: "ib0"  # Or "eth0", or leave empty

# Increase timeout
distributed:
  rdzv_timeout: 1800  # 30 minutes
```

#### 3. Out of Memory (OOM)

**Symptom**: CUDA out of memory errors

**Solution**:
```yaml
# Reduce GPU memory utilization
vllm:
  gpu_memory_utilization: 0.85  # From 0.90

# Reduce batch size
vllm:
  max_num_seqs: 8  # From 16

# Reduce context length
vllm:
  max_model_len_default: 16384  # From 32768
```

#### 4. Port Already in Use

**Symptom**: "Address already in use" error

**Solution**:
```yaml
# Change port in config.yaml
models:
  model-name:
    port: 8007  # Different port
```

#### 5. Cache Permission Errors

**Symptom**: Permission denied writing to cache

**Solution**:
```yaml
# Set explicit cache paths
cache:
  xdg_cache_home: "/data/horse/ws/$USER/cache"
  triton_cache_dir: "/data/horse/ws/$USER/cache/triton"
```

### Debugging

#### Enable Verbose Logging

```yaml
nccl:
  debug: "INFO"  # Or "TRACE" for maximum verbosity
```

#### Check NCCL Communication

```bash
# In job output, look for:
# "NCCL INFO Ring 00 : ..."
# Indicates successful GPU ring formation

# Check network connectivity
srun --nodes=2 --ntasks=2 hostname
```

#### Verify GPU Allocation

```bash
# Inside running job
nvidia-smi

# Should show allocated GPUs
# Check that all expected GPUs are visible
```

### Ray Observability (Multi-Node)

The system uses SLURM/torchrun, not Ray, but for Ray-based deployments:

```bash
# Ray dashboard (if using Ray)
ray dashboard
```

## Best Practices

### 1. Resource Allocation

- **Time Limits**: Add 20% buffer for model loading and initialization
- **Memory**: Start with `gpu_memory_utilization: 0.90`, reduce if OOM
- **CPUs**: Use 16 CPUs per GPU for optimal data preprocessing

### 2. Multi-Node Optimization

```yaml
# Optimal NCCL settings for Ethernet
nccl:
  socket_ifname: ""          # Auto-detect
  blocksize: 1048576         # 1MB blocks
  debug: "WARN"              # Minimal overhead

# Increase timeouts for large models
distributed:
  rdzv_timeout: 1800         # 30 min for slow networks
```

### 3. Model Selection

- **Development**: Start with smaller models (GPT-OSS-20B, Llama-4-Scout)
- **Production**: Use FP8 quantization for 2x throughput
- **Vision tasks**: GLM-4.6V or Qwen3-VL for multimodal

### 4. Cache Management

```bash
# Clean old model caches periodically
rm -rf $XDG_CACHE_HOME/huggingface/hub/*

# Set explicit HF cache to control disk usage
export HF_HOME=/scratch/$USER/hf_cache
```

### 5. Monitoring

```bash
# Create monitoring script
watch -n 5 'squeue -u $USER && nvidia-smi'

# Check logs in real-time
tail -f logs/*.out
```

### 6. Reproducibility

```yaml
# Fix random seeds in model config
vllm:
  seed: 42  # Reproducible generation
```

## Version History

- **v2.0** (2026-02-03): Complete refactoring
  - vLLM v0.15.0 support
  - Removed hardcoded paths
  - Enhanced multi-node configuration
  - Comprehensive NCCL settings
  - Centralized configuration management

- **v1.0**: Initial release
  - Basic SLURM integration
  - Single-node deployments

## Support

For issues:
1. Check job logs in `logs/` directory
2. Verify SLURM account access
3. Test with single-node models first
4. Consult [vLLM Documentation](https://docs.vllm.ai/en/latest/)

## References

- [vLLM Multi-Node Serving Guide](https://docs.vllm.ai/en/latest/examples/online_serving/multi-node-serving/)
- [vLLM v0.15.0 Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.15.0)
- [Distributed Troubleshooting](https://docs.vllm.ai/en/latest/serving/distributed_troubleshooting/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
