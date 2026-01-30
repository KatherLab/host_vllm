# VLLM HPC Job Scripts

This directory contains production-ready **SLURM job scripts** for hosting large language and vision-language models using **vLLM** on HPC clusters.

## � Quick Start

1. **Configure**: Edit `config.yaml` and set `active_model` to your desired model
2. **Run**: Execute `./run.sh` to submit the SLURM job
3. **Query**: Use `main.py` to interact with the model

```bash
# Set active model (e.g., glm-4.6v)
# Edit config.yaml: active_model: glm-4.6v

# Submit job
./run.sh

# Query the model
python main.py --model glm-4.6v --prompt "What's in this image?" --image-url "https://..."
```

## 📋 Model Profiles

### Language Models

| Profile | Model | HuggingFace ID | GPUs | Nodes | Quantization |
|---------|-------|----------------|------|-------|--------------|
| `gpt_oss_20b_single_h100.sh` | GPT-OSS 20B | `openai/gpt-oss-20b` | 1 (H100) | 1 | MXFP4 |
| `gpt_oss_120b_single_h100.sh` | GPT-OSS 120B | `openai/gpt-oss-120b` | 1 (H100) | 1 | MXFP4 |
| `glm4_7_flash_2gpus.sh` | GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` | 2 | 1 | FP8 |
| `glm4_7_2nodes_4gpus.sh` | GLM-4.7 | `zai-org/GLM-4.7` | 8 | 2 | FP8 |

### Vision-Language Models

| Profile | Model | HuggingFace ID | GPUs | Nodes | Quantization | Precision Options |
|---------|-------|----------------|------|-------|--------------|-------------------|
| `glm4_6v_2gpus_fp8.sh` | GLM-4.6V | `zai-org/GLM-4.6V-FP8` | 2 | 1 | FP8 | 8-bit (default) |
| `glm4_6v_4gpus_fp16.sh` | GLM-4.6V | `zai-org/GLM-4.6V` | 4 | 1 | BF16 | 16-bit |
| `qwen3_vl_235b_4nodes.sh` | Qwen3-VL-235B | `Qwen/Qwen3-VL-235B-A22B-Thinking` | 16 | 4 | FP8 | 8-bit only |

⚠️ **Precision Notes**: 
- 8-bit precision (FP8/MXFP4) is used by default for efficiency
- For 16-bit precision, double the GPUs are required (e.g., GLM-4.6V needs 4 GPUs instead of 2)

## 🤖 Model Details

### GPT-OSS (OpenAI)
- **20B Model**: ~24GB VRAM needed - runs on single H100 with MXFP4 quantization
- **120B Model**: ~80GB VRAM needed - runs on single H100 with MXFP4 quantization  
- Both models support reasoning (low/medium/high), function calling, and web browsing
- Requires special vLLM build: `vllm==0.10.1+gptoss`
- [Model Card](https://huggingface.co/openai/gpt-oss-120b)

### GLM-4.7 (Z.ai / Zhipu AI)
- **Flash**: 30B-A3B MoE model (30B active / 3B expert) - fits on 2 GPUs with FP8
- **Standard**: 358B parameters MoE model - requires 8 GPUs with FP8 for inference
- Both support tool calling, interleaved thinking, and agentic capabilities
- Requires vLLM nightly build for GLM-4.7 support
- [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) | [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)

### Qwen3-VL (Alibaba)
- **235B-A22B-Thinking**: 236B parameter MoE vision-language model (22B active parameters)
- Visual agent capabilities: PC/mobile GUI operation, visual coding, advanced spatial perception
- Native 256K context, expandable to 1M tokens
- Supports video understanding, OCR in 32 languages, 3D grounding
- Requires 16 GPUs (4 nodes × 4 GPUs) with FP8 quantization
- [Model Card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)

## 🚀 Usage

### Prerequisites

1. Load CUDA module:
```bash
module load CUDA
```

2. Install vLLM (model-specific instructions in scripts):

**For GPT-OSS:**
```bash
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

**For GLM-4.7:**
```bash
pip install -U vllm --pre \
  --index-url https://pypi.org/simple \
  --extra-index-url https://wheels.vllm.ai/nightly
```

**For Qwen-VL:**
```bash
pip install -U vllm transformers
```

### Submit a Job

```bash
cd /data/horse/ws/s1787956-host_vllm/code/host_vllm/jobs

# Single-node examples
sbatch ./gpt_oss_20b_single_h100.sh
sbatch ./gpt_oss_120b_single_h100.sh
sbatch ./glm4_7_flash_2gpus.sh

# Multi-node examples  
sbatch ./glm4_7_2nodes_4gpus.sh
sbatch ./qwen3_vl_235b_4nodes.sh
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/gpt_oss_120b_*.out
tail -f logs/glm4_7_flash_2gpus_*.out
```

### Test the Endpoint

Once the server is ready:

```bash
curl http://<hostname>:8000/v1/models
curl http://<hostname>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## ⚙️ Configuration Notes

| Setting | Default | Purpose |
|---------|---------|---------|
| `--tensor-parallel-size` | 1-16 | Number of GPUs for tensor parallelism |
| `--quantization fp8` | various | 8-bit inference (requires GPU support) |
| `--gpu-memory-utilization` | 0.90 | Fraction of GPU memory to use |
| `--max-model-len` | 32k-131k | Maximum context length |
| `--trust-remote-code` | enabled | Required for some models (GLM, Qwen) |

### Cache Directories

All scripts use centrally configured cache directories:
```bash
export XDG_CACHE_HOME=/data/horse/ws/s1787956-Cache
export TRITON_CACHE_DIR=/data/horse/ws/s1787956-Cache/triton
```

## 🔧 Customization Guide

| To Change | Edit This |
|-----------|-----------|
| Partition | Change `--partition=capella` in SBATCH headers |
| Memory | Adjust `--mem` (e.g., `--mem=512G`) |
| Time limit | Change `--time` (e.g., `--time=24:00:00`) |
| Port | Edit `VLLM_PORT` variable |
| Model variant | Change `VLLM_MODEL` to HuggingFace model ID |

## 📚 References

- [vLLM Documentation](https://docs.vllm.ai/)
- [GPT-OSS Cookbook](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
- [GLM-4.7 GitHub](https://github.com/zai-org/GLM-4.5)
- [Qwen2.5-VL Documentation](https://qwen.readthedocs.io/)|