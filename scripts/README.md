# Query Scripts

This directory contains modular query scripts that can be configured per-model in `config.yaml`.

## Usage

Each model in `config.yaml` can specify a `query_script` field:

```yaml
models:
  my-model:
    job_script: "jobs/my_model.sh"
    query_script: "scripts/my_custom_query.py"  # Custom query script for this model
```

## Default Script

- **`query_openai_compatible.py`** - Default OpenAI-compatible API client for all models

## Creating Custom Query Scripts

To create a custom query script for a specific model:

1. Create a new Python script in this directory
2. Make it executable: `chmod +x scripts/my_script.py`
3. Reference it in `config.yaml` under the model's `query_script` field

Example custom script structure:

```python
#!/usr/bin/env python3
"""Custom query script for my special model."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    
    # Your custom query logic here
    print(f"Querying {args.model} at {args.host}:{args.port}")

if __name__ == "__main__":
    main()
```

The `run.sh` script will automatically pass `--model`, `--host`, `--port`, and `--prompt` arguments when displaying the query command.
