#!/usr/bin/env python3
"""
Example script to query a vLLM-hosted model via OpenAI-compatible API.

Usage:
    python main.py --model gpt-oss-20b --prompt "Explain quantum computing"
    python main.py --model glm-4.7 --prompt "Write a Python function" --max-tokens 512
    python main.py --model qwen3-vl-235b --prompt "Describe this image" --image-url "https://..."
"""

import argparse
import os
import sys
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: uv add openai")
    sys.exit(1)


# Model configurations: name -> (port, default_max_tokens, supports_vision)
MODEL_CONFIGS = {
    "gpt-oss-20b": (8000, 4096, False),
    "gpt-oss-120b": (8000, 4096, False),
    "glm-4.7": (8002, 8192, False),
    "glm-4.7-flash": (8001, 8192, False),
    "glm-4.6v": (8004, 8192, True),
    "glm-4.6v-fp16": (8004, 8192, True),
    "qwen3-vl-235b": (8003, 4096, True),
}


def create_client(host: str = "localhost", port: int = 8000) -> OpenAI:
    """Create OpenAI client pointing to vLLM server."""
    base_url = f"http://{host}:{port}/v1"
    return OpenAI(base_url=base_url, api_key="dummy")


def query_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image_url: Optional[str] = None,
) -> str:
    """Send a query to the vLLM model server."""
    
    # Build message content
    if image_url and MODEL_CONFIGS.get(model_name, (0, 0, False))[2]:
        # Vision model with image
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
    else:
        # Text-only
        content = prompt
    
    messages = [{"role": "user", "content": content}]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def list_available_models():
    """Print available models and their configurations."""
    print("Available models:")
    print("-" * 60)
    for name, (port, max_tokens, vision) in MODEL_CONFIGS.items():
        vision_str = "✓ vision" if vision else "  text-only"
        print(f"  {name:<20} port={port:<6} max_tokens={max_tokens:<6} {vision_str}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Query a vLLM-hosted model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model gpt-oss-20b --prompt "Hello!"
  python main.py --model glm-4.6v --prompt "What's in this image?" --image-url "https://..."
  python main.py --model qwen3-vl-235b --prompt "Describe the scene" --image /path/to/image.jpg
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to query",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default="Are you even working?. Give me your training horizon! Do you now the current date?",
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("VLLM_HOST", "localhost"),
        help="vLLM server hostname (default: localhost or $VLLM_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override vLLM server port (default: model-specific)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: model-specific)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        default=None,
        help="URL of image for vision models",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Local path to image for vision models (converted to file:// URL)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # Get model configuration
    port, default_max_tokens, supports_vision = MODEL_CONFIGS[args.model]
    
    # Apply overrides
    if args.port:
        port = args.port
    max_tokens = args.max_tokens or default_max_tokens
    
    # Handle local image path
    image_url = args.image_url
    if args.image:
        abs_path = os.path.abspath(args.image)
        image_url = f"file://{abs_path}"
    
    # Validate vision model usage
    if image_url and not supports_vision:
        print(f"Error: Model '{args.model}' does not support vision/image inputs")
        sys.exit(1)
    
    # Create client and query
    print(f"Connecting to {args.host}:{port}...")
    client = create_client(args.host, port)
    
    print(f"Querying model: {args.model}")
    print(f"Prompt: {args.prompt}")
    if image_url:
        print(f"Image: {image_url}")
    print("-" * 60)
    
    response = query_model(
        client=client,
        model_name=args.model,
        prompt=args.prompt,
        max_tokens=max_tokens,
        temperature=args.temperature,
        image_url=image_url,
    )
    
    print(response)


if __name__ == "__main__":
    main()
