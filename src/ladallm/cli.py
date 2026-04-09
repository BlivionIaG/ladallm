import argparse

import numpy as np

from ladallm.safetensors import Safetensors


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm or Root Mean Square Normalization
    Input:
        x: [seq_len, hidden_size]
        weight: [hidden_size]
        eps: float - epsilon for numerical stability
    Returns:
        np.ndarray - normalized tensor [seq_len, hidden_size]
    """
    return (x / np.sqrt(np.mean(x**2, keepdims=True) + eps)) * weight


def main():
    parser = argparse.ArgumentParser(description="LadaLLM inference")
    parser.add_argument("prompt", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument(
        "--model",
        help="Path to the model file",
        type=str,
        default="./models/HuggingFaceTB/SmolLM2-135M/model.safetensors",
    )
    args = parser.parse_args()

    print(f"Generating: {args.prompt}")
    print(f"Loading model: {args.model}")
    tensors = Safetensors(args.model)
    print(f"Loaded {len(tensors.tensor_data)} tensors")
    print(
        f"Model embed tokens weight shape: {tensors.tensor_data['model.embed_tokens.weight'].shape}"
    )

    # ,,, load model and generate


if __name__ == "__main__":
    main()
