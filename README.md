# LadaLLM

Learning AI inference from scratch by building an LLM inference engine.

This project recreates vLLM and modern inference engine concepts for educational purposes. We build everything from the ground up to deeply understand how inference engines work internally.

## Architecture

The engine follows a **dual-layer architecture**:
- **Python Layer** (v0): Orchestration, scheduler, batching, cache management
- **C/C++ Layer** (v1+): Compute kernels (attention, GEMM, quantization) via OpenCL/ROCm/CUDA

## Backend Priority

1. **CPU** (v0: NumPy reference) — correctness baseline ✓
2. **OpenCL** (v1: first GPU backend)
3. **ROCm/HIP** (v4: AMD native)
4. **CUDA** (v6: NVIDIA native)
5. **Vulkan** (v7/optional)

## Current Status: v0

v0 establishes a correctness baseline using pure Python + NumPy on CPU. All later versions must match v0's outputs.

### Implemented Features

| Feature | Status | Description |
|---------|--------|-------------|
| F1 — Weight Loading | ✅ | Safetensors loader with memory mapping |
| F2 — RMSNorm | ✅ | Root Mean Square Layer Normalization |
| F3 — RoPE | ☐ | Rotary Position Embedding |
| F4 — Attention | ☐ | Multi-head attention with KV cache |
| F5 — SwiGLU MLP | ☐ | Gated feed-forward network |
| F6 — Decoder Block | ☐ | Full transformer layer stack |
| F7 — KV Cache | ☐ | Naive contiguous cache |
| F8 — Prefill/Decode | ☐ | Two-phase generation |
| F9 — Sampler | ☐ | Greedy sampling |
| F10 — Scheduler | ☐ | FIFO request scheduling |
| F11 — Engine Loop | ☐ | Top-level orchestration |
| F12 — Stop Conditions | ☐ | EOS/max_tokens handling |

See [docs/v0/README.md](docs/v0/README.md) for detailed feature docs.

## Quick Start

### Installation

```bash
# Clone and install in development mode
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=ladallm --cov-report=term-missing

# Run specific feature tests
pytest tests/test_safetensors.py -v   # F1: Weight loading
pytest tests/test_rmsnorm.py -v       # F2: RMSNorm
```

### Load a Model

```python
from ladallm.safetensors import Safetensors

# Load SmolLM2-135M weights
st = Safetensors("models/SmolLM2-135M/model.safetensors")
print(f"Loaded {len(st.tensor_data)} tensors")

# Access specific weights
embed = st.tensor_data["model.embed_tokens.weight"]
print(f"Embedding shape: {embed.shape}")  # [49152, 576]

st.close()
```

### Apply RMSNorm

```python
from ladallm.cli import rms_norm
import numpy as np

# Example: normalize activations
x = np.random.randn(10, 576).astype(np.float32)
weight = np.ones(576, dtype=np.float32)

normalized = rms_norm(x, weight)
```

## Project Structure

```
vllm-from-scratch/
├── docs/
│   ├── architecture_v0.md      # v0 architecture & concepts
│   ├── architecture_v1.md      # v1 architecture (planned)
│   └── v0/
│       ├── README.md           # Feature index
│       ├── f01-weight-loading.md
│       ├── f02-rmsnorm.md
│       └── ...
├── src/ladallm/
│   ├── __init__.py
│   ├── safetensors.py          # F1: Weight loading
│   └── cli.py                  # CLI + F2: RMSNorm
├── tests/
│   ├── test_safetensors.py     # F1 tests
│   ├── test_rmsnorm.py         # F2 tests
│   └── conftest.py             # Test fixtures
├── pyproject.toml
└── README.md
```

## Target Model: SmolLM2-135M

v0 is tested against **HuggingFaceTB/SmolLM2-135M**:
- 135M parameters, ~270MB in FP16
- Llama-style architecture (RMSNorm, RoPE, GQA, SwiGLU)
- Leaves ~730MB for KV cache on 1GB VRAM

Download:
```bash
mkdir -p models/HuggingFaceTB/SmolLM2-135M
hf download HuggingFaceTB/SmolLM2-135M --local-dir ./models/HuggingFaceTB/SmolLM2-135M
```

## Version Roadmap

| Version | Theme | Target Hardware |
|---------|-------|----------------|
| v0 | Naive baseline (Python + NumPy) | CPU |
| v1 | Paged attention + continuous batching | OpenCL |
| v2 | Weight + KV-cache quantization | OpenCL |
| v3 | Speculative decoding | OpenCL |
| v4 | ROCm backend | AMD GPU |
| v5 | Multi-GPU (TP + PP) | OpenCL + ROCm |
| v6 | CUDA backend | NVIDIA GPU |

See [docs/architecture_v0.md](docs/architecture_v0.md) for detailed v0 plan.

## Documentation

Each feature has its own deep-dive doc in `docs/v0/` covering:
- What the feature is and why it exists
- Math and shapes
- Step-by-step implementation guide
- Edge cases and testing strategy

## Performance Goals

- **Throughput**: tokens/second per GPU
- **Concurrency**: simultaneous requests
- **TTFT**: Time To First Token (latency)

## Contributing

This is a learning project. The goal is understanding, not production code.

- Read the architecture docs to understand concepts
- Implement features following the step-by-step guides
- Write tests to verify correctness
- Match v0 outputs for all later versions

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Further Reading

- [vLLM Paper](https://arxiv.org/abs/2309.06180) — PagedAttention
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original transformer
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE paper
- [RMSNorm Paper](https://arxiv.org/abs/1910.07467) — Root Mean Square Layer Normalization
