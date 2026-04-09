# Backend Abstraction Design

> This document defines the dual-layer architecture (Python orchestration + C++ compute) that underlies v1+ of the inference engine. It is the contract between the Python scheduler/model-runner and the C++ kernels.

## Overview

The engine follows a **dual-layer architecture** inspired by vLLM:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON LAYER (Orchestration)                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Scheduler  │  │ Batch Builder│  │  KV Manager  │            │
│  │   (F105)    │  │   (F106)     │  │  (F101-F104) │            │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘            │
│         │                │                  │                    │
│         └────────────────┴──────────────────┘                    │
│                          │                                       │
│                   ┌──────┴──────┐                                │
│                   │ Model Runner│                                │
│                   │   (F104)    │                                │
│                   └──────┬──────┘                                │
└──────────────────────────┼──────────────────────────────────────┘
                           │ FFI / Binding Layer
                           │ (ctypes/cffi)
┌──────────────────────────┼──────────────────────────────────────┐
│                    C++ LAYER (Compute Kernels)                   │
│  ┌───────────────────────┴───────────────────────┐              │
│  │           Backend Interface (F109)            │              │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────────┐     │              │
│  │  │ OpenCL  │ │  ROCm   │ │    CPU      │     │              │
│  │  │ (v1-v3) │ │ (v4-v5) │ │ (NumPy/v0)  │     │              │
│  │  └────┬────┘ └────┬────┘ └──────┬──────┘     │              │
│  └───────┼──────────┼──────────┼───────────────┘              │
│          │          │          │                                │
│  ┌───────┴──┐ ┌─────┴───┐ ┌────┴────┐ ┌──────────┐            │
│  │   GEMM   │ │ Paged   │ │Element- │ │  Sampler │            │
│  │ (F111)   │ │Attention│ │ wise Ops│ │ (F113)   │            │
│  │          │ │ (F104)  │ │(F112)   │ │          │            │
│  └──────────┘ └─────────┘ └─────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle:** The Python layer is concerned with *what to run*; the C++ layer is concerned with *how to run it fast*.

---

## Python Layer Responsibilities

The Python layer handles all systems logic, state management, and orchestration:

### 1. Scheduler (`scheduler.py`)
- Request lifecycle management (WAITING → PREFILL → DECODE → DONE)
- Admission control and priority decisions
- Token budget allocation per step
- Orchestrates when to run prefill vs decode vs mixed batches

### 2. Batch Builder (`batch_builder.py`)
- Continuous batching: packing tokens from multiple requests
- Chunked prefill: splitting long prompts into budget-sized pieces
- Mixing prefill and decode in the same batch
- Managing per-token metadata (positions, block tables, sequence lengths)

### 3. KV Cache Manager (`kv_cache.py`)
- Block pool allocation and reference counting (F101)
- Per-request block tables (F102)
- Prefix caching hash table (F104a)
- LRU eviction for unreferenced blocks (F104c)

### 4. Model Runner (`model_runner.py`)
- Top-level orchestration of a single forward pass
- Coordinates Q/K/V projection → attention → MLP → output
- Handles both prefill (many tokens) and decode (1 token per request)
- Calls into C++ backend for actual computation

### 5. Tokenizer & Engine (`tokenizer.py`, `engine.py`)
- Text preprocessing and detokenization
- High-level `generate()` API
- Streaming output handling

---

## C++ Layer Responsibilities

The C++ layer contains all compute kernels that touch tensors at scale. These are compiled for the target backend (OpenCL, ROCm/HIP, CUDA, Vulkan) and called via FFI.

### Core Kernels (Every Backend Must Implement)

| Kernel | Input | Output | Description |
|--------|-------|--------|-------------|
| `gemm` | A[M,K], B[K,N] | C[M,N] | Matrix multiply for projections, MLP, LM head |
| `rmsnorm` | x[seq,d], g[d] | out[seq,d] | Root-mean-square normalization |
| `rope` | Q[seq,heads,d], K[seq,heads,d], positions[seq] | Q', K' | Rotary position embedding |
| `paged_attention` | Q[seq_q,heads,d], KV_pool[blocks,...], block_tables[reqs,...] | out[seq_q,heads,d] | Paged attention with block-table indexing |
| `silu_mul` | gate[seq,d], up[seq,d] | out[seq,d] | SwiGLU: `silu(gate) * up` |
| `add` | a[...], b[...] | out[...] | Residual addition |
| `softmax` | logits[seq,vocab] | probs[seq,vocab] | Softmax for sampling |
| `sample_argmax` | logits[seq,vocab] | token_ids[seq] | Greedy sampling |
| `sample_top_k_p` | logits[seq,vocab], k, p | token_ids[seq] | Top-k and top-p sampling |

### Quantized Kernels (v2+)

| Kernel | Input | Output | Description |
|--------|-------|--------|-------------|
| `gemm_w8a16` | A_fp16, B_int8, scales_fp16 | C_fp16 | Weight-only 8-bit matmul |
| `gemm_w4a16` | A_fp16, B_int4, scales_fp16 | C_fp16 | Weight-only 4-bit matmul (grouped) |
| `quantize_kv_int8` | K_fp16, V_fp16 | K_int8, V_int8, scales | KV cache quantization on write |
| `paged_attention_int8_kv` | Q, KV_pool_int8, scales, block_tables | out | Attention with int8 KV cache |
| `paged_attention_fp8_kv` | Q, KV_pool_fp8, block_tables | out | Attention with fp8 KV cache (if supported) |

### Speculative Decoding Kernels (v3+)

| Kernel | Input | Output | Description |
|--------|-------|--------|-------------|
| `paged_attention_multi_verify` | Q[seq_q,heads,d], KV_pool, block_tables, verify_len | out, logits | Multi-token verification (F304) |
| `sample_speculative_accept` | draft_tokens[k], target_logits[k+1,vocab], draft_probs[k,vocab] | accepted_count, emitted_tokens | Modified rejection sampling (F306) |

---

## FFI / Binding Layer

Python calls into C++ through a thin wrapper. The interface is designed to be:
- **Minimal**: Only tensor ops and buffer management cross the boundary
- **Explicit**: All device handles, queues, and buffers are explicit
- **Async-friendly**: Operations enqueue work and return; synchronization is explicit

### Python Side (Conceptual)

```python
class Backend(ABC):
    """Abstract base for all compute backends."""
    
    # Device management
    @abstractmethod
    def device(self, index: int) -> Device: ...
    
    # Buffer lifecycle
    @abstractmethod
    def alloc(self, device: Device, nbytes: int) -> Buffer: ...
    @abstractmethod
    def copy_to_device(self, host_arr: np.ndarray, buf: Buffer): ...
    @abstractmethod
    def copy_from_device(self, buf: Buffer) -> np.ndarray: ...
    
    # Command queue (for async execution)
    @abstractmethod
    def queue(self, device: Device) -> Queue: ...
    @abstractmethod
    def queue_finish(self, queue: Queue): ...
    
    # Core ops - all async, all take explicit queue
    @abstractmethod
    def gemm(self, queue: Queue, a: Buffer, b: Buffer, out: Buffer, 
             M: int, N: int, K: int): ...
    
    @abstractmethod
    def rmsnorm(self, queue: Queue, x: Buffer, gamma: Buffer, 
                out: Buffer, eps: float, d_model: int): ...
    
    @abstractmethod
    def rope(self, queue: Queue, q: Buffer, k: Buffer, 
             cos_sin: Buffer, positions: Buffer, ...): ...
    
    @abstractmethod
    def paged_attention(self, queue: Queue, q: Buffer,
                       k_pool: Buffer, v_pool: Buffer,
                       block_tables: Buffer, ...): ...
    
    # ... other ops
```

### C++ Side (Conceptual)

```cpp
// C-compatible exports for FFI
extern "C" {
    // Backend lifecycle
    BackendHandle* backend_create(const char* backend_type);
    void backend_destroy(BackendHandle* backend);
    
    // Device management
    DeviceHandle* device_get(BackendHandle* backend, int index);
    
    // Buffer lifecycle
    BufferHandle* buffer_alloc(DeviceHandle* device, size_t nbytes);
    void buffer_free(BufferHandle* buf);
    void copy_host_to_device(void* host_ptr, BufferHandle* buf, size_t nbytes);
    void copy_device_to_host(BufferHandle* buf, void* host_ptr, size_t nbytes);
    
    // Queue management
    QueueHandle* queue_create(DeviceHandle* device);
    void queue_finish(QueueHandle* queue);
    
    // Ops - enqueue work, return immediately
    void op_gemm(QueueHandle* queue, BufferHandle* a, BufferHandle* b, 
                 BufferHandle* out, int M, int N, int K);
    
    void op_rmsnorm(QueueHandle* queue, BufferHandle* x, BufferHandle* gamma,
                    BufferHandle* out, float eps, int d_model);
    
    void op_rope(QueueHandle* queue, BufferHandle* q, BufferHandle* k,
                 BufferHandle* cos_sin, BufferHandle* positions, ...);
    
    void op_paged_attention(QueueHandle* queue, BufferHandle* q,
                           BufferHandle* k_pool, BufferHandle* v_pool,
                           BufferHandle* block_tables, ...);
    
    // ... other ops
}
```

### Why This Design

1. **Separation of concerns**: Python handles hard systems logic where debugging and iteration matter; C++ handles number-crunching where every memory access pattern matters
2. **Backend portability**: Adding ROCm (v4), CUDA (v6), or Vulkan (v7) means implementing the same C++ interface — zero Python changes
3. **Testability**: The NumPy backend (v0) implements the same interface, giving us a correctness oracle for all GPU backends
4. **Multi-GPU ready**: Explicit device handles and queues mean v5's tensor/pipeline parallelism is mostly implementing collectives in C++, not redesigning the boundary

---

## Backend Implementations

### NumPy Backend (v0, fallback)
- Pure Python + NumPy
- No FFI layer needed
- Correctness reference for all other backends
- Runs on CPU when no GPU is available

### OpenCL Backend (v1-v3)
- Kernels written in OpenCL C
- Runtime kernel compilation from source strings
- Explicit context, command queues, device handles
- Reference for multi-GPU design (explicit everything)

### ROCm/HIP Backend (v4-v5)
- Kernels written in HIP C++ (near-CUDA)
- Compiled offline with `hipcc`
- Links against rocBLAS for optimized GEMM
- Validates abstraction on AMD hardware

### CUDA Backend (v6)
- Kernels ported from HIP (mechanical `hipify` pass)
- Links against cuBLAS
- Validates abstraction on NVIDIA hardware

### Vulkan Backend (v7, optional)
- Kernels written as compute shaders (GLSL/HLSL)
- For platforms where Vulkan is the only option

---

## Multi-GPU Design (v5 Preview)

The backend interface is designed for multi-GPU from v1, even though v1-v4 only use one device. Key design choices:

1. **Explicit device handles**: Every buffer knows its device; every op takes a device or queue
2. **Sharded weights**: Weight tensors are always stored as a list of shards (length 1 in v1-v4)
3. **Collective stubs**: `all_gather`, `reduce_scatter`, `all_reduce` exist as identity ops in v1-v4
4. **Parallelism plan**: Config carries `{tp, pp}` even when both are 1

In v5, tensor parallelism (TP) and pipeline parallelism (PP) become:
- **TP**: Sharded weights with real collectives in attention/MLP
- **PP**: Staged execution across devices with activation transfer

The Python layer (scheduler, batch builder) remains unchanged.

---

## Conformance Testing

Every backend must pass:

1. **Unit tests**: Each kernel tested in isolation against NumPy reference
2. **Integration tests**: Full forward pass produces identical outputs to NumPy
3. **Greedy conformance**: `backend_output == numpy_output` byte-for-byte
4. **Sampling distribution test**: Statistical equivalence across many seeds

Test matrix:
- NumPy (reference)
- OpenCL (v1-v3)
- OpenCL + quantization (v2-v3)
- OpenCL + speculative decoding (v3)
- ROCm (v4)
- ROCm + quantization + spec-decode (v4)
- CUDA (v6)

---

## Build System

```
backend/
├── include/
│   └── backend_interface.h       # C++ interface definition
├── numpy/                        # NumPy reference (header-only C++ that calls NumPy)
├── opencl/
│   ├── kernels/
│   │   ├── gemm.cl
│   │   ├── attention.cl
│   │   └── ...
│   ├── opencl_backend.cpp
│   └── opencl_backend.h
├── rocm/
│   ├── kernels/
│   │   ├── gemm.hip.cpp
│   │   ├── attention.hip.cpp
│   │   └── ...
│   ├── rocm_backend.cpp
│   └── rocm_backend.h
├── cuda/                         # (v6)
├── vulkan/                       # (v7, optional)
└── python_binding.cpp            # pybind11 or cffi wrapper
```

Build targets:
- `libbackend_numpy.so` - Always built, no dependencies
- `libbackend_opencl.so` - Requires OpenCL headers/runtime
- `libbackend_rocm.so` - Requires ROCm toolchain
- `libbackend_cuda.so` - (v6) Requires CUDA toolkit

---

## Further Reading

- [architecture_v0.md](architecture_v0.md) - NumPy-only baseline (no C++ layer)
- [architecture_v1.md](architecture_v1.md) - First C++ backend (OpenCL)
- [architecture_v2.md](architecture_v2.md) - Quantized kernels
- [architecture_v3.md](architecture_v3.md) - Speculative decoding kernels
- [architecture_v4.md](architecture_v4.md) - ROCm backend (second C++ implementation)
