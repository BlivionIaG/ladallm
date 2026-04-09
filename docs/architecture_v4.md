# Architecture v4 — ROCm Backend

> **Prerequisites:** [`architecture_v0.md`](architecture_v0.md), [`architecture_v1.md`](architecture_v1.md), [`architecture_v2.md`](architecture_v2.md), and [`architecture_v3.md`](architecture_v3.md). v4 is a port, not a rewrite — it assumes v1's backend abstraction, paged attention, continuous batching, prefix caching, v2's quantization, and v3's speculative decoding are all working on OpenCL.

v4 has one headline goal: **add a second C++ compute backend (ROCm / HIP)**, and in doing so, *validate that v1's C++ backend abstraction (as extended by v2 and v3) was designed correctly*. If porting to a second backend requires changes outside the C++ compute layer or the thin FFI boundary, the design was wrong and we fix it here.

This doc is shorter than v0–v3 because most of the conceptual ground (paged attention, batching, GPU programming, multi-device design, quantization, speculative decoding) was already covered. v4 is mainly about learning **ROCm specifically** and **what a second backend teaches you about abstractions**.

---

## 1. What we are building

A second compute backend: **ROCm** (AMD's GPU stack), accessed via **HIP** (the C++ kernel language and runtime AMD provides). After v4:

- The engine runs unchanged on OpenCL *or* ROCm, selected by config.
- The same model outputs the same tokens on both (byte-exact under greedy sampling, per the conformance test v1 established and v2/v3 extended).
- Throughput and TTFT on AMD hardware are competitive with the OpenCL path on the same hardware — often noticeably better, since HIP sits closer to the metal on AMD GPUs than OpenCL does.
- The backend abstraction has been *battle-tested* by being implemented twice. Anything that was secretly OpenCL-specific in v1/v2/v3 gets refactored out here.

### Why ROCm is the right second backend

- **AMD hardware first.** The project's backend priority order is CPU → OpenCL → ROCm → CUDA → Vulkan. AMD consumer and data-center GPUs (RDNA, CDNA) are the primary non-NVIDIA GPU target; ROCm is their native stack.
- **HIP is near-source-compatible with CUDA.** HIP kernels look like CUDA kernels with `hip` prefixes instead of `cuda`. This means v4 is *also* a cheap rehearsal for v6's CUDA backend — most HIP kernels will port to CUDA with a `hipify` pass and minor edits. Doing ROCm before CUDA lets us learn the abstraction lessons on one platform and then cash them in on the next.
- **It exposes abstraction leaks.** OpenCL and HIP have different launch models, different synchronization primitives, different memory models, and different toolchains. Anything in v1–v3 that secretly assumed OpenCL semantics will break loudly here. That is *exactly* what we want — it's the whole point of doing a second backend.
- **ROCm has a real paged-attention ecosystem.** Upstream projects (vLLM's ROCm fork, Composable Kernel, rocBLAS) give us reference kernels we can read and learn from.

---

## 2. Concepts

### 2.1 ROCm and HIP in 60 seconds

**ROCm** is AMD's open-source GPU compute stack: drivers, runtime, compiler, libraries (rocBLAS, rocFFT, MIOpen, Composable Kernel), and tools. Think of it as "the thing that plays the same role CUDA plays on NVIDIA".

**HIP** (Heterogeneous-Compute Interface for Portability) is the language and runtime API you actually write code against. A HIP kernel looks like this:

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

If you squint, that is a CUDA kernel. The differences are mostly cosmetic (`hipLaunchKernelGGL`, `hipMalloc`, `hipMemcpy`, `hipStream_t`). HIP code compiles to either AMD GPUs (via the ROCm compiler, producing GCN or RDNA ISA) *or* NVIDIA GPUs (via a thin wrapper over CUDA). For our purposes we target AMD; the CUDA compatibility is a bonus that we will exploit in v6.

The vocabulary mapping is worth memorizing because you will read it in every piece of ROCm documentation:

| CUDA term        | HIP term          | OpenCL term         | What it is                                    |
| ---------------- | ----------------- | ------------------- | --------------------------------------------- |
| thread           | thread            | work-item           | one instance of the kernel                    |
| warp (32)        | wavefront (32/64) | sub-group           | hardware SIMD unit                            |
| block            | block             | work-group          | group of threads sharing local memory         |
| grid             | grid              | NDRange             | the full launch                               |
| shared memory    | LDS / shared mem  | local memory        | fast scratchpad per block                     |
| stream           | stream            | command queue       | ordered async work submission                 |
| `__syncthreads`  | `__syncthreads`   | `barrier()`         | block-level sync                              |

Wavefronts on RDNA are 32 or 64 threads depending on the GPU; on CDNA (MI200/300) they are 64. This matters when tuning kernels because your inner loops usually assume a specific SIMD width.

### 2.2 The backend interface, revisited

v1 defined a backend abstraction with explicit device handles, device-owned buffers, async command queues, sharded weights, and a `{tp, pp}` parallelism plan. v2 extended that interface with quantized weight/KV-cache dtypes. v3 added the multi-token verification path needed for speculative decoding. v4's job is to *implement that same (extended) interface* against ROCm.

Concretely, the v1 interface looked something like:

```
class Backend:
    def device(self, index: int) -> Device: ...
    def alloc(self, device, nbytes) -> Buffer: ...
    def copy(self, src_buf, dst_device) -> Buffer: ...
    def queue(self, device) -> Queue: ...

    # ops — each takes a queue and returns when its work is enqueued
    def gemm(self, queue, a, b, out, ...): ...
    def rmsnorm(self, queue, x, g, out, eps): ...
    def rope(self, queue, q, k, cos, sin, positions): ...
    def paged_attention(self, queue, q, k_pool, v_pool, block_tables, ...): ...
    def silu_mul(self, queue, gate, up, out): ...
    def add(self, queue, a, b, out): ...
    def sample_argmax(self, queue, logits, out_id): ...
```

For v4 we add a second implementation of the same interface: `RocmBackend`. The model code, scheduler, KV block manager, prefix hash table, continuous-batching loop, quantization calibration, draft-model wiring, spec-decoding verifier — **none of that changes**. If any of it *does* change, we've discovered a leak in the abstraction and fix it at the abstraction level, not by hacking around it in the ROCm backend.

**Connection.** This is the single biggest learning moment of v4: the quality of the v1/v2/v3 design is measured by how few lines outside `backend/rocm/` need to change. Keep score.

### 2.3 Porting the kernels

Each OpenCL kernel from v1–v3 becomes one HIP kernel. The math does not change; the language, launch syntax, and memory model do. A rough porting recipe:

1. **GEMM.** Start with the naive port. Then replace it with **rocBLAS** (`rocblas_sgemm` / `rocblas_gemm_ex`) for the big matmuls (Q/K/V projection, MLP, LM head). rocBLAS on AMD is analogous to cuBLAS on NVIDIA — use it, don't reinvent it. Keep the hand-written kernel around as a fallback and as a learning artifact.
2. **Paged-attention kernel (with multi-token verification).** This is the kernel most likely to need real work. AMD's **Composable Kernel** library has reference implementations that are worth reading. The v1 OpenCL version ports over as a starting point but will need retuning (tile sizes, LDS usage, wavefront width). v3 already extended the kernel to verify k draft tokens at once — the HIP port inherits that complexity.
3. **Elementwise ops** (RMSNorm, RoPE, SiLU, residual add). Near-mechanical port. Fuse aggressively — each unfused op is another kernel launch and another round-trip through global memory.
4. **Sampler.** Port the argmax / top-k / spec-decoding accept-reject kernel. Tiny, but the spec-decoding accept logic from v3 has to come along.

### 2.4 Learning to profile ROCm

You cannot optimize what you cannot measure. v4 introduces three tools you will use constantly:

- **`rocprof`** — the ROCm profiler. Gives per-kernel timings, occupancy, LDS usage, memory-bandwidth counters. This is your primary tool for "why is my kernel slow".
- **`rocminfo`** — prints device properties. You need these to pick correct work-group sizes, LDS budgets, and wavefront widths at runtime instead of hardcoding them.
- **HIP events** — lightweight per-stream timestamps you call from your own code. Use them to build a per-step timeline: how much time did prefill take, how much was the decode attention, how much was the sampler? This feeds directly into the benchmark from v1 F115.

Worth learning early: the difference between **kernel time** (what the GPU spent executing) and **wall time** (what your host code experienced). When they diverge, you are either bottlenecked on the host (Python overhead, scheduling, memory allocation) or on host↔device transfers. Both look different under the profiler and have different fixes.

### 2.5 Hardening the multi-device design

v1 designed the backend to take explicit device handles and to store weights as length-1 shards, in preparation for multi-GPU. v4 is where we get to *exercise* that design for the first time on a backend that has strong native multi-device primitives. Even though v4 still runs on one GPU, we take two small steps toward v5 (which is *the* multi-GPU milestone):

1. **Enumerate all visible devices at startup**, even if we only use one. The config now picks which device index to use, with `device_index = 0` being the default. This is ~5 lines of code but forces every codepath to at least *see* multiple devices.
2. **Add a multi-device smoke test** that allocates buffers on each device independently and verifies they don't collide. It does no real compute across devices, but it proves nothing in the engine has a secret "device 0" assumption.

Real tensor-parallel / pipeline-parallel collectives stay deferred to v5 — where they will be implemented on **both OpenCL and ROCm** from day one (because the user wants to test multi-GPU on OpenCL hardware too). What v4 validates is that the *plumbing* is ready for both.

---

## 3. Features to implement in v4

| #    | Feature                                          | Concept | Notes                                                  |
| ---- | ------------------------------------------------ | ------- | ------------------------------------------------------ |
| F401 | ROCm / HIP toolchain + buildsystem integration   | 2.1     | `hipcc`, runtime discovery, CI on an AMD box if possible |
| F402 | `RocmBackend` skeleton implementing the v1/v2/v3 interface | 2.2 | Device enumeration, buffer alloc, stream management |
| F403 | HIP GEMM (naive)                                 | 2.3     | Correctness first                                      |
| F404 | rocBLAS / hipBLASLt-backed GEMM                  | 2.3     | The real production path for big matmuls              |
| F405 | HIP elementwise kernels (RMSNorm, RoPE, SiLU, add) | 2.3   | Ported from OpenCL, fused where reasonable             |
| F406 | HIP paged-attention kernel (quantized KV + multi-token verify) | 2.3 | The hardest port; reference Composable Kernel  |
| F407 | HIP on-device sampler (incl. spec-decoding accept-reject) | 2.3 | Argmax + top-k + accept logic                    |
| F408 | HIP quantized-GEMM kernels (W8A16, W4A16)        | 2.3     | v2's dequant-fused matmul, HIP flavor                  |
| F409 | Backend selector (config flag: numpy/opencl/rocm) | 2.2    | One entry point, three implementations                 |
| F410 | Conformance test: ROCm ≡ NumPy ≡ OpenCL (incl. quantized + spec-decoding paths) | 2.2 | Extend v3's conformance matrix to ROCm |
| F411 | ROCm profiling hooks (`rocprof` integration + HIP events) | 2.4 | Feeds into the benchmark harness                 |
| F412 | Multi-device enumeration and smoke test          | 2.5     | Prep for v5                                            |
| F413 | v0→v1→v2→v3→v4 benchmark update                  | 2.4     | Throughput / TTFT on AMD, side-by-side with OpenCL     |

**Build order suggestion:** F401 → F402 → F409 (select backends but ROCm is still a stub) → F403 → F405 → F407 (enough to run tiny fp16 forward passes end-to-end) → F410 (lock in fp16 correctness) → F406 → F408 (the quantized + multi-token-verify paths) → F404 (drop in rocBLAS) → F411 → F412 → F413. Correctness before speed, always.

---

## 4. Success criteria for v4

- `python examples/run_v4.py --backend rocm` runs the same workload as v1/v2/v3, on an AMD GPU, end-to-end.
- Greedy output is byte-identical across `numpy`, `opencl`, and `rocm` backends (for the fp16 path) and within tight tolerance on the quantized and speculative-decoding paths (F410).
- Measured throughput on ROCm ≥ OpenCL throughput on the same AMD GPU, for the same workload. (If it isn't, the kernels need more tuning — but correctness-first means we accept a slower first pass and optimize from there.)
- **Zero model-layer, scheduler-layer, KV-block-manager, quantization-calibration, or spec-decoding-orchestrator changes** were required to add ROCm. If any were, the abstraction is wrong and needs a retro-fix.
- `rocprof` traces exist for at least prefill, decode, and a spec-decoding step, checked into `docs/perf/` as a reference.

When those hold, v4 is done and v5 (multi-GPU TP + PP on OpenCL and ROCm) becomes the next doc.

---

## 5. What v4 still does NOT have

| Feature              | Why deferred                                                                     | Target version |
| -------------------- | -------------------------------------------------------------------------------- | -------------- |
| Multi-GPU (TP / PP)  | v4 preps the hooks; v5 implements real collectives on both OpenCL and ROCm       | v5             |
| CUDA backend         | Explicitly the LAST backend priority; near-mechanical port from HIP when we do it | v6            |
| Vulkan backend       | Optional, not a priority — only if there's a concrete reason to add it           | v7 / optional  |
| Distributed / multi-host | Far out                                                                      | later          |

---

## 6. Further reading

- **AMD ROCm documentation** — `rocm.docs.amd.com`. Start with the HIP programming guide.
- **HIP Porting Guide** — AMD's official "CUDA → HIP" guide. Reverse it mentally: everything it says about going CUDA→HIP applies, in reverse, to our eventual HIP→CUDA port in v6.
- **AMD Composable Kernel** (`ROCm/composable_kernel` on GitHub) — reference implementations of high-performance kernels, including attention. Read their paged-attention kernel before writing yours.
- **rocBLAS** and **hipBLASLt** documentation — the GEMM libraries you will call from F404.
- **vLLM ROCm fork** — the AMD-targeted branches of vLLM. Great production reference for how real systems wire paged attention to HIP kernels.
- **Simon Boehm's GEMM optimization post** (still) — written in CUDA but every optimization has a direct HIP equivalent. The single best GEMM tutorial.
- **`rocprof` user guide** — it is terse but essential.
