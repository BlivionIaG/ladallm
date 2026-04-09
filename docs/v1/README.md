# v1 — Feature Index

Plan: [`../architecture_v1.md`](../architecture_v1.md). Status legend: `☐` planned, `◐` in progress, `✓` done.

## Memory layer (paged attention + prefix caching)

| #     | Feature                                            | Doc                                                              | Status |
| ----- | -------------------------------------------------- | ---------------------------------------------------------------- | ------ |
| F101  | KV block pool allocator + refcounts                | [`f101-block-pool.md`](f101-block-pool.md)                       | ☐      |
| F102  | Per-request block table                            | [`f102-block-table.md`](f102-block-table.md)                     | ☐      |
| F103  | Paged-attention reference (NumPy)                  | [`f103-paged-attention-numpy.md`](f103-paged-attention-numpy.md) | ☐      |
| F104  | Paged-attention OpenCL kernel                      | [`f104-paged-attention-opencl.md`](f104-paged-attention-opencl.md) | ☐    |
| F104a | Prefix hash table                                  | [`f104a-prefix-hash.md`](f104a-prefix-hash.md)                   | ☐      |
| F104b | Prefix-match-on-admission hook                     | [`f104b-prefix-admission.md`](f104b-prefix-admission.md)         | ☐      |
| F104c | LRU eviction for unreferenced blocks               | [`f104c-lru-eviction.md`](f104c-lru-eviction.md)                 | ☐      |

## Scheduling layer (continuous batching)

| #    | Feature                                        | Doc                                                  | Status |
| ---- | ---------------------------------------------- | ---------------------------------------------------- | ------ |
| F105 | Multi-request scheduler with token budget      | [`f105-scheduler.md`](f105-scheduler.md)             | ☐      |
| F106 | Batch builder (mix prefill + decode)           | [`f106-batch-builder.md`](f106-batch-builder.md)     | ☐      |
| F107 | Chunked prefill                                | [`f107-chunked-prefill.md`](f107-chunked-prefill.md) | ☐      |
| F108 | Streaming token output                         | [`f108-streaming.md`](f108-streaming.md)             | ☐      |

## Compute layer (OpenCL, multi-GPU-ready interface)

| #     | Feature                                                   | Doc                                                      | Status |
| ----- | --------------------------------------------------------- | -------------------------------------------------------- | ------ |
| F109  | Backend abstraction (NumPy + OpenCL)                      | [`f109-backend-abstraction.md`](f109-backend-abstraction.md) | ☐  |
| F109a | Sharded-weights + identity `all_gather`/`reduce_scatter`  | [`f109a-sharded-weights.md`](f109a-sharded-weights.md)   | ☐      |
| F109b | Parallelism plan in model config (`tp=1, pp=1`)           | [`f109b-parallelism-plan.md`](f109b-parallelism-plan.md) | ☐      |
| F110  | OpenCL device/context/queue/buffer plumbing               | [`f110-opencl-plumbing.md`](f110-opencl-plumbing.md)     | ☐      |
| F111  | GEMM kernel (naive → tiled)                               | [`f111-gemm.md`](f111-gemm.md)                           | ☐      |
| F112  | Elementwise kernels (RMSNorm, RoPE, SiLU, add)            | [`f112-elementwise.md`](f112-elementwise.md)             | ☐      |
| F113  | On-device sampler                                         | [`f113-on-device-sampler.md`](f113-on-device-sampler.md) | ☐      |
| F114  | Backend conformance test (NumPy ≡ OpenCL)                 | [`f114-conformance.md`](f114-conformance.md)             | ☐      |

## Plumbing

| #    | Feature                            | Doc                                            | Status |
| ---- | ---------------------------------- | ---------------------------------------------- | ------ |
| F115 | Throughput / TTFT benchmark        | [`f115-benchmark.md`](f115-benchmark.md)       | ☐      |
| F116 | v1 example script                  | [`f116-example.md`](f116-example.md)           | ☐      |

## Convention

Feature docs follow [`../_feature_template.md`](../_feature_template.md) and are written when the feature is built.
