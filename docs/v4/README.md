# v4 — Feature Index (ROCm Backend)

Plan: [`../architecture_v4.md`](../architecture_v4.md). Status legend: `☐` planned, `◐` in progress, `✓` done.

| #    | Feature                                                              | Doc                                                              | Status |
| ---- | -------------------------------------------------------------------- | ---------------------------------------------------------------- | ------ |
| F401 | ROCm / HIP toolchain integration                                     | [`f401-toolchain.md`](f401-toolchain.md)                         | ☐      |
| F402 | `RocmBackend` skeleton (v1/v2/v3 interface)                          | [`f402-rocm-skeleton.md`](f402-rocm-skeleton.md)                 | ☐      |
| F403 | HIP GEMM (naive)                                                     | [`f403-hip-gemm-naive.md`](f403-hip-gemm-naive.md)               | ☐      |
| F404 | rocBLAS / hipBLASLt-backed GEMM                                      | [`f404-rocblas-gemm.md`](f404-rocblas-gemm.md)                   | ☐      |
| F405 | HIP elementwise kernels (RMSNorm, RoPE, SiLU, add)                   | [`f405-hip-elementwise.md`](f405-hip-elementwise.md)             | ☐      |
| F406 | HIP paged-attention kernel (quantized KV + multi-token verify)       | [`f406-hip-paged-attention.md`](f406-hip-paged-attention.md)     | ☐      |
| F407 | HIP on-device sampler (incl. spec-decoding accept-reject)            | [`f407-hip-sampler.md`](f407-hip-sampler.md)                     | ☐      |
| F408 | HIP quantized-GEMM kernels (W8A16, W4A16)                            | [`f408-hip-quant-gemm.md`](f408-hip-quant-gemm.md)               | ☐      |
| F409 | Backend selector config flag                                         | [`f409-backend-selector.md`](f409-backend-selector.md)           | ☐      |
| F410 | Conformance test: ROCm ≡ NumPy ≡ OpenCL (quant + spec-decoding)      | [`f410-conformance.md`](f410-conformance.md)                     | ☐      |
| F411 | ROCm profiling hooks (`rocprof` + HIP events)                        | [`f411-profiling.md`](f411-profiling.md)                         | ☐      |
| F412 | Multi-device enumeration + smoke test                                | [`f412-multi-device-smoke.md`](f412-multi-device-smoke.md)       | ☐      |
| F413 | v0→v1→v2→v3→v4 benchmark update                                      | [`f413-benchmark.md`](f413-benchmark.md)                         | ☐      |

## Convention

Feature docs follow [`../_feature_template.md`](../_feature_template.md) and are written when the feature is built.
