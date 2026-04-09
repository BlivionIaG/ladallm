# v2 — Feature Index (Quantization)

Plan: [`../architecture_v2.md`](../architecture_v2.md). Status legend: `☐` planned, `◐` in progress, `✓` done.

## Weight side

| #    | Feature                                               | Doc                                                        | Status |
| ---- | ----------------------------------------------------- | ---------------------------------------------------------- | ------ |
| F201 | Quantized weight file format + loader                 | [`f201-quant-loader.md`](f201-quant-loader.md)             | ☐      |
| F202 | RTN PTQ script (fp16 → W8A16 / W4A16)                 | [`f202-rtn-ptq.md`](f202-rtn-ptq.md)                       | ☐      |
| F203 | Quantized dtypes + scales in the weights struct      | [`f203-quant-dtypes.md`](f203-quant-dtypes.md)             | ☐      |
| F204 | Dequant-fused GEMM (W8A16)                            | [`f204-gemm-w8a16.md`](f204-gemm-w8a16.md)                 | ☐      |
| F205 | Dequant-fused GEMM (W4A16, group=128)                 | [`f205-gemm-w4a16.md`](f205-gemm-w4a16.md)                 | ☐      |
| F206 | GPTQ calibrator (stretch)                             | [`f206-gptq.md`](f206-gptq.md)                             | ☐      |
| F207 | AWQ calibrator (stretch)                              | [`f207-awq.md`](f207-awq.md)                               | ☐      |

## KV cache side

| #    | Feature                                               | Doc                                                        | Status |
| ---- | ----------------------------------------------------- | ---------------------------------------------------------- | ------ |
| F208 | Int8 KV block layout (data + scale array)             | [`f208-int8-kv-layout.md`](f208-int8-kv-layout.md)         | ☐      |
| F209 | Quant-on-write in the KV write path                   | [`f209-kv-quant-write.md`](f209-kv-quant-write.md)         | ☐      |
| F210 | Dequant-on-read in the paged-attention kernel         | [`f210-kv-dequant-read.md`](f210-kv-dequant-read.md)       | ☐      |
| F211 | fp8 KV variant (device-capability gated)              | [`f211-fp8-kv.md`](f211-fp8-kv.md)                         | ☐      |

## Measurement and tooling

| #    | Feature                                               | Doc                                                        | Status |
| ---- | ----------------------------------------------------- | ---------------------------------------------------------- | ------ |
| F212 | Perplexity harness                                    | [`f212-perplexity.md`](f212-perplexity.md)                 | ☐      |
| F213 | VRAM budget calculator                                | [`f213-vram-budget.md`](f213-vram-budget.md)               | ☐      |
| F214 | Conformance matrix (NumPy ≡ OpenCL per quant config)  | [`f214-quant-conformance.md`](f214-quant-conformance.md)   | ☐      |
| F215 | Benchmark update with per-config memory annotations   | [`f215-quant-benchmark.md`](f215-quant-benchmark.md)       | ☐      |
| F216 | v2 example script                                     | [`f216-example.md`](f216-example.md)                       | ☐      |

## Convention

Feature docs follow [`../_feature_template.md`](../_feature_template.md) and are written when the feature is built.
