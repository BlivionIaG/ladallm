# Architecture v2 — Quantization (Weights and KV Cache)

> **Prerequisites:** [`architecture_v0.md`](architecture_v0.md) and [`architecture_v1.md`](architecture_v1.md). v2 assumes paged attention, continuous batching, prefix caching, and the OpenCL backend from v1 are all working.

v2 has a very specific motivation: **the target OpenCL hardware has limited VRAM**. Even a modest Llama-style model at fp16 plus a reasonable amount of KV cache will not fit comfortably. Without quantization, v1 technically runs but cannot demonstrate its own wins (concurrency, long prefixes, batched prefill) because we cannot fit enough in memory to exercise them. v2 exists to unlock that.

Two things get quantized in v2:

1. **Weights** — shrink the model itself. A 7B fp16 model is ~14 GB; at int4 it is ~3.5 GB. This is the difference between "model fits with room for work" and "model doesn't fit at all".
2. **KV cache** — shrink the per-token runtime cost. At long contexts the KV cache dominates memory and directly caps concurrency; cutting it from fp16 to int8 roughly doubles the number of concurrent requests you can hold.

Both land in v2 because **both are needed to reach the v1 vision on VRAM-limited hardware**. Either one alone helps; together they compound.

---

## 1. What we are building

A quantization subsystem that lets the engine load models with quantized weights, store the KV cache in low-precision formats, and run the model's forward pass correctly on those compressed tensors — without any changes to the model code, the scheduler, or the block manager. All the work happens inside the **C++ compute layer** and the weight loader.

After v2:

- The model weights can be stored on disk and in device memory in **fp16** (baseline), **int8 weight-only (W8A16)**, or **int4 weight-only (W4A16)** formats, selected by config.
- The KV cache can be stored in **fp16** (baseline), **int8**, or **fp8** (if the device supports it), selected by config.
- A simple **post-training quantization (PTQ)** script converts a fp16 checkpoint to a quantized one using round-to-nearest (RTN) per-channel scales. A more accurate **GPTQ**-style calibrator is a stretch goal.
- An **accuracy evaluation** compares fp16 vs quantized perplexity on a small held-out set, so we *see* the quality cost of each format.
- A **VRAM budget tool** prints, for a given model + context length + concurrency, the memory each configuration needs — so picking a config is not guesswork.

### Why quantization is the right v2

- **Unlocks v1 on real hardware.** Without it, the OpenCL backend cannot actually serve the workloads v1 was designed for.
- **Sits inside the v1 C++ compute layer.** Quantized matmul is a *backend kernel*, not a model-code change. Every quantization technique we add becomes a new implementation of the same `gemm` / `paged_attention` C++ interface. If we got the v1 abstraction right, v2 is entirely additive.
- **Teaches the numerical half of inference.** v0 taught the shape math; v1 taught the systems side; v2 teaches the *numerical precision* side. Rounding error, outliers, per-channel scales, the difference between weight-only and activation quantization — these are core inference engine concepts and every production engine has them.
- **Orthogonal to the backend port.** Doing quantization *before* ROCm (v4) means when we port to ROCm, we port *both* the fp16 and quantized kernels at once (and the v3 spec-decoding kernels too). The alternative — quantize after ROCm — means doing kernel work twice.

---

## 2. Concepts

Same shape as v0/v1: **Intuition → Math/Mechanism → Connections**.

### 2.1 Why quantize at all (three separate wins)

Quantization is usually sold as "saves memory", but it actually wins on three fronts, and understanding which win matters is half the battle when choosing a scheme.

1. **Memory footprint.** The obvious one. A weight matrix stored as int4 is 4× smaller than fp16. A KV cache stored as int8 is 2× smaller than fp16. On VRAM-limited hardware this is the difference between "runs" and "OOM".
2. **Memory bandwidth.** Decode is memory-bound. Every step re-reads the model weights and the entire KV cache from memory. Smaller = fewer bytes to move = faster step, independent of whether you have *enough* memory. A W4A16 model decodes roughly 2–3× faster than fp16 on the same hardware even when fp16 fits.
3. **Compute.** If your hardware has int8 matmul units (most do nowadays) you can actually *compute* in the lower precision, not just store. v2 mostly does **weight-only** quantization (store low, compute fp16) because it is simpler and portable; true low-precision compute is a v2-stretch or v4 topic.

For our VRAM-limited case, (1) is the life-or-death win and (2) is the bonus.

### 2.2 The big picture of a quantization scheme

Every weight/activation quantization scheme answers the same four questions. Keep them in mind as we go through specific schemes — the schemes only differ in their answers.

1. **What's the target dtype?** int8, int4, fp8, fp4, nf4… each has different range, precision, and hardware support.
2. **What's the granularity of the scale?** Per-tensor (one scale for the whole matrix), per-channel (one scale per row or column), per-group (one scale per contiguous block of, say, 128 values). Finer = more accurate but more metadata.
3. **Symmetric or asymmetric?** Symmetric uses `q = round(x / scale)` and maps zero to zero — cheapest. Asymmetric uses `q = round(x / scale) + zero_point` and can better fit skewed distributions.
4. **When does dequantization happen?** "Weight-only" schemes dequantize back to fp16 inside the matmul kernel just before the multiply. "Integer" schemes keep things in int8 all the way through the multiply-accumulate.

v2 defaults: **symmetric, per-channel (weights) / per-token (activations and KV), weight-only compute**. Simplest thing that works; we'll name the fancier alternatives as we go.

### 2.3 Weight quantization — the math

Given a weight row `w` of length K (one output channel of a linear layer), symmetric per-channel quantization to `b` bits is:

```
s        = max(|w|) / (2^(b-1) - 1)     # one scale per channel
q        = round(w / s)                  # integers in [-2^(b-1)+1, 2^(b-1)-1]
dequant  = q * s                         # approximation of w
```

Storage: instead of `K` fp16 numbers, we store `K` `b`-bit integers plus one fp16 scale per row. For int4 with K=4096, that's `4096 * 4 / 8 = 2048` bytes of weights plus `2` bytes of scale, vs. `8192` bytes at fp16 — a 4× shrink, minus epsilon.

**Group quantization.** A single scale per row loses accuracy when the row has wildly different magnitudes in different regions. **Group quantization** breaks the row into contiguous groups (typically 128 values) and gives each group its own scale. More metadata, much better accuracy — this is what GPTQ, AWQ, and llama.cpp's Q4_K all use. For v2 we should at least support `group_size = -1` (per channel, simplest) and `group_size = 128` (the standard).

**Round-to-nearest (RTN)** is the simplest PTQ algorithm: just apply the formula above. It is surprisingly good at 8 bits, okay at 4 bits for most weights, and **bad** at 4 bits for a small set of "outlier" weights that dominate the row's magnitude.

**GPTQ** is a more careful algorithm: it quantizes weights *one column at a time*, and after each column it updates the remaining (not-yet-quantized) columns to compensate for the rounding error. The update uses second-order information from a small calibration dataset. It takes minutes (not seconds) to run and gives dramatically better 4-bit accuracy. For v2 we start with RTN and treat GPTQ as a stretch goal.

**AWQ** (Activation-aware Weight Quantization) observes that *which* weights are "outliers" depends on the *activation* they'll be multiplied by. It rescales weights and activations *before* quantization so the outliers are spread more evenly. Also calibration-based. Another stretch goal.

**Connection.** All three schemes produce the same *runtime* artifact: a block of ints plus some scales. The dequant-and-matmul kernel does not care how the ints were chosen. So we can ship RTN in v2 and swap in GPTQ/AWQ later with zero kernel changes.

### 2.4 Weight quantization — the kernel

On the runtime side, there are two families of kernels:

- **Dequant-then-matmul (separate).** Dequantize the whole weight matrix back to fp16 in scratch memory, then do a standard fp16 GEMM. Correct but defeats the point — we're back to fp16 bandwidth.
- **Dequant-fused matmul.** The matmul kernel reads quantized weights from memory, dequantizes them register-side right before the multiply, and accumulates in fp16 (or fp32). This is the whole win: weights cross the memory bus in their compressed form; only the tiny register-side dequant runs at full precision.

v2 implements the second kind. A fused **W4A16** kernel reads 8 int4 weights per 32-bit word, unpacks them, multiplies by the scale, and multiplies by the fp16 activation — all in registers. This is what makes quantized inference *fast*, not just *small*.

Writing this kernel from scratch is a meaningful exercise. Two things to get right:

- **Packing layout.** Int4 has to be stored in a way the kernel can unpack efficiently. Naive packing (two int4s per byte) works but creates awkward shift-and-mask patterns. Interleaved layouts (e.g. storing `[w0, w2, w4, w6]` in one 32-bit word and `[w1, w3, w5, w7]` in another) reduce instruction count. llama.cpp's `.gguf` file format is a good reference for production layouts.
- **Dequant + FMA fusion.** The compiler usually will *not* fuse your dequant with the multiply unless you make it obvious. Write the inner loop so the dequantized value is used exactly once, immediately. Check the generated code with the GPU profiler — you should see a tight sequence of unpack → multiply-add, with nothing in between.

**Connection.** The kernel signature becomes part of the backend interface: `gemm_w4a16(queue, activations_fp16, weights_int4, scales_fp16, out_fp16, ...)`. Every backend implements its own version; the model code calls the interface and does not know the difference.

### 2.5 KV cache quantization

Weights are quantized **once**, at load time. The KV cache is different: it is written *at every decode step* and read at every subsequent one. So KV quantization has to be cheap on *both* sides of the store/load boundary.

**What we quantize.** The K and V tensors at every position, stored in the paged block pool. One scale per block (or per token within a block) is the standard. Typical target: **int8 symmetric per-token**.

**The math.** For a single new-token K vector `k` of length `head_dim * n_kv_heads`:

```
s       = max(|k|) / 127
q_k     = round(k / s)                    # int8
```

Store `q_k` in the block and `s` in a parallel scale array. On read:

```
k_back  = q_k * s                          # fp16 again
```

The dequant is one multiply per value and happens *inside* the attention kernel, register-side, just like the weight-only dequant for GEMM. No extra memory traffic beyond the ints and the scales.

**Per-token vs per-block scales.** Per-token (one scale per sequence position) is the most accurate and what vLLM-style implementations do. Per-block is slightly less accurate but has less metadata and a friendlier memory layout. Per-channel-per-token is the gold standard but adds a lot of scales. v2 default: **per-token, symmetric, int8**.

**fp8.** If the hardware supports fp8 (E4M3 or E5M2), it is often a better choice than int8 for KV cache: same memory footprint, better dynamic range, no scale metadata needed (the format has its own exponent). Most AMD and NVIDIA GPUs from the last two years support fp8. OpenCL support is patchy depending on the vendor — we'll check at device discovery and fall back to int8 when it's unavailable.

**What changes in the paged-attention kernel.** The kernel that computes `softmax(Q @ K^T / sqrt(d)) @ V` now has to dequantize K and V inline as it reads them from the block pool. This is one of the harder kernels in v2 because paged attention is already the most complex kernel in the engine, and now it has an extra dequant on every load. The reward is that the memory bandwidth it consumes — which dominates decode time — is cut in half.

**Connection.** Quantized KV cache is the single biggest lever for concurrency on VRAM-limited hardware. At long contexts the KV cache is the dominant consumer; halving it roughly doubles how many requests you can hold at once. This is the whole reason KV-cache quantization is in v2 and not later.

### 2.6 Accuracy: what we pay for the shrink

Quantization is lossy. Every bit you remove costs some accuracy. The question is *how much*, and v2 exists in part to measure it honestly.

Rough rules of thumb from the literature and from llama.cpp / AWQ / GPTQ papers:

| Scheme                        | Typical quality cost                          |
| ----------------------------- | ---------------------------------------------- |
| fp16 (baseline)               | 0                                              |
| int8 weight-only RTN          | negligible (< 0.1 perplexity increase)         |
| int8 KV cache                 | negligible                                     |
| int4 weight-only RTN          | noticeable on hard prompts (~0.5–1.0 ppl)      |
| int4 weight-only RTN, g=128   | small (~0.2 ppl)                               |
| int4 GPTQ, g=128              | very small (~0.05–0.1 ppl)                     |
| int4 AWQ, g=128               | very small (~0.05 ppl), better on instruct     |
| fp8 KV cache                  | negligible                                     |

These numbers depend heavily on model family, calibration data, and evaluation set. **Do not trust them** — measure. v2 includes a simple perplexity harness (F208) precisely so we stop guessing and start measuring on the actual model we care about.

**Outliers.** The reason 4-bit quantization can go wrong is that a few weight magnitudes in each layer are *much* larger than the rest (orders of magnitude). A per-row scale is dominated by those outliers and loses precision on everything else. Group quantization helps because each group only has to fit a few hundred values. AWQ helps because it *moves* the outliers to activations (which are then kept at fp16). SmoothQuant helps similarly. Knowing this in advance means we will not be confused when RTN-int4 has worse-than-expected quality on certain layers.

**Connection.** Accuracy measurement is not an afterthought; it's a first-class v2 feature. Any quantization scheme we add must have a perplexity number attached to it or we cannot responsibly recommend it.

### 2.7 Putting it together: the VRAM math

Here is a worked example for intuition. Suppose you have a 7B Llama-style model with `d_model = 4096`, `n_layers = 32`, `n_kv_heads = 8`, `head_dim = 128`, and you want to serve up to `C` concurrent requests at up to `L` tokens each.

- **Model weights.**
  - fp16: 7B × 2 bytes ≈ **14 GB**
  - int8 (W8A16): 7B × 1 byte ≈ **7 GB**
  - int4 (W4A16, group=128): 7B × 0.5 bytes + scales ≈ **3.8 GB**

- **KV cache per token per layer** = `2 (K and V) × n_kv_heads × head_dim × dtype_size`
  - fp16: `2 × 8 × 128 × 2 = 4096` bytes = 4 KB per token per layer
  - int8: `2 × 8 × 128 × 1 + scales` ≈ 2 KB
- **Total KV cache** = `n_layers × L × C × per-token-per-layer`
  - With `n_layers = 32`, `L = 2048`, `C = 8`:
    - fp16: `32 × 2048 × 8 × 4 KB = 2 GB`
    - int8: ~1 GB

So a W4A16 + int8-KV configuration takes roughly `3.8 + 1 = 4.8 GB` for the model + cache, compared to `14 + 2 = 16 GB` for fp16 + fp16-KV. The difference is whether you can run on an 8 GB GPU at all.

**The VRAM budget tool (F208)** makes this calculation a config query: you tell it the model, the context length, and the desired concurrency, and it tells you which quantization configs fit.

**Connection.** This is the number that justifies the whole version. If the tool says "fp16 needs 16 GB, W4A16+int8-KV needs 4.8 GB, your card has 8 GB", then v2 is the version that turns "cannot run" into "runs comfortably".

---

## 3. Features to implement in v2

### Weight side

| #    | Feature                                           | Concept | Notes                                                   |
| ---- | ------------------------------------------------- | ------- | ------------------------------------------------------- |
| F201 | Quantized weight file format + loader             | 2.3     | Extend the safetensors/npz loader to carry scales/zero-points |
| F202 | RTN PTQ script (fp16 → W8A16 / W4A16)             | 2.3     | `tools/quantize.py`; one model, one command             |
| F203 | Quantized weight dtype + per-channel/per-group scales in the weights struct | 2.3 | Types flow through the model runner unchanged    |
| F204 | Dequant-fused GEMM kernel (W8A16)                 | 2.4     | Start here; easier than int4                            |
| F205 | Dequant-fused GEMM kernel (W4A16, group=128)      | 2.4     | Int4 packing + fused dequant                            |
| F206 | GPTQ calibrator (stretch)                         | 2.3     | Optional, much better int4 accuracy                     |
| F207 | AWQ calibrator (stretch)                          | 2.3     | Optional                                                |

### KV cache side

| #    | Feature                                           | Concept | Notes                                                   |
| ---- | ------------------------------------------------- | ------- | ------------------------------------------------------- |
| F208 | Int8 KV block layout (data + per-token scale array) | 2.5   | Extends the paged block pool with a parallel scale pool |
| F209 | Quant-on-write in the paged-attention write path  | 2.5     | One cheap kernel: take fp16 K/V, write int8 + scale     |
| F210 | Dequant-on-read in the paged-attention kernel     | 2.5     | Modify the v1 paged-attention kernel to dequant inline  |
| F211 | fp8 KV variant, device-capability gated           | 2.5     | Falls back to int8 if hardware does not support fp8     |

### Measurement and tooling

| #    | Feature                                           | Concept | Notes                                                   |
| ---- | ------------------------------------------------- | ------- | ------------------------------------------------------- |
| F212 | Perplexity harness (fp16 vs each quant config)    | 2.6     | Small held-out set; produces a table                    |
| F213 | VRAM budget calculator                            | 2.7     | CLI that reports memory per config                      |
| F214 | Conformance matrix: NumPy ≡ OpenCL for each quant config | 2.4, 2.5 | Extends v1's F114; tolerance-based for quantized paths |
| F215 | Benchmark update: throughput / TTFT / concurrency per quant config | 2.7 | v0→v1→v2 comparison with memory annotations  |
| F216 | v2 example script                                 | —       | `run_v2.py --model ... --weight-quant w4a16 --kv-quant int8` |

**Build order suggestion:** F201 → F202 → F204 (W8A16 path working end-to-end first — int8 is the easy case) → F212 → F214 (lock in fp16 ≈ W8A16 correctness and measure) → F203 pieces for int4 → F205 (W4A16) → F208 → F209 → F210 (KV int8) → F213 → F215 → F211 (fp8 if hardware allows) → F216 → F206/F207 as stretch. Correctness and measurement before every kernel optimization.

---

## 4. Success criteria for v2

- The engine loads and runs a model in fp16, W8A16, and W4A16 weight formats, with the config flag being the only code change between them.
- The engine runs with fp16 and int8 KV cache formats, config-selected.
- `python tools/vram_budget.py` reports a coherent memory budget for each configuration.
- The perplexity harness (F212) produces a table of quality costs per configuration, checked into `docs/perf/quantization.md`.
- On the target OpenCL hardware, at least one `(weight-quant, kv-quant)` combination **fits comfortably** and runs a meaningful concurrent workload (where v1's fp16 path did not).
- Zero changes to `models/`, `scheduler.py`, or `engine.py` outside the weight loader and the backend ops. Every change is in the backend or in `tools/`.

When those hold, v2 is done and v3 (speculative decoding — built on top of quantized target + draft models) becomes the next doc.

---

## 5. What v2 still does NOT have

| Feature              | Why deferred                                                                    | Target version |
| -------------------- | ------------------------------------------------------------------------------- | -------------- |
| Speculative decoding | Lands right after quantization — uses a small draft model + the v2 target model | v3             |
| ROCm backend         | Next in backend priority; ports fp16, quantized, and spec-decoding kernels      | v4             |
| Multi-GPU (TP / PP)  | v4 preps the hooks; v5 implements real collectives on both OpenCL and ROCm      | v5             |
| CUDA backend         | Explicitly the LAST backend priority (user preference)                          | v6             |
| Vulkan backend       | Optional, not a priority                                                        | v7 / optional  |
| Integer activation quantization (W8A8) | Requires int8 matmul paths and careful calibration; weight-only is enough for v2 | later |
| Mixed precision per layer | All layers quantized uniformly in v2                                       | later          |

---

## 6. Further reading

- **GPTQ** — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (Frantar et al., 2022). The standard reference for accurate 4-bit weight quantization.
- **AWQ** — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., 2023). The outlier-aware alternative to GPTQ; often better on instruction-tuned models.
- **SmoothQuant** — *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (Xiao et al., 2022). For when you want W8A8, not just W8A16.
- **llama.cpp GGUF format documentation** — the production reference for int4 weight layouts and group quantization. Read the `Q4_K`, `Q5_K`, `Q6_K` definitions.
- **bitsandbytes** — the PyTorch library that popularized 8-bit and 4-bit inference; source code is readable and their kernels are a good reference.
- **vLLM KV cache quantization docs** (`vllm/attention/ops/`) — real-world paged-attention kernels with int8 and fp8 KV support.
- **FP8 Formats for Deep Learning** (Micikevicius et al., 2022) — the paper that standardized E4M3 and E5M2.
