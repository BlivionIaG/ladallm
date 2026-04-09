# F02 — RMSNorm

> Version: v0  •  Concept: [4.4 RMSNorm](../architecture_v0.md#44-rmsnorm)  
> Depends on: F1 (weight loading)  •  Depended on by: F6 (decoder block)

## What this feature is

Root Mean Square Layer Normalization (RMSNorm) rescales each token's activation vector to unit RMS (root mean square) magnitude, then applies a learned per-channel scale. It's the normalization used in modern Llama-style models before every attention and MLP sublayer.

## Why it exists

Without normalization, activations drifting through 30+ transformer layers would explode or vanish numerically. RMSNorm keeps the "volume" of activations controlled and stable. It's a stripped-down LayerNorm that skips mean-centering and bias — cheaper to compute, same quality in practice.

Without it: the model outputs garbage or NaN within a few layers.

## The concept, refreshed

**Input:**
- `x`: `[seq_len, hidden_size]` — activations from previous layer (or embeddings)
- `weight`: `[hidden_size]` — learned scale factor (gamma), loaded from checkpoint
- `eps`: scalar (default 1e-6) — prevents division by zero

**Output:**
- `out`: `[seq_len, hidden_size]` — rescaled activations, same shape

**The math:**

```
rms(x) = sqrt(mean(x_i²) + eps)       # scalar per token
out = (x / rms(x)) * weight           # broadcast division, then scale
```

**Expanded per-token:**
```python
for each token t in 0..seq_len-1:
    rms_t = sqrt(mean(x[t, :] ** 2) + eps)
    out[t, :] = (x[t, :] / rms_t) * weight
```

**Invariants:**
- Output has same shape as input
- Each token normalized independently (no mixing across sequence)
- Weight is multiplicative only (no additive bias)

## How to implement it

### Step 1: Compute RMS per token

**What:** Calculate the root mean square across the feature dimension for each token.

**Why:** We need the normalization factor for each token independently.

**Guidance:** Use `np.mean(x ** 2, axis=-1, keepdims=True)`:
- `axis=-1` → normalize across features (not tokens)
- `keepdims=True` → preserves shape for broadcasting: `[seq_len, 1]`
- `** 2` → elementwise square
- `np.sqrt(... + eps)` → RMS with numerical stability

### Step 2: Normalize and scale

**What:** Divide input by RMS, then multiply by learned weight.

**Why:** Division normalizes to unit RMS; weight rescales to learned magnitude.

**Guidance:** `return (x / rms) * weight`. NumPy broadcasting handles the shape mismatch: `[seq_len, hidden]` / `[seq_len, 1]` → `[seq_len, hidden]`.

### Step 3: Verify shapes

**What:** Add assertions or checks to catch shape errors early.

**Why:** Wrong axis choice is a common bug that silently produces wrong results.

**Guidance:** Check `x.shape[-1] == weight.shape[0]` before computation.

## Edge cases and gotchas

- **Epsilon placement:** `mean(x**2 + eps)` is wrong — adds eps to every element. Correct: `mean(x**2) + eps` adds to the final mean.
- **keepdims:** Forgetting `keepdims=True` makes `rms` shape `[seq_len]`, which cannot broadcast against `[seq_len, hidden]`.
- **Axis confusion:** `axis=0` normalizes across tokens (wrong — mixes different words). `axis=-1` normalizes per token (correct).
- **dtype consistency:** RMSNorm should preserve input dtype (F16 → F16, F32 → F32). Weight should match or be cast automatically.
- **Numerical overflow:** `x**2` on large F16 values can overflow. Consider upcasting to F32 internally if needed.

## How to test it

**Unit tests:**
- Input all ones: `rms_norm(np.ones(1, N), np.ones(N))` → all ones (RMS of 1s is 1, no change)
- Zero input: very small epsilon should prevent division by zero, output near zero
- Shape preservation: output.shape == input.shape
- Broadcasting: verify it works with batch dimension `[batch, seq, hidden]`

**Integration tests:**
- Load SmolLM2 weights, extract `model.layers.0.input_layernorm.weight`
- Run on random input, verify no NaN/Inf
- Compare output statistics (mean, std) are reasonable

**Observable effects:**
- Without RMSNorm: decoder block outputs explode or vanish within layers
- With RMSNorm: stable activations throughout the 30-layer stack

## Where it lives in the codebase

- `src/ladallm/cli.py` — `rms_norm(x, weight, eps=1e-6)` function

## Further reading

- *Root Mean Square Layer Normalization* — Zhang & Sennrich (2019). The original RMSNorm paper. Shows it matches LayerNorm quality with fewer params.
- Llama 2 / Llama 3 papers — Both use pre-norm with RMSNorm.
