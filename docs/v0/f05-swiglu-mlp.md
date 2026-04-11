# F5 — SwiGLU MLP

> Version: v0  •  Concept: [4.10 SwiGLU MLP](../architecture_v0.md#410-swiglu-mlp)
> Depends on: F4 (Attention)  •  Depended on by: F6 (Decoder block)

## What this feature is

The SwiGLU MLP is the feed-forward network that runs after attention in each transformer layer. While attention mixes information *across* tokens, the MLP transforms each token's representation *independently*, adding non-linearity and model capacity. SwiGLU is a gated variant that uses a "gate" branch to control which features pass through.

## Why it exists

Without the MLP, the transformer would just be attention layers rearranging the same vectors. The MLP gives the model the ability to:
- Learn complex non-linear transformations
- Store factual knowledge in its weights
- Transform representations between attention rounds

SwiGLU specifically (vs older MLP designs) achieves better quality for the same parameter budget by using a gating mechanism that selectively passes information.

## The concept, refreshed

### Inputs and Outputs

```python
out = swiglu_mlp(x, W_gate, W_up, W_down)
# x:      [seq_len, d_model]  - Input activations
# W_gate: [d_model, d_ff]     - Gate projection weights
# W_up:   [d_model, d_ff]     - Up projection weights
# W_down: [d_ff, d_model]     - Down projection weights
# Returns:
#   out:  [seq_len, d_model]  - Transformed activations
```

For SmolLM2-135M:
- `d_model` = 576 (hidden_size)
- `d_ff` = 1536 (intermediate_size)
- Ratio: 2.67× (SwiGLU uses less than the 4× of standard MLPs)

### The Math

**Step 1: Parallel projections to d_ff**

```
gate = x @ W_gate    # [seq_len, d_model] @ [d_model, d_ff] → [seq_len, d_ff]
up   = x @ W_up      # [seq_len, d_model] @ [d_model, d_ff] → [seq_len, d_ff]
```

**Step 2: SiLU activation (Swish)**

SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))

```
gated = silu(gate) * up    # elementwise: [seq_len, d_ff] * [seq_len, d_ff]
```

The gate branch decides which channels of the up projection get to pass through.

**Step 3: Project back to d_model**

```
out = gated @ W_down     # [seq_len, d_ff] @ [d_ff, d_model] → [seq_len, d_model]
```

### The Gating Mechanism

SwiGLU replaces the traditional activation → multiply pattern with:

```
Traditional:  out = activation(x @ W1) @ W2
SwiGLU:       out = (silu(x @ W_gate) * (x @ W_up)) @ W_down
```

The `silu(gate)` acts as a soft switch (values between 0 and 1) controlling each feature dimension independently.

## How to implement it

### Step 1: Implement sigmoid

Simple but needed for SiLU:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### Step 2: Implement SiLU activation

Either inline or as helper:

```python
def silu(x):
    return x * sigmoid(x)
```

### Step 3: Implement SwiGLU MLP

Memory-optimized version using in-place operations:

```python
def swiglu_mlp(x, W_gate, W_up, W_down):
    gate = x @ W_gate                    # [seq_len, d_ff]
    # In-place: reuse gate buffer for SiLU output
    np.multiply(gate, sigmoid(gate), out=gate)
    # In-place: multiply with up projection (computed on the fly)
    np.multiply(gate, x @ W_up, out=gate)
    return gate @ W_down
```

This saves ~10-20% memory by reusing the `gate` buffer instead of allocating temporaries.

### Step 4: Add to model layer

Integrate into the decoder block:

```python
class ModelLayer:
    def __init__(self, weights, layer_idx, ...):
        # ... attention weights ...
        
        # MLP weights
        prefix = f"model.layers.{layer_idx}.mlp"
        self.w_gate = weights[f"{prefix}.gate_proj.weight"]
        self.w_up = weights[f"{prefix}.up_proj.weight"]
        self.w_down = weights[f"{prefix}.down_proj.weight"]
    
    def forward(self, x, ...):
        # ... attention path ...
        
        # MLP path
        mlp_out = swiglu_mlp(x, self.w_gate, self.w_up, self.w_down)
        x = x + mlp_out  # residual connection
        
        return x
```

## Edge cases and gotchas

### Memory optimization pitfalls

In-place operations (`out=` parameter) are great for memory but:
- Make debugging harder (you lose intermediate values)
- Can cause subtle bugs if you accidentally reuse a buffer you still need
- Don't use if you need gradients (training), but fine for inference

### Numerical stability

`sigmoid` of large negative values underflows to 0. For very large positive values, `exp(-x)` underflows and `sigmoid(x)` returns 1. This is usually fine for inference.

### Weight shape confusion

Common mistake: mixing up `W_down` shape. It should be `[d_ff, d_model]` (projects down), not `[d_model, d_ff]`.

### Dtype consistency

Ensure all inputs are the same dtype (float32). Mixed dtypes (e.g., float16 weights, float32 input) can cause unexpected casting or performance issues.

### SmolLM2 specific

The model uses `intermediate_size=1536` which is 2.67× `hidden_size=576`. Don't assume 4× expansion like older transformers.

## How to test it

### Unit tests

- **sigmoid**: Zero → 0.5, positive > 0.5, negative < 0.5, range (0,1)
- **swiglu_mlp**: Correct output shape, single token handling, various batch sizes
- **determinism**: Same input → same output
- **numerical sanity**: No NaN or Inf with random inputs
- **zero weights**: Zero input + zero weights → zero output

### Integration tests

- **SmolLM2 dimensions**: Test with d_model=576, d_ff=1536
- **End-to-end**: MLP integrated into decoder block
- **Memory check**: Verify in-place ops work correctly

### Observable effects

With MLP implemented:
- Model can express non-linear transformations
- Generations become more coherent (MLP stores factual patterns)
- Memory usage increases linearly with batch size
- Slightly slower than attention on short sequences (more matmuls)

## Where it lives in the codebase

- `src/ladallm/mlp.py` — `sigmoid()`, `swiglu_mlp()`
- `tests/test_mlp.py` — Comprehensive test suite

## Further reading

- [GLU Variants Paper](https://arxiv.org/abs/2002.05202) — Noam Shazeer, "GLU Variants Improve Transformer"
- [SwiGLU in PaLM](https://arxiv.org/abs/2204.02311) — Chowdhery et al., uses SwiGLU in production
- [Llama architecture](https://arxiv.org/abs/2302.13971) — Touvron et al., popularized SwiGLU in open models