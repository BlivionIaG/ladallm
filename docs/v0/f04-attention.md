# F4 — Attention + KV-cache reads

> Version: v0  •  Concept: [4.5-4.8 Linear projections, Multi-head attention, Causal masking, GQA](../architecture_v0.md#45-linear-projections-q-k-v)
> Depends on: F3 (RoPE)  •  Depended on by: F5 (SwiGLU MLP), F6 (Decoder block)

## What this feature is

Attention is the core mechanism of transformer models. It allows each token to "look at" other tokens in the sequence and gather relevant information. For a decoder-only LLM (like Llama), this is **self-attention**: each token attends to itself and all previous tokens.

The feature implements:
- **Q/K/V projections**: Converting input vectors into Query, Key, and Value representations
- **Multi-head attention**: Splitting attention across multiple heads for parallel pattern recognition
- **Causal masking**: Preventing tokens from attending to future positions (critical for autoregressive generation)
- **GQA (Grouped Query Attention)**: Sharing KV heads across query heads to reduce memory
- **KV cache integration**: Storing K/V tensors to avoid recomputation during generation

## Why it exists

Without attention, the model would be a feed-forward network with no context awareness. The key insight of "Attention Is All You Need" (Vaswani et al., 2017) was that direct connections between all positions (modulated by learned attention weights) outperform recurrence and convolution for sequence modeling.

The KV cache optimization exists because in autoregressive generation, we compute K and V for each token exactly once, but we need them for every subsequent token's attention computation. Storing them reduces per-token cost from O(n²) to O(n).

## The concept, refreshed

### Inputs and Outputs

```python
# QKV Projection
q, k, v = compute_qkv(
    x,        # [seq_len, hidden_size]          - Input activations
    w_q,      # [num_heads*head_dim, hidden_size] - Query weights
    w_k,      # [num_kv_heads*head_dim, hidden_size] - Key weights
    w_v,      # [num_kv_heads*head_dim, hidden_size] - Value weights
    num_heads,
    num_kv_heads,
    head_dim
)
# Returns:
#   q: [seq_len, num_heads, head_dim]     - Query tensor
#   k: [seq_len, num_kv_heads, head_dim]  - Key tensor (GQA: fewer heads)
#   v: [seq_len, num_kv_heads, head_dim]  - Value tensor (GQA: fewer heads)

# Attention computation
out = attention_forward(
    q,           # [seq_q, num_heads, head_dim] or [num_heads, head_dim] for decode
    k_cache,     # [cache_len, num_kv_heads, head_dim] - Cached keys
    v_cache,     # [cache_len, num_kv_heads, head_dim] - Cached values
    head_dim,
    num_kv_heads,
    num_heads,
    attn_scale,  # 1/sqrt(head_dim) - precomputed
    mask=None    # Optional causal mask [seq_q, seq_k]
)
# Returns:
#   out: [seq_q, num_heads, head_dim] - Attention output
```

### The Math

**Step 1: Q/K/V Linear Projections**

```
Q = x @ W_q.T  # [seq_len, num_heads * head_dim]
K = x @ W_k.T  # [seq_len, num_kv_heads * head_dim]
V = x @ W_v.T  # [seq_len, num_kv_heads * head_dim]

# Reshape to separate heads
Q: [seq_len, num_heads, head_dim]
K: [seq_len, num_kv_heads, head_dim]
V: [seq_len, num_kv_heads, head_dim]
```

**Step 2: GQA Tiling (for K and V)**

With GQA, each KV head is shared by `group_size = num_heads / num_kv_heads` query heads:

```
K_tiled = repeat(K, group_size, axis=head_axis)  # [seq_len, num_heads, head_dim]
V_tiled = repeat(V, group_size, axis=head_axis)  # [seq_len, num_heads, head_dim]
```

**Step 3: Attention Scores**

```
scores = Q @ K_tiled.T / sqrt(head_dim)  # [seq_q, num_heads, seq_k]
```

The division by `sqrt(head_dim)` prevents dot products from growing too large with dimension (which would saturate softmax).

**Step 4: Causal Mask (for prefill)**

```
# Create lower-triangular mask
mask[i, j] = 0       if j <= i  (allowed)
mask[i, j] = -inf    if j > i   (forbidden)

scores = scores + mask
```

**Step 5: Softmax**

```
weights = softmax(scores, axis=-1)  # [seq_q, num_heads, seq_k]
# Each row sums to 1
```

**Step 6: Weighted Sum**

```
out = weights @ V_tiled  # [seq_q, num_heads, head_dim]
```

### The KV Cache

During **prefill** (first forward pass):
- Compute K, V for all prompt tokens
- Store in cache: `K_cache[:seq_len] = K`, `V_cache[:seq_len] = V`

During **decode** (subsequent tokens):
- Compute K, V for just the new token
- Append to cache: `K_cache[cache_len] = K_new`
- Retrieve all cached K, V for attention

Cache layout: `[num_layers, max_seq_len, num_kv_heads, head_dim]`

## How to implement it

### Step 1: Implement softmax

Create numerically stable softmax:

```python
def softmax(x, axis=-1):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### Step 2: Implement QKV projection

Matrix multiply then reshape:

```python
def compute_qkv(x, w_q, w_k, w_v, num_heads, num_kv_heads, head_dim):
    seq_len = x.shape[0]
    q = (x @ w_q.T).reshape(seq_len, num_heads, head_dim)
    k = (x @ w_k.T).reshape(seq_len, num_kv_heads, head_dim)
    v = (x @ w_v.T).reshape(seq_len, num_kv_heads, head_dim)
    return q, k, v
```

### Step 3: Implement causal mask

Use `np.triu` for vectorized creation:

```python
def causal_mask(seq_len_q, seq_len_k):
    return np.triu(np.ones((seq_len_q, seq_len_k)), k=1) * float("-inf")
```

### Step 4: Implement attention forward

```python
def attention_forward(q, k_cache, v_cache, head_dim, num_kv_heads, num_heads, 
                      attn_scale, mask=None):
    # GQA: Tile K, V to match Q head count
    group_size = num_heads // num_kv_heads
    k = np.repeat(k_cache, group_size, axis=1)
    v = np.repeat(v_cache, group_size, axis=1)
    
    # Handle decode case (q without batch dim)
    if q.ndim == 2:
        q = q[np.newaxis, ...]
    
    # Compute scores
    scores = np.einsum("qhd,khd->qkh", q, k) * attn_scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask
    
    # Softmax and weighted sum
    weights = softmax(scores, axis=-1)
    out = np.einsum("qkh,khd->qhd", weights, v)
    
    return out
```

### Step 5: Implement KV cache

Create class with sequential memory layout:

```python
class KVCache:
    def __init__(self, max_seq_len, num_layers, num_kv_heads, head_dim):
        # [layers, seq, heads, dim] for cache-friendly access
        self.k = np.zeros((num_layers, max_seq_len, num_kv_heads, head_dim))
        self.v = np.zeros((num_layers, max_seq_len, num_kv_heads, head_dim))
        self.length = 0
        self.current_layer = 0
    
    def append(self, k, v):
        # k, v: [seq_len, num_kv_heads, head_dim]
        seq_len = k.shape[0]
        self.k[self.current_layer, self.length:self.length+seq_len] = k
        self.v[self.current_layer, self.length:self.length+seq_len] = v
        self.length += seq_len
    
    def __getitem__(self, layer):
        return self.k[layer, :self.length], self.v[layer, :self.length]
```

### Step 6: Integrate into model layer

```python
class ModelLayer:
    def __init__(self, weights, layer_idx, num_heads, num_kv_heads, 
                 head_dim, hidden_size):
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.w_q = weights[f"{prefix}.q_proj.weight"]
        self.w_k = weights[f"{prefix}.k_proj.weight"]
        self.w_v = weights[f"{prefix}.v_proj.weight"]
        self.w_o = weights[f"{prefix}.o_proj.weight"]
        
        self.attn_scale = 1.0 / np.sqrt(head_dim)
        # ... store other params
    
    def forward(self, x, positions, kv_cache, cos_table, sin_table, 
                is_prefill=True):
        # 1. QKV projection
        q, k, v = compute_qkv(x, self.w_q, self.w_k, self.w_v, ...)
        
        # 2. Apply RoPE (F3)
        q, k = apply_rope(q, k, positions, cos_table, sin_table)
        
        # 3. Update cache
        if kv_cache is not None:
            kv_cache.append(k, v)
            k_cached, v_cached = kv_cache[self.layer_idx]
        else:
            k_cached, v_cached = k, v
        
        # 4. Causal mask for prefill
        mask = None
        if is_prefill and seq_len > 1:
            mask = causal_mask(seq_len, k_cached.shape[0])
        
        # 5. Attention
        attn_out = attention_forward(q, k_cached, v_cached, ...)
        
        # 6. Output projection
        attn_out = attn_out.reshape(seq_len, -1)
        out = attn_out @ self.w_o.T
        
        return out
```

## Edge cases and gotchas

### Causal mask during decode

During **decode** (single token), you don't need a causal mask because the query is at the "latest" position and can attend to all cached keys. Only **prefill** needs the mask.

### GQA head count mismatch

If `num_heads % num_kv_heads != 0`, the GQA tiling breaks. Always verify: `assert num_heads % num_kv_heads == 0`.

### KV cache bounds

If you exceed `max_seq_len`, you'll get an IndexError. Add a check or use the scheduler's stop conditions (F12) to prevent this.

### Dtype consistency

Weights from safetensors may be float16 or bfloat16. Ensure your cache and computation use consistent dtypes, or explicitly cast.

### Attention scale precomputation

Computing `1/sqrt(head_dim)` every call wastes cycles. Precompute in `__init__` and store as `self.attn_scale`.

### np.einsum vs matmul

`einsum` is convenient but slightly slower than explicit matmul for simple cases. For v0, the clarity is worth it. For v1+, consider optimizing hot paths.

## How to test it

### Unit tests

- **softmax**: Sums to 1, handles large values (numerical stability), handles negatives
- **compute_qkv**: Correct output shapes, works with GQA (different head counts)
- **causal_mask**: Lower triangle is 0, upper is -inf, works with rectangular shapes
- **attention_forward**: Correct shapes for prefill/decode, GQA tiling works, no NaN with mask

### Integration tests

- **Prefill pipeline**: QKV → RoPE → Cache → Attention with mask
- **Decode pipeline**: Single token with cached history, no mask
- **SmolLM2 config**: Test with actual model dimensions

### Observable effects

With attention implemented:
- Model can generate coherent text (not random tokens)
- Prefill is slower (computes QKV for all prompt tokens)
- Decode is faster with cache (reuses cached K/V)
- Memory grows linearly with sequence length (cache size)

## Where it lives in the codebase

- `src/ladallm/attention.py` — `softmax()`, `compute_qkv()`, `causal_mask()`, `attention_forward()`
- `src/ladallm/kvcache.py` — `KVCache` class
- `src/ladallm/model.py` — `ModelLayer.forward()` integrates attention
- `tests/test_attention.py` — Comprehensive test suite

## Further reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original transformer paper
- [GQA Paper](https://arxiv.org/abs/2305.13245) — Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models"
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., for v1+ optimization ideas
- [vLLM Paper](https://arxiv.org/abs/2309.06180) — KV cache management and paged attention
