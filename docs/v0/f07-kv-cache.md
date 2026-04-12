# F7 — Naive KV Cache

> Version: v0  •  Concept: [4.14 The KV cache](../architecture_v0.md#414-the-kv-cache-the-central-optimization)
> Depends on: F4 (Attention)  •  Depended on by: F8 (Prefill/Decode split), F10 (Scheduler)

## What this feature is

The **NaiveKVCache** is a per-layer storage system for Key (K) and Value (V) tensors computed during transformer inference. Instead of recomputing K and V for every token in the sequence on every forward pass, the cache stores them as they're computed and reuses them for subsequent tokens.

Each transformer layer has its own independent cache containing:
- **K buffer**: Pre-allocated to `[max_seq_len, num_kv_heads, head_dim]`
- **V buffer**: Pre-allocated to `[max_seq_len, num_kv_heads, head_dim]`
- **Length counter**: Tracks how many positions are actually filled

## Why it exists

Without a KV cache, generating token *t* would require running the entire model on tokens 1..*t* from scratch. For a sequence of length *T*, this is O(T²) total computation — quadratic growth that makes long sequences prohibitively expensive.

With the cache:
1. **Prefill**: Process all prompt tokens once, storing their K/V in the cache
2. **Decode**: For each new token, only compute K/V for that token, append to cache, and attend against all cached K/V

Per-step computation drops from O(t·d_model) (full model) to O(t·head_dim) (just attention), where *t* is the current sequence length. This single optimization is the difference between "LLM inference works" and "LLM inference is intractable."

## The concept, refreshed

### Memory Layout

For a single layer with:
- `max_seq_len = 2048` (maximum sequence length)
- `num_kv_heads = 3` (for GQA with 9 query heads)
- `head_dim = 64` (576 hidden_size / 9 heads)

```
K buffer shape: [2048, 3, 64]
V buffer shape: [2048, 3, 64]

Memory per layer: 2 × 2048 × 3 × 64 × 4 bytes = 3.14 MB
Memory for 30 layers: 30 × 3.14 MB = 94.4 MB
```

### Prefill vs Decode Patterns

**Prefill** (prompt processing):
```python
# Prompt has P tokens
k_prefill: [P, num_kv_heads, head_dim]
v_prefill: [P, num_kv_heads, head_dim]

cache.append(k_prefill, v_prefill)
# cache.length is now P
```

**Decode** (token generation):
```python
# Each step processes 1 new token
for i in range(max_new_tokens):
    k_new: [1, num_kv_heads, head_dim]
    v_new: [1, num_kv_heads, head_dim]
    
    cache.append(k_new, v_new)
    # cache.length increments by 1
```

### Why "Naive"?

This implementation pre-allocates to `max_seq_len` regardless of actual usage:
- A 10-token prompt still allocates for 2048 tokens
- No memory sharing between requests with identical prefixes
- No eviction or compression

This is **intentionally wasteful**. v1 will introduce **Paged Attention** (block-based allocation like virtual memory) as the optimized alternative. The naive approach provides a simple, predictable baseline.

## How to implement it

### Step 1: Define the cache class

Create `src/ladallm/kvcache.py` with the `NaiveKVCache` class:

```python
class NaiveKVCache:
    def __init__(self, max_seq_len, num_kv_heads, head_dim, dtype=np.float32):
        # Pre-allocate buffers
        self.k = np.zeros((max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        self.v = np.zeros((max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        self._length = 0
```

**Why this shape?** The outer dimension is `max_seq_len` so we can grow dynamically by updating `_length` without reallocation.

### Step 2: Implement append

```python
def append(self, k_new, v_new):
    """Append K,V at current position."""
    seq_len = k_new.shape[0]
    end_pos = self._length + seq_len
    
    # Write to next available positions
    self.k[self._length:end_pos] = k_new
    self.v[self._length:end_pos] = v_new
    self._length = end_pos
```

**Key points:**
- `k_new` can be multiple tokens (prefill) or single token (decode)
- Write position is always `[length, length + seq_len)`
- No reallocation — just index into pre-allocated buffer

### Step 3: Implement get

```python
def get(self):
    """Return cached K,V up to current length."""
    return self.k[:self._length], self.v[:self._length]
```

**Important**: This returns a **view**, not a copy. The attention computation reads from this view.

### Step 4: Create per-layer caches

```python
def create_layer_caches(max_seq_len, num_layers, num_kv_heads, head_dim):
    """Create one cache per layer."""
    return [
        NaiveKVCache(max_seq_len, num_kv_heads, head_dim)
        for _ in range(num_layers)
    ]
```

Each layer needs independent storage because K/V differ across layers.

### Step 5: Integrate with model

In `LlamaModel.forward()`, pass the list of caches:

```python
def forward(self, input_ids, kv_caches=None, is_prefill=True):
    for layer_idx, layer in enumerate(self.layers):
        layer_cache = kv_caches[layer_idx] if kv_caches else None
        x = layer.forward(x, positions, layer_cache, ...)
```

In `LlamaDecoderBlock.forward()`, use the cache:

```python
def forward(self, x, positions, kv_cache, ...):
    # ... compute q, k, v via RoPE ...
    
    if kv_cache is not None:
        kv_cache.append(k, v)
        k_cached, v_cached = kv_cache.get()
    else:
        k_cached, v_cached = k, v  # No caching (e.g., training)
    
    # Attention uses cached K,V
    attn_out = attention_forward(q, k_cached, v_cached, ...)
```

**Critical fix**: Each layer must have **independent length tracking**. The bug in the original implementation used a shared length counter across layers, causing layers to write to wrong positions.

## Edge cases and gotchas

### 1. Cache Overflow

If generation exceeds `max_seq_len`, raise an error:
```python
if self._length + seq_len > self.max_seq_len:
    raise RuntimeError("Cache overflow")
```

In production (v1+), this triggers eviction or rejection. In v0, fail fast.

### 2. Shape Mismatches

Validate that appending tensors match expected dimensions:
```python
if k_new.shape[1:] != (self.num_kv_heads, self.head_dim):
    raise ValueError("Shape mismatch")
```

### 3. Views vs Copies

`get()` returns a view into the buffer. Modifying returned arrays affects the cache:
```python
k_cached, _ = cache.get()
k_cached[0, 0, 0] = 999  # Modifies cache.k!
```

Treat returned arrays as read-only.

### 4. Empty Cache

Before any `append()`, `get()` returns arrays of shape `(0, num_kv_heads, head_dim)`. Attention must handle this (though typically you never attend with empty cache).

### 5. Dtype Consistency

All operations must preserve dtype. Don't mix float32 and float64 — this causes subtle bugs in NumPy broadcasting.

## How to test it

### Unit Tests

See `tests/test_kvcache.py`:

1. **Basic operations**: Create cache, append K/V, verify get() returns correct data
2. **Prefill pattern**: Append 10 tokens at once, verify all stored
3. **Decode pattern**: Append tokens one at a time, verify incremental growth
4. **Independence**: Multiple caches don't interfere with each other
5. **Validation**: Wrong shapes raise errors, overflow is caught
6. **SmolLM2 config**: Test with actual model dimensions

### Integration Tests

Use with the full model (see `tests/test_model.py`):

```python
caches = create_layer_caches(max_seq_len=2048, num_layers=30, ...)
logits = model.forward(input_ids, kv_caches=caches, is_prefill=True)
```

Verify that:
- Model runs without errors
- Cache lengths increase appropriately
- Outputs are deterministic (same input → same output)

### Observable Effects

With KV cache enabled:
- **Memory usage**: +~90MB for SmolLM2 (30 layers × 3.14 MB)
- **Generation speed**: Constant time per token vs O(T) without cache
- **Prefill cost**: One-time O(P²) for prompt of length P

## Where it lives in the codebase

- `src/ladallm/kvcache.py` — `NaiveKVCache` class and `create_layer_caches()` helper
- `tests/test_kvcache.py` — Unit tests
- `src/ladallm/model.py` — Integration with `LlamaModel` and `LlamaDecoderBlock`

## Further reading

- [vLLM Paper](https://arxiv.org/abs/2309.06180) — Section 3 on PagedAttention (contrast with naive approach)
- [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/) — Detailed FLOPs/memory analysis
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original description of decoder-only architecture

## Status

✓ Implemented: Per-layer NaiveKVCache with independent length tracking
☐ Next: F8 Prefill vs Decode split — use cache in generation loop
