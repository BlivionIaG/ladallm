# F3 — RoPE (Rotary Position Embedding)

> Version: v0  •  Concept: [4.9 RoPE](../architecture_v0.md#49-rope-rotary-position-embedding)
> Depends on: none  •  Depended on by: F4 (Attention)

## What this feature is

RoPE (Rotary Position Embedding) injects position information into transformer attention by **rotating** query and key vectors in 2D subspaces. Unlike absolute position embeddings (which add a vector to the input), RoPE applies rotation matrices to Q and K after projection, with rotation angles that depend on the token's position in the sequence.

## Why it exists

Without position information, the transformer is **permutation-invariant**: shuffling tokens produces the same outputs. The model needs to know "this token came before that token" to model language.

The clever insight of RoPE: encode position through rotation such that the **relative distance** between two positions naturally emerges in the dot product. When Q at position `m` and K at position `n` are rotated by their respective angles, the resulting `Q @ K.T` depends only on `(m - n)`, not on absolute positions.

This property makes RoPE **KV-cache friendly**: when you process a new token, you only rotate its Q and K by the new position angle. Previously cached K values don't need re-rotation.

## The concept, refreshed

### Inputs and Outputs

```python
# Precompute (once at model load)
cos_table, sin_table = precompute_rope_tables(
    max_seq_len=2048,    # Maximum positions to support
    head_dim=64,         # Dimensions per attention head
    base=10000.0         # Frequency scaling parameter
)
# Returns: [max_seq_len, head_dim/2] each

# Apply (every forward pass)
q_rot, k_rot = apply_rope(
    q,               # [seq_len, num_heads, head_dim]    - Query tensor
    k,               # [seq_len, num_kv_heads, head_dim] - Key tensor
    positions,       # [seq_len]                         - Absolute positions
    cos_table,       # [max_seq_len, head_dim/2]         - Precomputed cosines
    sin_table        # [max_seq_len, head_dim/2]         - Precomputed sines
)
# Returns: q_rot [seq_len, num_heads, head_dim], k_rot [seq_len, num_kv_heads, head_dim]
```

### The Math

**Step 1: Compute rotation angles**

For each position `m` and dimension pair `i`:

```
θ_{m,i} = m * base^(-2i/head_dim)
        = m / (base^(2i/head_dim))
```

- `m`: Token position (0, 1, 2, ...)
- `i`: Dimension pair index (0 to head_dim/2 - 1)
- `base`: Typically 10000.0

**Step 2: Precompute cos and sin**

```
cos_table[m, i] = cos(θ_{m,i})
sin_table[m, i] = sin(θ_{m,i})
```

**Step 3: Apply rotation during forward pass**

For each pair of dimensions `[x, y]` at position `m`:

```
x' = x * cos(θ_m) - y * sin(θ_m)
y' = x * sin(θ_m) + y * cos(θ_m)
```

In matrix form (per 2D pair):
```
[x']   [cos θ   -sin θ] [x]
[y'] = [sin θ    cos θ] [y]
```

### Key Properties

1. **Relative position encoding**: The dot product `Q_m @ K_n` depends only on `(m - n)`
2. **Rotation preserves norm**: `||[x', y']|| = ||[x, y]||`
3. **KV-cache compatible**: Cached K values never need re-rotation

## How to implement it

### Step 1: Precompute the tables

Create `precompute_rope_tables()` in `src/ladallm/rope.py`:

1. Generate position indices: `np.arange(max_seq_len)`
2. Generate dimension indices: `np.arange(0, head_dim, 2)`
3. Compute inverse frequencies: `1 / (base ** (dims / head_dim))`
4. Compute angles via broadcasting: `positions[:, None] * inv_freq[None, :]`
5. Return `np.cos(angles)`, `np.sin(angles)`

### Step 2: Implement the rotation

Create `apply_rope()` in `src/ladallm/rope.py`:

1. **Lookup**: Index into tables using `positions` → `[seq_len, head_dim/2]`
2. **Broadcast**: Add head dimension → `[seq_len, 1, head_dim/2]`
3. **Extract pairs**: Use striding to get even/odd indices
   - `x = q[..., 0::2]` (dimensions 0, 2, 4, ...)
   - `y = q[..., 1::2]` (dimensions 1, 3, 5, ...)
4. **Apply rotation**:
   - `x_rot = x * cos - y * sin`
   - `y_rot = x * sin + y * cos`
5. **Reconstruct**: Interleave `x_rot` and `y_rot` back into output array

### Step 3: Integrate with model (F4)

In your attention implementation:

```python
def attention_forward(self, x, positions):
    # Project to Q, K, V
    q = x @ W_q.T  # [seq_len, num_heads * head_dim]
    k = x @ W_k.T  # [seq_len, num_kv_heads * head_dim]
    
    # Reshape to separate heads
    q = q.reshape(seq_len, num_heads, head_dim)
    k = k.reshape(seq_len, num_kv_heads, head_dim)
    
    # Apply RoPE
    q, k = apply_rope(q, k, positions, self.cos_table, self.sin_table)
    
    # Continue with attention computation...
```

## Edge cases and gotchas

### Position 0 is the identity

At position 0, all angles are 0, so `cos=1` and `sin=0`. The rotation becomes:
```
x' = x * 1 - y * 0 = x
y' = x * 0 + y * 1 = y
```
Vectors at position 0 are unchanged. **Don't skip the call** — consistency is better than micro-optimization.

### GQA (Grouped Query Attention)

SmolLM2 uses GQA: 9 query heads but only 3 KV heads. The tables don't care about head count — they're indexed by position only. Broadcast the lookup result across all heads (Q and K independently).

### dtype consistency

`np.cos()` and `np.sin()` return `float64` by default. Your Q/K tensors are likely `float32`. NumPy will upcast during multiplication. For v0 this is fine; for v1+ you'll want explicit dtype control.

### Position beyond table size

If you pass `positions >= max_seq_len`, you'll get an `IndexError`. In practice, this means:
- Precompute for `max_position_embeddings` from config (2048 for SmolLM2)
- Add stop condition in scheduler when approaching the limit (F12)

### The "skip V" trap

**Only Q and K get rotated. V does not.** This is a common mistake. The position information flows through `Q @ K.T`; V just provides the values to aggregate. Rotating V would be meaningless.

## How to test it

### Unit tests (`tests/test_rope.py`)

1. **Shape tests**: Verify output shapes match input shapes
2. **Identity at position 0**: Same input → same output at position 0
3. **Rotation changes values**: Non-zero positions produce different outputs
4. **Norm preservation**: Vector lengths preserved within each pair
5. **Pythagorean identity**: `cos² + sin² = 1` for all table entries
6. **GQA handling**: Works with different `num_heads` and `num_kv_heads`
7. **Position bounds**: IndexError when position >= table size
8. **SmolLM2 config**: Test with actual model dimensions (head_dim=64)

### Integration test

```python
# Verify relative position property
q = np.random.randn(1, 1, 64)
k = np.random.randn(1, 1, 64)

# Dot product at relative distance 5 should be similar
# regardless of absolute positions
dot_1 = np.sum(apply_rope(q, k, [10], tables)[0] * 
               apply_rope(q, k, [5], tables)[1])
dot_2 = np.sum(apply_rope(q, k, [20], tables)[0] * 
               apply_rope(q, k, [15], tables)[1])

# Both have relative distance 5
assert np.isclose(dot_1, dot_2, rtol=0.1)  # Approximate due to different mags
```

## Where it lives in the codebase

- `src/ladallm/rope.py` — `precompute_rope_tables()`, `apply_rope()`
- `tests/test_rope.py` — comprehensive test suite

## Further reading

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Original RoPE paper
- [LLaMA paper](https://arxiv.org/abs/2302.13971) — Popularized RoPE in open-source LLMs
- [HuggingFace blog on RoPE](https://huggingface.co/blog/mayong/rotary-embeddings-explained) — Visual explanation
