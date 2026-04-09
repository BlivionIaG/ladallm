# F01 — Weight Loading

> Version: v0  •  Concept: [4.2 Tokens and embeddings](../architecture_v0.md#42-tokens-and-embeddings)  
> Depends on: none  •  Depended on by: F2–F6 (all model layers)

## What this feature is

A weight loader that reads HuggingFace `safetensors` files into memory-mapped NumPy arrays. This is the entry point for getting model parameters into the engine — without it, the model has no weights to compute with.

## Why it exists

Every tensor operation (RMSNorm, attention, MLP) needs weight matrices. The weights are stored on disk in `safetensors` format — a simple binary format with a JSON header describing tensor names, shapes, dtypes, and byte offsets. Without a loader, the model cannot function; with an inefficient loader, loading large models becomes painfully slow or memory-hungry.

## The concept, refreshed

**Input:** A `.safetensors` file on disk.

**Output:** A dictionary mapping tensor names (strings like `"model.embed_tokens.weight"`) to NumPy arrays with correct shapes and dtypes.

**File layout:**
```
┌─────────────────┬──────────────────┬──────────┬─────────────────┐
│  Header Length  │      Header      │ Padding  │   Tensor Data   │
│   (8 bytes)     │     (JSON)       │  (align) │   (raw bytes)   │
│   uint64 LE     │   {metadata}     │   0x00   │   float16/etc   │
└─────────────────┴──────────────────┴──────────┴─────────────────┘
```

**Header JSON structure:**
```json
{
  "tensor_name": {
    "dtype": "F16",
    "shape": [49152, 576],
    "data_offsets": [0, 56623104]
  }
}
```

**The transformation:**
1. Read 8 bytes → header length N
2. Read N bytes → JSON header
3. Skip padding to 8-byte boundary
4. For each tensor: slice bytes at `data_offsets`, cast to dtype, reshape

**Invariants:**
- Data section starts at an 8-byte aligned offset
- Tensor data is contiguous in the file
- Dtypes are uppercase strings ("F16", "F32", "BF16", etc.)

## How to implement it

### Step 1: Parse the header

**What:** Read the 8-byte length prefix and the JSON header.

**Why:** The header tells us what tensors exist and where to find them.

**Guidance:** Use `struct.unpack("<Q", ...)` for the length, then `json.loads()` for the header. Calculate `data_start = 8 + header_len + padding` where padding aligns to 8 bytes.

### Step 2: Memory-map the file

**What:** Use `mmap.mmap()` to map the entire file into virtual memory.

**Why:** Zero-copy access — tensor data stays on disk until accessed, shared across processes, no explicit read calls.

**Guidance:** Open with `access=mmap.ACCESS_READ`. Store the mmap object as an instance variable to prevent garbage collection.

### Step 3: Create tensor views

**What:** For each tensor in the header, create a NumPy array that views the mmap'd bytes.

**Why:** This gives us NumPy arrays without copying data from the mmap.

**Guidance:** Use `np.frombuffer(mmap, dtype=dtype, offset=start, count=count).reshape(shape)`. Map safetensors dtype strings to NumPy dtypes:
- `"F16"` → `np.float16`
- `"F32"` → `np.float32`
- `"BF16"` → `ml_dtypes.bfloat16` (requires `ml_dtypes` package)

### Step 4: Resource cleanup

**What:** Implement a `close()` method that unmaps the file.

**Why:** Prevents resource leaks and allows the OS to reclaim memory.

**Guidance:** Call `self._mmap.close()` and set to `None`.

## Edge cases and gotchas

- **Key name mismatch:** The header uses `"data_offsets"`, not `"offset"`. Using the wrong key throws a KeyError.
- **Dtype case:** Header uses uppercase ("F16"), but we normalize to lowercase for the map lookup.
- **BF16 support:** NumPy has no native bfloat16. Use `ml_dtypes` package which adds `ml_dtypes.bfloat16` as a numpy-compatible dtype.
- **Memory mapping lifecycle:** The mmap object must be kept alive as long as any tensor arrays exist. Store it as `self._mmap`, not a local variable.
- **Zero-copy tradeoff:** While mmap is efficient, the first access to a tensor triggers a page fault and disk read. For models larger than RAM, this causes thrashing.

## How to test it

**Unit tests:**
- Create a synthetic safetensors file with known tensors, verify shapes and values match
- Test each supported dtype (F16, F32, BF16, I32, etc.)
- Verify memory mapping works (file can be "read" without loading everything)

**Integration tests:**
- Load SmolLM2-135M weights and verify all expected tensors are present
- Check that embedding weights have shape `[vocab_size, hidden_size]`
- Verify that layer count matches config (30 layers for SmolLM2-135M)

**Observable effects:**
- Loading a 270MB model should take <1 second (just parsing header, not reading data)
- Memory usage should not spike to 2× model size (confirming zero-copy)

## Where it lives in the codebase

- `src/ladallm/safetensors.py` — `Safetensors` class

## Further reading

- [safetensors format specification](https://github.com/huggingface/safetensors#format) — Official spec with byte-level details
- [Python mmap documentation](https://docs.python.org/3/library/mmap.html) — How memory mapping works in Python
- [ml_dtypes package](https://github.com/jax-ml/ml_dtypes) — Bfloat16 and other ML dtypes for NumPy
