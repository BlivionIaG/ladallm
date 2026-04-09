# v0 — Build It Yourself

> Prerequisite reading: [`../architecture_v0.md`](../architecture_v0.md). Read it once before starting; keep it open in another tab as you work.
>
> Goal: by the end of this tutorial you will have, on your own machine, a Python + NumPy program that loads a small Llama-style model from a real checkpoint and generates text. No PyTorch, no Transformers library, no CUDA — just NumPy and the math from the architecture doc.
>
> Style: this is an **exercise**. Each step gives you (a) what you are about to build, (b) a hint at the shape of the code, (c) a checkpoint you must pass before moving on. Code snippets are deliberately partial — the blanks are yours to fill in.

---

## 0. Setup

### 0.1 The model you will load

We need a real checkpoint that is **small enough to run on CPU in NumPy**. Use [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0). It is a Llama-2 architecture, 22 layers, `d_model = 2048`, 32 query heads, 4 KV heads (so GQA group size 8), `head_dim = 64`, vocab 32000.

If 1.1B is too slow on your machine, you can substitute any other Llama-architecture model — you will not have to change a line of code, only the path.

Download with whichever tool you prefer (`huggingface-cli download`, `git lfs`, or the `huggingface_hub` Python package). You need:

- `model.safetensors` (the weights)
- `tokenizer.json` and `tokenizer_config.json` (the tokenizer)
- `config.json` (architectural hyperparameters)

Put them somewhere stable like `~/models/tinyllama/`.

### 0.2 Environment

```
python -m venv .venv
source .venv/bin/activate
pip install numpy safetensors tokenizers
```

That is the entire dependency list. Note what is **not** here: `torch`, `transformers`, `accelerate`. We are deliberately doing this with no neural-network framework. You will write every matmul.

### 0.3 Project skeleton

Create the directory layout from `architecture_v0.md` §9:

```
vllm-from-scratch/
├── vllm_fs/
│   ├── __init__.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── kv_cache.py
│   ├── sampler.py
│   ├── model_runner.py
│   ├── scheduler.py
│   ├── engine.py
│   └── models/
│       ├── __init__.py
│       └── tiny_llama.py
├── examples/
│   └── run_v0.py
└── tests/
    └── test_v0_smoke.py
```

`touch` everything. Empty files are fine.

**Checkpoint 0.** `python -c "import numpy, safetensors, tokenizers; print('ok')"` prints `ok`.

---

## 1. Config — read `config.json` into a typed object

You are about to write a *lot* of shape-sensitive code. The sooner the architectural hyperparameters live in a single, named place, the fewer "wait, was that 2048 or 4096?" bugs you will have.

Open `vllm_fs/config.py`. Define a `ModelConfig` dataclass with at least:

- `vocab_size: int`
- `hidden_size: int`        — this is `d_model`
- `intermediate_size: int`  — this is `d_ff` for the MLP
- `num_hidden_layers: int`  — N decoder blocks
- `num_attention_heads: int` — query heads, `H_q`
- `num_key_value_heads: int` — KV heads, `H_kv`. Must divide `num_attention_heads`
- `head_dim: int`            — derived: `hidden_size // num_attention_heads`
- `max_position_embeddings: int`
- `rms_norm_eps: float`
- `rope_theta: float`        — the `base` from §4.9, default 10000.0
- `tie_word_embeddings: bool`

Add a classmethod `ModelConfig.from_json(path)` that reads `config.json` and constructs an instance. Be tolerant: not every Llama config sets every field with the same name (`num_key_value_heads` may be missing → fall back to `num_attention_heads`).

**Checkpoint 1.** In a Python REPL:

```python
from vllm_fs.config import ModelConfig
cfg = ModelConfig.from_json("/path/to/tinyllama/config.json")
print(cfg)
# group size for GQA — must be a clean integer
assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
print("group size:", cfg.num_attention_heads // cfg.num_key_value_heads)
```

For TinyLlama you should see `group size: 8`.

---

## 2. Weight loading — F1

You need to read `model.safetensors` and end up with a Python dict mapping HuggingFace's tensor names to NumPy arrays.

Open `vllm_fs/models/tiny_llama.py`. Write a function `load_weights(path: str) -> dict[str, np.ndarray]` using `safetensors.numpy.load_file`. That gives you a dict already; cast every value to `np.float32` so the rest of the engine has a single dtype to reason about. (TinyLlama ships in fp16 or bf16 — fp32 is slower but you cannot afford precision bugs while you are still learning the math.)

Print the keys you got. You should see things like:

```
model.embed_tokens.weight                      [vocab_size, hidden_size]
model.layers.0.input_layernorm.weight          [hidden_size]
model.layers.0.self_attn.q_proj.weight         [hidden_size, hidden_size]
model.layers.0.self_attn.k_proj.weight         [n_kv_heads * head_dim, hidden_size]
model.layers.0.self_attn.v_proj.weight         [n_kv_heads * head_dim, hidden_size]
model.layers.0.self_attn.o_proj.weight         [hidden_size, hidden_size]
model.layers.0.post_attention_layernorm.weight [hidden_size]
model.layers.0.mlp.gate_proj.weight            [intermediate_size, hidden_size]
model.layers.0.mlp.up_proj.weight              [intermediate_size, hidden_size]
model.layers.0.mlp.down_proj.weight            [hidden_size, intermediate_size]
model.norm.weight                              [hidden_size]
lm_head.weight                                 [vocab_size, hidden_size]   (or absent if tied)
```

> ⚠ **Gotcha — HuggingFace stores linear weights transposed.** If `q_proj.weight` has shape `[out_features, in_features]`, then to compute `y = x @ W` you need `y = x @ q_proj.weight.T`. We will not silently transpose at load time — instead we will *always* write `x @ W.T` in the model code so the shapes match what is on disk. Pick the convention now and never break it.

**Checkpoint 2.** Load the weights and assert the shapes match `cfg`. Pick layer 0 and check:

```python
W = load_weights("/path/to/tinyllama/model.safetensors")
assert W["model.embed_tokens.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
assert W["model.layers.0.self_attn.q_proj.weight"].shape == (
    cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size
)
assert W["model.layers.0.self_attn.k_proj.weight"].shape == (
    cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size
)
print("ok")
```

If you tripped on the GQA shapes, re-read §4.5 and §4.8.

---

## 3. Tokenizer — F13

We use the `tokenizers` Rust crate (via its Python bindings). It is faster than a pure-Python BPE and is what HuggingFace itself uses under the hood, so we get exact-match tokenization with TinyLlama for free.

In `vllm_fs/tokenizer.py`:

```python
from tokenizers import Tokenizer

class TinyTokenizer:
    def __init__(self, path: str):
        self.tk = Tokenizer.from_file(path)
        # find the EOS token id from tokenizer_config.json or just hardcode for now
        self.eos_id = ...

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids)
```

Why a wrapper instead of using `Tokenizer` directly? Because the engine should not know which tokenizer library you used. In v1 we may swap implementations and the engine code should not care.

**Checkpoint 3.**

```python
tk = TinyTokenizer("/path/to/tinyllama/tokenizer.json")
ids = tk.encode("Hello, world!")
print(ids, "->", tk.decode(ids))
```

Round-trip should give back something semantically equivalent (whitespace handling may differ slightly with BPE; that is fine).

---

## 4. RMSNorm — F2

The first piece of model math. It is five lines, but writing it deliberately *now* sets the convention you will use for every other op: every function in `tiny_llama.py` is a plain function over NumPy arrays with explicit shapes in a docstring or comment.

In `vllm_fs/models/tiny_llama.py`:

```python
def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    """
    x: [..., d_model]   - the residual stream for one or more tokens
    g: [d_model]        - the learned scale
    returns: same shape as x
    """
    # 1. mean of squares along the last axis (keepdims=True so it broadcasts)
    # 2. divide x by sqrt(that + eps)
    # 3. multiply by g
    ...
```

Compute in fp32 even if `x` is fp16, then cast back. Numerical stability matters here.

**Checkpoint 4.** Sanity-test against a hand computation:

```python
x = np.array([[3.0, 4.0]])         # rms = 3.535
g = np.array([1.0, 1.0])
out = rmsnorm(x, g, 1e-6)
np.testing.assert_allclose(out, x / np.sqrt(np.mean(x*x)), atol=1e-5)
```

---

## 5. RoPE — F3

This is the first place the math gets tricky. Re-read §4.9 of the architecture doc before you start. The goal is two functions:

1. `precompute_rope(head_dim, max_seq_len, base) -> (cos, sin)`, both shaped `[max_seq_len, head_dim/2]`.
2. `apply_rope(x, cos, sin, positions)` where `x` is `[seq_len, n_heads, head_dim]` and `positions` is `[seq_len]` (the absolute position of each token — important during decode where the new token's position is `length`, not `0`).

> ⚠ **Subtle layout issue — half-rotation vs interleaved.** There are two conventions for "pair up the channels":
>
> - **Interleaved**: pair `(x[0], x[1]), (x[2], x[3]), ...`. This is what the original RoFormer paper describes.
> - **Half-rotation** (also called "GPT-NeoX style"): split into halves and pair `(x[i], x[i + head_dim/2])`. This is what HuggingFace's Llama uses.
>
> **TinyLlama uses the half-rotation convention.** If you use the wrong one your model will produce gibberish but won't crash. This is the single most common bug in from-scratch transformer code. When in doubt, look at HF's `modeling_llama.py:apply_rotary_pos_emb` to confirm.

Sketch:

```python
def precompute_rope(head_dim, max_seq_len, base=10000.0):
    # inv_freq[i] = 1 / base ** (2i / head_dim) for i in [0, head_dim/2)
    # angles[m, i] = m * inv_freq[i]
    # return cos(angles), sin(angles), each [max_seq_len, head_dim/2]
    ...

def apply_rope(x, cos, sin, positions):
    """
    x:         [seq_len, n_heads, head_dim]
    cos, sin:  [max_seq_len, head_dim/2]
    positions: [seq_len]                 -- absolute token positions
    """
    # gather the rows of cos/sin corresponding to `positions` -> [seq_len, head_dim/2]
    # split x into x1 = x[..., :head_dim/2], x2 = x[..., head_dim/2:]
    # rotated = concat(x1*cos - x2*sin, x1*sin + x2*cos, axis=-1)
    # broadcast cos/sin over the heads axis
    ...
```

**Checkpoint 5.** Apply RoPE at position 0 — the rotation angle is 0, so the output should equal the input exactly.

```python
cos, sin = precompute_rope(64, 2048)
x = np.random.randn(1, 4, 64).astype(np.float32)
out = apply_rope(x, cos, sin, np.array([0]))
np.testing.assert_allclose(out, x, atol=1e-6)
```

Then check that two different positions give different outputs:

```python
out0 = apply_rope(x, cos, sin, np.array([0]))
out5 = apply_rope(x, cos, sin, np.array([5]))
assert not np.allclose(out0, out5)
```

---

## 6. Naive KV cache — F7

Now we build the data structure that everything else hangs off. From §4.14: per request, per layer, we hold a contiguous K and V buffer sized to `max_seq_len`, plus a length counter.

In `vllm_fs/kv_cache.py`:

```python
class KVCache:
    """
    Naive contiguous KV cache for ONE request.
    Per layer: K, V of shape [max_seq_len, n_kv_heads, head_dim].
    """
    def __init__(self, num_layers, max_seq_len, n_kv_heads, head_dim, dtype=np.float32):
        self.k = np.zeros((num_layers, max_seq_len, n_kv_heads, head_dim), dtype=dtype)
        self.v = np.zeros((num_layers, max_seq_len, n_kv_heads, head_dim), dtype=dtype)
        self.length = 0  # number of valid positions in [0, length)

    def append(self, layer_idx, k_new, v_new):
        """
        k_new, v_new: [n_new_tokens, n_kv_heads, head_dim]
        Writes them at positions [length, length + n_new_tokens).
        Note: caller advances `self.length` exactly ONCE per step,
        AFTER all layers have been written for that step.
        """
        ...

    def get(self, layer_idx):
        """Returns K[:length], V[:length] for this layer."""
        ...

    def advance(self, n):
        self.length += n
```

> ⚠ Why advance once *after* all layers, not inside `append`? Because every layer in the same forward pass writes to the same set of positions. If you advanced inside `append`, layer 1 would write to position `length+1`, layer 2 to `length+2`, etc. Always think about cache state at the *step* level, not the layer level.

**Checkpoint 6.** Build a cache, write some bogus data into 3 positions across all layers, read it back, confirm shapes:

```python
cache = KVCache(num_layers=2, max_seq_len=16, n_kv_heads=4, head_dim=8)
for layer in range(2):
    k = np.ones((3, 4, 8)) * (layer + 1)
    v = np.ones((3, 4, 8)) * (layer + 10)
    cache.append(layer, k, v)
cache.advance(3)
assert cache.length == 3
K, V = cache.get(0)
assert K.shape == (3, 4, 8) and (K == 1).all()
K, V = cache.get(1)
assert (K == 2).all() and (V == 11).all()
```

---

## 7. Attention — F4

This is the big one. Re-read §4.5–4.8 first. We are going to write **one** attention function that handles both prefill (`seq_len > 1`) and decode (`seq_len == 1`) — the only difference between them is the shape of the input and whether the causal mask matters.

Sketch in `tiny_llama.py`:

```python
def attention(
    x,                  # [seq_len, hidden_size]   -- input residual stream slice
    cache,              # KVCache for this request
    layer_idx,
    cfg,                # ModelConfig
    weights,            # dict of fp32 arrays
    cos, sin,           # RoPE tables
    positions,          # [seq_len]    absolute positions
):
    H_q, H_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.head_dim
    group = H_q // H_kv
    L = x.shape[0]

    # 1. Q, K, V projection. Remember the .T for HF's transposed storage.
    Wq = weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
    Wk = weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
    Wv = weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
    Wo = weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]

    Q = (x @ Wq.T).reshape(L, H_q,  D)
    K = (x @ Wk.T).reshape(L, H_kv, D)
    V = (x @ Wv.T).reshape(L, H_kv, D)

    # 2. RoPE on Q and K (NOT on V)
    Q = apply_rope(Q, cos, sin, positions)
    K = apply_rope(K, cos, sin, positions)

    # 3. Append the new K, V to the cache
    cache.append(layer_idx, K, V)
    K_all, V_all = cache.get(layer_idx)         # shapes [length, H_kv, D] AFTER append+advance...
    # ⚠ but we have not advanced yet. Either: advance at the end of the FULL forward pass,
    # OR write a `peek` that includes the just-appended tokens. Pick one and be consistent.

    # 4. GQA: tile K_all, V_all from H_kv to H_q heads.
    K_tiled = np.repeat(K_all, group, axis=1)    # [L_total, H_q, D]
    V_tiled = np.repeat(V_all, group, axis=1)

    # 5. Per-head scaled dot-product attention.
    #    For numerical clarity, transpose to [H_q, L, D] and [H_q, L_total, D] first.
    Qh   = Q.transpose(1, 0, 2)                  # [H_q, L, D]
    Kh   = K_tiled.transpose(1, 0, 2)            # [H_q, L_total, D]
    Vh   = V_tiled.transpose(1, 0, 2)            # [H_q, L_total, D]

    scores = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(D)   # [H_q, L, L_total]

    # 6. Causal mask. For each query position i (absolute = positions[i]), mask out
    #    cache positions j > positions[i].
    #    During decode (L=1) this is a no-op because the new query is the latest position.
    mask = build_causal_mask(positions, L_total=K_all.shape[0])   # [L, L_total]
    scores = scores + mask                                         # broadcasts over H_q

    # 7. softmax in fp32, then weighted sum of V
    weights_attn = stable_softmax(scores, axis=-1)
    out = weights_attn @ Vh                                        # [H_q, L, D]

    # 8. Concat heads -> [L, H_q*D] = [L, hidden_size], then output projection
    out = out.transpose(1, 0, 2).reshape(L, H_q * D)
    return out @ Wo.T                                              # [L, hidden_size]
```

You will need helper functions `build_causal_mask` and `stable_softmax`. Write them; they are 5 lines each. The softmax must subtract the row max before exponentiating, otherwise long sequences overflow.

**The `append` vs `advance` decision.** Two valid designs:

- **(A) Append-then-read.** `append` writes into the cache buffer immediately. The model code reads `cache.k[layer_idx, :length + L]` to include the just-appended rows. `advance(L)` is called once at the end of the full forward pass, after all layers.
- **(B) Read-then-append.** Same idea but the model code stitches the new K/V onto what `cache.get(layer_idx)` returns: `K_all = concat([cache.k[layer_idx, :length], K], axis=0)`. Then `advance` and write happens at the end.

Both work. **(A)** is closer to how a real GPU paged-attention kernel will look in v1; pick (A) and be very deliberate about when you call `advance`. Add an assertion at the top of the engine step that `cache.length` matches what the model expects.

**Checkpoint 7.** No clean unit test until the full forward works — but you can sanity-check that the shapes flow:

```python
cache = KVCache(cfg.num_hidden_layers, 128, cfg.num_key_value_heads, cfg.head_dim)
x = np.random.randn(5, cfg.hidden_size).astype(np.float32)   # 5-token "prompt"
cos, sin = precompute_rope(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
out = attention(x, cache, layer_idx=0, cfg=cfg, weights=W,
                cos=cos, sin=sin, positions=np.arange(5))
assert out.shape == (5, cfg.hidden_size)
print("attention shape ok")
```

---

## 8. SwiGLU MLP — F5

The simplest piece since the math.

```python
def silu(z):
    return z / (1.0 + np.exp(-z))   # equivalently z * sigmoid(z)

def mlp(x, layer_idx, weights):
    Wg = weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
    Wu = weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
    Wd = weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
    gate = x @ Wg.T
    up   = x @ Wu.T
    return (silu(gate) * up) @ Wd.T
```

**Checkpoint 8.** Shape check:

```python
x = np.random.randn(5, cfg.hidden_size).astype(np.float32)
out = mlp(x, layer_idx=0, weights=W)
assert out.shape == (5, cfg.hidden_size)
```

---

## 9. The decoder block and the full forward — F6

Now wire it together. One block:

```python
def decoder_block(x, cache, layer_idx, cfg, weights, cos, sin, positions):
    g_in   = weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
    g_post = weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]

    # Pre-norm + attention + residual
    h = rmsnorm(x, g_in, cfg.rms_norm_eps)
    h = attention(h, cache, layer_idx, cfg, weights, cos, sin, positions)
    x = x + h

    # Pre-norm + MLP + residual
    h = rmsnorm(x, g_post, cfg.rms_norm_eps)
    h = mlp(h, layer_idx, weights)
    x = x + h
    return x
```

The full forward:

```python
def forward(input_ids, cache, cfg, weights, cos, sin, positions):
    """
    input_ids: [seq_len]
    positions: [seq_len]   absolute positions of each input id
    returns logits: [seq_len, vocab_size]
    """
    E = weights["model.embed_tokens.weight"]
    x = E[input_ids]                                # [seq_len, hidden_size]
    for layer in range(cfg.num_hidden_layers):
        x = decoder_block(x, cache, layer, cfg, weights, cos, sin, positions)
    g_final = weights["model.norm.weight"]
    x = rmsnorm(x, g_final, cfg.rms_norm_eps)

    if cfg.tie_word_embeddings:
        W_lm = E
    else:
        W_lm = weights["lm_head.weight"]
    logits = x @ W_lm.T                             # [seq_len, vocab_size]

    cache.advance(len(input_ids))                   # ⚠ exactly once, after all layers
    return logits
```

**Checkpoint 9 — the moment of truth.** Run a single prefill on a real prompt and look at the top-1 token:

```python
cfg = ModelConfig.from_json("/path/to/tinyllama/config.json")
W   = load_weights("/path/to/tinyllama/model.safetensors")
tk  = TinyTokenizer("/path/to/tinyllama/tokenizer.json")

cos, sin = precompute_rope(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
cache = KVCache(cfg.num_hidden_layers, 128, cfg.num_key_value_heads, cfg.head_dim)

ids = tk.encode("The capital of France is")
positions = np.arange(len(ids))
logits = forward(np.array(ids), cache, cfg, W, cos, sin, positions)
next_id = int(np.argmax(logits[-1]))
print(tk.decode(ids + [next_id]))
```

You should see something like `"The capital of France is Paris"`. If you don't:

1. **Garbage that vaguely looks like English** → almost certainly RoPE (wrong rotation convention, see §5 warning).
2. **Garbage that looks like another language** → probably the `.T` convention slipped on one of the projections.
3. **Crash on a shape mismatch** → most likely GQA group tiling or `o_proj` shape.
4. **Same wrong token every time** → check that `forward` is reading the *last row* of logits, not the first.
5. **NaN everywhere** → softmax overflow; make sure you subtract the max before `exp`.

This is the single most rewarding checkpoint of the whole tutorial. **Do not move on until it works.** Every later piece assumes a correct forward.

---

## 10. Greedy sampler — F9

After step 9 you already inlined the sampler. Now make it a proper module so the engine can swap it out later.

In `vllm_fs/sampler.py`:

```python
def greedy(logits_row: np.ndarray) -> int:
    """logits_row: [vocab_size] — the LAST row of the logits matrix."""
    return int(np.argmax(logits_row))
```

That is the entire file in v0. We will get fancier in v3 (speculative decoding cares a lot about the sampler).

---

## 11. Model runner: prefill vs decode — F8

The runner is the thin layer that knows *which kind of forward call* to make for a given request state. From §4.15: prefill takes the whole prompt at positions `[0, P)`; decode takes one token at position `length`.

In `vllm_fs/model_runner.py`:

```python
class ModelRunner:
    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.weights = weights
        self.cos, self.sin = precompute_rope(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    def prefill(self, request) -> int:
        """Run the whole prompt through the model. Returns the first sampled token id."""
        ids = np.array(request.prompt_ids)
        positions = np.arange(len(ids))
        logits = forward(ids, request.cache, self.cfg, self.weights, self.cos, self.sin, positions)
        return greedy(logits[-1])

    def decode(self, request) -> int:
        """Run a single token. Returns the next sampled token id."""
        last = request.generated_ids[-1] if request.generated_ids else request.prompt_ids[-1]
        ids = np.array([last])
        positions = np.array([request.cache.length])      # absolute position of the new token
        logits = forward(ids, request.cache, self.cfg, self.weights, self.cos, self.sin, positions)
        return greedy(logits[-1])
```

> ⚠ Notice: in `decode` the input is the *previously sampled* token (the one decode `t-1` produced). The prefill output token is also "previously sampled" — that is what the engine will append before the first decode call.

**Checkpoint 11.** Wire a quick smoke test in a REPL: run `prefill` then `decode` three times and print the running detokenized string. Should look like English continuing the prompt.

---

## 12. Scheduler and Request — F10, F12

In v0 the scheduler is a queue of one. The point is not the sophistication of the policy — it is to set up the *state machine* (architecture doc §4.16) so that v1 can drop in continuous batching without rewriting `engine.py`.

In `vllm_fs/scheduler.py`:

```python
from dataclasses import dataclass, field
from enum import Enum

class Status(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE  = "decode"
    DONE    = "done"

@dataclass
class Request:
    prompt_ids: list[int]
    max_tokens: int
    eos_id: int
    cache: "KVCache"
    generated_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING

    def is_done(self) -> bool:
        if len(self.generated_ids) >= self.max_tokens:
            return True
        if self.generated_ids and self.generated_ids[-1] == self.eos_id:
            return True
        if self.cache.length >= self.cache.k.shape[1]:
            return True   # cache full
        return False

class Scheduler:
    def __init__(self):
        self.queue: list[Request] = []

    def admit(self, req: Request):
        self.queue.append(req)

    def pick(self) -> Request | None:
        for r in self.queue:
            if r.status != Status.DONE:
                return r
        return None

    def remove_done(self):
        self.queue = [r for r in self.queue if r.status != Status.DONE]
```

Note that the scheduler has no model knowledge whatsoever. It moves states. The `engine` and `model_runner` do the work.

---

## 13. The engine top-level loop — F11

The orchestrator. In `vllm_fs/engine.py`:

```python
class Engine:
    def __init__(self, model_dir: str, max_seq_len: int = 2048):
        self.cfg = ModelConfig.from_json(f"{model_dir}/config.json")
        self.weights = load_weights(f"{model_dir}/model.safetensors")
        self.tokenizer = TinyTokenizer(f"{model_dir}/tokenizer.json")
        self.runner = ModelRunner(self.cfg, self.weights)
        self.scheduler = Scheduler()
        self.max_seq_len = max_seq_len

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        ids = self.tokenizer.encode(prompt)
        cache = KVCache(self.cfg.num_hidden_layers, self.max_seq_len,
                        self.cfg.num_key_value_heads, self.cfg.head_dim)
        req = Request(prompt_ids=ids, max_tokens=max_tokens,
                      eos_id=self.tokenizer.eos_id, cache=cache)
        self.scheduler.admit(req)

        while True:
            r = self.scheduler.pick()
            if r is None:
                break
            self.step(r)

        self.scheduler.remove_done()
        return self.tokenizer.decode(req.prompt_ids + req.generated_ids)

    def step(self, r: Request):
        if r.status == Status.WAITING:
            r.status = Status.PREFILL
            tok = self.runner.prefill(r)
            r.generated_ids.append(tok)
            r.status = Status.DECODE
        elif r.status == Status.DECODE:
            tok = self.runner.decode(r)
            r.generated_ids.append(tok)
        if r.is_done():
            r.status = Status.DONE
```

Read this top-down a couple of times. The `step()` method is the thing v1 will completely rewrite to handle a *batch* of requests instead of one — but the *abstraction* (status field, runner-knows-how-to-prefill-and-decode, scheduler-picks-who-runs) stays put.

---

## 14. End-to-end script — F14

`examples/run_v0.py`:

```python
from vllm_fs.engine import Engine

if __name__ == "__main__":
    eng = Engine("/path/to/tinyllama")
    out = eng.generate("The quick brown fox", max_tokens=32)
    print(out)
```

**Checkpoint 14 — v0 is "done".** Run it. You should see ~32 tokens of plausible continuation. It will be slow (NumPy on CPU, no batching, fp32 weights). Time it. Count tokens per second. **Write that number down.** Every later version's success is measured against this baseline.

A sample expectation on a modern laptop CPU with TinyLlama: somewhere between 0.5 and 5 tokens/second. If you get 0.05 tok/s, something is doing extra work — the most common cause is recomputing the prompt every decode step (i.e., the KV cache is silently broken and the model is doing prefill every step). Throw a `print(input_ids.shape)` inside `forward` and watch what happens during a decode loop. It should be `(1,)` after the first call.

---

## 15. Smoke test — F15

`tests/test_v0_smoke.py`:

```python
def test_greedy_is_deterministic():
    eng = Engine("/path/to/tinyllama")
    a = eng.generate("Hello,", max_tokens=8)
    b = eng.generate("Hello,", max_tokens=8)
    assert a == b   # greedy must be byte-identical across runs
```

This is the regression net for *every refactor* you do later. v1 will rewrite half the engine; this test must keep passing.

---

## 16. Reflect

You now have a complete autoregressive inference engine in ~600 lines of Python and NumPy. Before moving on to v1, take five minutes and answer these questions in your own words (or in a notebook). The answers are *the point of v0*:

1. Where in the code is the per-step quadratic cost of the naive (no-cache) approach actually saved? Point at a specific line.
2. If you took the model from §9 and serviced 50 concurrent users with 50 separate `KVCache` instances of size `max_seq_len = 2048`, how much memory would just the KV cache consume? (Compute it from `num_hidden_layers * 2 * max_seq_len * n_kv_heads * head_dim * 4 bytes` per request.) Is that "fits on a consumer GPU" or not? This is the pain v1's paged attention will relieve.
3. What is the slowest line of code per decode step? (Hint: time the matmuls in `attention` vs the matmuls in `mlp`. Then think about which one will *also* be slow on a GPU and which one will not.)
4. If you wanted to add a second backend (CPU NumPy and GPU OpenCL), which functions would you have to abstract behind an interface? Make a list. Compare it to v1's plan.

When you can answer those without looking at the code, you have *learned v0* — not just *built v0*. That distinction is the entire reason this project exists.

---

## What's next

- **`docs/architecture_v1.md`** — the plan for paged attention, continuous batching, prefix caching, and the OpenCL backend. Read it the same way you read v0: once for shape, once for math.
- **`docs/v1/README.md`** — the per-feature index. Each feature has (or will have) its own implementation guide following [`../_feature_template.md`](../_feature_template.md).
- A `docs/tutorial/v1-build-it-yourself.md` will eventually exist — but it builds *on top of* the code you wrote here. Don't throw away your v0; it is the baseline you will measure against forever.
