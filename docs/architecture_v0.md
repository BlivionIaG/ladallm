# Architecture v0 — Minimal Inference Engine (Learning Edition)

This document has two jobs:

1. Define the **minimal architecture** for the first iteration of `vllm-from-scratch`.
2. **Teach you the concepts** that architecture is built from, so by the end of reading it you understand *why* every box in the diagram exists and *what math* runs inside it.

If you only want the architecture, skim sections 1–3 and 7–11. If you want to learn, read section 4 in order — its subsections build on each other.

---

## 1. What we are building

An **inference engine** for an autoregressive decoder-only LLM (Llama-style). Two words to unpack:

- **Autoregressive** — the model generates one token at a time, and each new token is appended to the input for the next step. There is no global plan, just a tight "predict next token, append, repeat" loop.
- **Inference** — we are not training. The weights are fixed; we only run the forward pass.

A complete engine answers four questions:

1. **How do I turn text into numbers and back?** → tokenizer
2. **Given numbers, how do I compute the next token?** → model forward pass
3. **How do I avoid redoing work I already did?** → KV cache
4. **How do I serve work efficiently?** → scheduler + runtime loop

v0 answers all four in the simplest possible way. Each later version replaces one answer with a faster one.

## 2. Language & stack

**v0 is pure Python + NumPy on CPU.** This is deliberate — it establishes a **correctness baseline** that all later versions must match.

- **Python**: Readability is the top priority. v0 teaches the math and data flow without build systems or type gymnastics. Every operation is explicit — no `nn.Module` magic hiding what we're trying to learn.
- **NumPy**: For tensor math. Dependency-light, forces us to write everything explicitly.
- **No PyTorch / JAX / Transformers**: Frameworks hide the mechanics we need to learn.

**Why pure Python for v0:**
1. **Correctness first**. If v0 produces wrong tokens, it's a bug in our understanding of the model. Adding GPU kernels to buggy math just makes debugging harder.
2. **Reference for conformance tests**. Every later version (v1 OpenCL, v4 ROCm, v6 CUDA) must produce byte-identical outputs to v0 under greedy sampling. v0 is the oracle.
3. **Learning the concepts before optimizing them**. You can't optimize what you don't understand. v0 is where you understand attention, RoPE, GQA, and the KV cache.

**The dual-layer architecture starts in v1.** v0 is the exception — everything after v0 uses Python for orchestration (scheduler, batching, cache management) and C/C++ for compute kernels (attention, GEMM, quantization). v0's NumPy implementation becomes the fallback path for when no GPU is available.

## 3. Big-picture data flow

```
text ──tokenize──▶ ids ──embed──▶ vectors
                                    │
                          ┌─────────┴─────────┐
                          │ N × decoder block │  ◀── reads/writes KV cache
                          └─────────┬─────────┘
                                    │
                              final RMSNorm
                                    │
                                 LM head
                                    │
                                 logits
                                    │
                                 sampler
                                    │
                              next token id
                                    │
                              (loop back)
```

Every box above is a Python module in v0.

---

## 4. Concepts (the learning section)

Each concept follows the same shape:

- **Intuition** — what it is and why it exists, in plain language
- **Math** — the actual formulas, with concrete shapes
- **Connections** — how it ties into the rest of the engine

Read them in order on a first pass. They build on each other.

### 4.1 Autoregressive generation

**Intuition.** A decoder LLM models the conditional probability `P(next_token | all_previous_tokens)`. To generate text you sample one token from that distribution, append it, and ask the model again. There is no lookahead, no beam search by default — just a loop.

**Math.** If the prompt is `x_1, ..., x_P` and we want to generate `T` tokens:

```
for t in P, P+1, ..., P+T-1:
    logits  = model(x_1, ..., x_t)         # shape [vocab_size]
    x_{t+1} = sample(softmax(logits))
```

**Connection.** The very first call (`t = P`) processes the *whole prompt*. Every later call processes *one new token*. This split — **prefill** vs **decode** — is so important it gets its own section (4.15).

### 4.2 Tokens and embeddings

**Intuition.** Models don't see characters; they see *tokens*, which are learned chunks of text (often subwords). The tokenizer turns text into integer IDs. The first thing the model does with those IDs is look them up in an **embedding table** to get one vector per token — that's how an integer becomes something a neural network can do math on.

**Math.** Given an embedding matrix `E` of shape `[vocab_size, d_model]` and input IDs of shape `[seq_len]`:

```
x = E[ids]    # shape [seq_len, d_model]    -- just an indexed lookup
```

No multiplication — it's literally a row lookup.

**Connection.** Many models *tie* the embedding matrix and the LM head (4.12), reusing the same matrix to map IDs → vectors at the input and vectors → logits at the output. Saves memory; you should know whether the checkpoint you load does this.

### 4.3 The decoder block at a glance

A decoder block is the unit that gets stacked N times (often 12, 32, 80…). One block is:

```
            ┌──────────────┐         ┌──────────────┐
   x ──┬──▶ │   RMSNorm    │ ──▶ Attn │              │ ──▶ + ──┬──▶ RMSNorm ──▶ MLP ──▶ + ──▶ x'
       │    └──────────────┘         └──────────────┘    ▲     │                        ▲
       └─────────────────────────────────────────────────┘     └────────────────────────┘
                       residual                                          residual
```

The two `+` arrows are **residual connections** (4.11): each sublayer's output is added back to its input. The continuously-updated `x` is called the **residual stream** — think of it as the model's working memory, refined a little by every layer.

A full forward pass is: `embed → N blocks → final RMSNorm → LM head → logits`.

### 4.4 RMSNorm

**Intuition.** As activations flow through dozens of layers their magnitudes drift — some dimensions blow up, others shrink to nothing. Normalization rescales each token's vector to a controlled magnitude before each sublayer, keeping the math numerically stable. **RMSNorm** is a stripped-down LayerNorm: it only divides by the root-mean-square (no mean subtraction) and has only a learned scale (no learned bias). It is cheaper than LayerNorm and works just as well in practice — which is why every modern Llama-style model uses it.

**Math.** For a single token vector `x` of shape `[d_model]` and a learned weight `g` of shape `[d_model]`:

```
rms(x)      = sqrt(mean(x_i^2) + eps)
RMSNorm(x)  = (x / rms(x)) * g
```

`eps` is a tiny constant (e.g. `1e-6`) to avoid division by zero on a near-zero vector.

**Connection.** Llama-style blocks use **pre-norm**: RMSNorm is applied to the *input* of each sublayer, but the residual stream itself is never normalized in place. There is also one final RMSNorm after the last block, just before the LM head.

### 4.5 Linear projections: Q, K, V

**Intuition.** Attention needs three views of each token:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer to be matched against?"
- **Value (V)** — "what information do I contribute if I'm matched?"

Each view is computed by a separate linear projection (matrix multiply) of the (normalized) input.

**Math.** With `x` of shape `[seq_len, d_model]`:

```
Q = x @ W_q     # [seq_len, n_q_heads  * head_dim]
K = x @ W_k     # [seq_len, n_kv_heads * head_dim]
V = x @ W_v     # [seq_len, n_kv_heads * head_dim]
```

Then reshape to split out the heads:

```
Q : [seq_len, n_q_heads,  head_dim]
K : [seq_len, n_kv_heads, head_dim]
V : [seq_len, n_kv_heads, head_dim]
```

Why might `n_q_heads != n_kv_heads`? See GQA (4.8).

**Connection.** Q describes the *current* tokens. K and V are what gets stored in the KV cache (4.14) — they describe past tokens and need to be remembered so future queries can look them up.

### 4.6 Multi-head attention

**Intuition.** Instead of one big attention computation in `d_model` space, we split the channels into `H` independent **heads**, each of size `head_dim = d_model / H`. Each head can attend to different patterns (one head might track local syntax, another long-range references, etc.). Heads run in parallel and are concatenated at the end.

**Math.** For a single head with `Q_h, K_h, V_h` of shape `[seq_len, head_dim]`:

```
scores  = Q_h @ K_h^T / sqrt(head_dim)    # [seq_len_q, seq_len_k]
weights = softmax(scores + causal_mask)   # [seq_len_q, seq_len_k]
out_h   = weights @ V_h                   # [seq_len_q, head_dim]
```

Concatenate the H head outputs along the channel axis and apply an output projection:

```
out = concat(out_1, ..., out_H) @ W_o     # [seq_len, d_model]
```

The `/ sqrt(head_dim)` is a temperature: it stops the dot products from growing with dimension and saturating the softmax (which would kill gradients during training and cause numerical pathologies during inference).

**Connection.** This is *the* operation a transformer LLM exists to compute. Every speed optimization in real engines (FlashAttention, paged attention, kernel fusion) is about computing this faster or with less memory. v0 does it the slow obvious way so you can see what is being optimized later.

### 4.7 Causal masking

**Intuition.** A decoder LLM is autoregressive: token `t` is allowed to attend only to tokens `1..t`, never to future tokens. Otherwise the model could trivially "cheat" by peeking ahead. We enforce this by adding a mask to the attention scores that sets future positions to `-inf`, so softmax assigns them weight 0.

**Math.** The causal mask `M` is a `[seq_len_q, seq_len_k]` matrix:

```
M[i, j] = 0       if j <= i    (allowed)
M[i, j] = -inf    if j >  i    (forbidden)
```

It is added to `scores` before the softmax.

**Connection.** During **decode** the query is a single new token and the keys span all past tokens, so the mask reduces to "everything is allowed" — you can skip it entirely. During **prefill** the mask is a triangle. This is one of the small but real differences between the two paths (4.15).

### 4.8 Grouped-Query Attention (GQA)

**Intuition.** The KV cache is the dominant memory consumer at inference time. If you have `H` query heads and a separate K/V per head, the cache scales linearly with `H`. **GQA** shrinks this by sharing each K/V head across a *group* of query heads. With `n_q_heads = 32` and `n_kv_heads = 8`, each KV head is shared by 4 query heads — and the cache is 4× smaller. Empirical quality loss is negligible.

**Math.** Before computing `Q @ K^T`, tile each KV head along the head axis to match the number of query heads:

```
group_size = n_q_heads / n_kv_heads
K_tiled    = repeat_interleave(K, group_size, axis=head_axis)
V_tiled    = repeat_interleave(V, group_size, axis=head_axis)
```

Now `K_tiled` and `V_tiled` have `n_q_heads` heads, and you can do per-head attention exactly as in 4.6.

**Connection.** GQA is why modern model configs have *both* `num_attention_heads` and `num_key_value_heads`. It also means the KV cache is allocated with shape `[seq_len, n_kv_heads, head_dim]`, not `[seq_len, n_q_heads, head_dim]` — important when sizing it (4.14).

### 4.9 RoPE (Rotary Position Embedding)

**Intuition.** Attention as defined so far is *permutation-invariant*: shuffle the input tokens, you get the same outputs (modulo the mask). The model has no notion of "this token came before that one". We need to inject position information.

The clever idea behind **RoPE**: encode position by **rotating** Q and K in 2D subspaces by an angle that depends on the token's position. When you then take `Q @ K^T`, the rotations interact such that the result depends only on the *relative* offset between the two positions — exactly what attention should care about.

**Math.** Pair up the `head_dim` channels into `head_dim / 2` 2D pairs. For position `m` and pair `i`, define an angle:

```
θ_{m,i} = m * base^(-2i / head_dim)        # base is typically 10000
```

The rotation applied to that pair is:

```
[ x' ]     [ cos θ   -sin θ ] [ x ]
[ y' ]  =  [ sin θ    cos θ ] [ y ]
```

You apply this to **Q and K** (not V) right after the linear projection. The `cos` and `sin` tables are precomputed once with shape `[max_seq_len, head_dim / 2]`.

**Connection.** RoPE plays beautifully with the KV cache: when you process a new token at position `t`, you only rotate *its* Q and K by `θ_t`. You never re-rotate cached Ks. This is one big reason RoPE is the default in modern LLMs — other positional schemes (absolute embeddings, ALiBi, relative bias) play less nicely with caching.

### 4.10 SwiGLU MLP

**Intuition.** Attention mixes information *across* tokens. The MLP transforms each token's vector *independently*, giving the model nonlinearity and capacity. Modern Llama-style models use a *gated* variant called **SwiGLU**, which performs better than the classic `linear → gelu → linear` for the same parameter budget.

**Math.** With three weight matrices `W_gate`, `W_up`, `W_down`:

```
gate = x @ W_gate                      # [seq_len, d_ff]
up   = x @ W_up                        # [seq_len, d_ff]
out  = (silu(gate) * up) @ W_down      # [seq_len, d_model]

silu(z) = z * sigmoid(z)
```

The elementwise product `silu(gate) * up` is the "gating": the `gate` branch decides which channels of `up` get to pass through.

**Connection.** `d_ff` is typically about 2.7× `d_model` for SwiGLU (vs 4× for plain MLPs), because the gating branch already adds parameters. This is why model configs distinguish `intermediate_size` from `hidden_size`.

### 4.11 Residual connections

**Intuition.** Each sublayer (attention, MLP) does not *replace* its input — it computes an *update* and adds it back: `x = x + sublayer(norm(x))`. The continuously-updated `x` is the **residual stream**. Without residuals, training deep networks fails because gradients vanish. At inference time the consequence is more subtle but still important: every layer makes a small adjustment to a shared running representation, rather than building a new one from scratch.

**Math.**

```
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

**Connection.** The residual stream stays at shape `[seq_len, d_model]` from the embedding layer all the way to the final RMSNorm. If you ever see those dimensions appear in a debugger, you're looking at the residual stream.

### 4.12 LM head and logits

**Intuition.** After the last block and the final RMSNorm, each token's vector lives in `d_model` space. To predict the next token we need a score for each item in the vocabulary. The **LM head** is a linear projection from `d_model` to `vocab_size`, and its outputs are called **logits** — un-normalized log-probabilities.

**Math.**

```
logits = x_final @ W_lm_head     # [seq_len, vocab_size]
```

If embeddings are tied, `W_lm_head` is just `E^T` (no separate parameters).

**Connection.** During generation we only care about `logits[-1]` — the score for the position right after the last input token. The other rows are useful during training but ignored at inference.

### 4.13 Sampling

**Intuition.** Logits are scores, not a token. We need a rule to turn scores into a choice. The simplest rule, **greedy**, just takes the argmax. More interesting rules (temperature, top-k, top-p) sample from the softmax distribution with various truncations. v0 uses greedy because it is deterministic — easier to debug.

**Math.**

```
greedy:       next = argmax(logits)
temperature:  probs = softmax(logits / T); next = sample(probs)
top-k:        keep the k largest logits, set the rest to -inf, then sample
```

**Connection.** Sampling is the *only* component of v0 that introduces randomness. If two runs disagree under greedy sampling, the bug is *not* in the sampler.

### 4.14 The KV cache (the central optimization)

This is the most important concept in the whole document. Read it twice if you need to.

**Intuition.** The naive approach to generating token `t+1` is to feed `x_1..x_t` through the entire model from scratch. That's O(t) work per generated token, so producing T tokens from a P-token prompt costs O((P+T)²) total. Quadratic. Miserable.

But notice: when we compute attention at step `t+1`, the only *new* thing is the query for the new token. All the keys and values for past tokens were already computed at previous steps. If we **store K and V for every token we have ever processed**, generating a new token only requires:

1. Compute Q, K, V for the *one* new token.
2. Append the new K and V to the stored arrays.
3. Compute attention of the new Q against *all* stored K and V.

Per-step work drops from O(t·d) (rerunning the whole model on the whole sequence) to O(t·d) for *just the attention dot product*, with all the per-layer matmuls collapsing to size 1 in the sequence dimension. This single trick is the difference between "LLM inference works" and "LLM inference is intractable".

**Math.** Per layer, the cache holds:

```
K_cache : [max_seq_len, n_kv_heads, head_dim]
V_cache : [max_seq_len, n_kv_heads, head_dim]
```

A counter `length` tracks how many positions are filled. At each decode step:

```
Q_new, K_new, V_new = project(x_new)         # one new token
K_cache[length] = K_new
V_cache[length] = V_new
length += 1

K_used = K_cache[:length]                    # all past + new
V_used = V_cache[:length]
out    = attend(Q_new, K_used, V_used)
```

**Why engines like vLLM exist.** Look at the cache from a *systems* angle:

- Each request needs its own cache (different prompt → different K/V).
- You don't know how long a request will be when it arrives, so you must either over-allocate (waste memory) or grow the buffer (fragmentation).
- KV cache is *the* dominant memory consumer at inference time. For long sequences it is bigger than the model weights themselves. If you waste it, you can serve fewer concurrent users.
- If two requests share a prefix (very common in chat: the same system prompt), naive caches duplicate that prefix's K/V across both — pure waste.

vLLM's **paged attention** solves this by allocating the cache in fixed-size *blocks*, like an operating system's virtual memory pager. Requests grow without contiguous reservation, blocks can be freed and reused, and shared prefixes can be expressed by sharing block pointers across requests. v0 deliberately does the *naive* thing — one contiguous buffer per request, sized to `max_seq_len` — *so that you feel the pain*. v1 will introduce paged attention as the relief.

**Connection.** Almost every other concept on this page ultimately serves the cache: GQA makes it smaller; RoPE makes it append-friendly; the prefill/decode split is the workflow that fills then uses it; the scheduler decides whose cache gets to advance next.

### 4.15 Prefill vs. decode

**Intuition.** The model runs in two very different regimes:

- **Prefill.** First call for a request. Input length = prompt length P. We process the whole prompt at once, computing K and V for all P positions and writing them into the cache. The output we care about is the logits at position P-1 (to sample the *first* generated token). The matmuls are big (Q @ K^T is P×P) — **compute-bound**.

- **Decode.** Every subsequent call. Input length = 1. We compute Q, K, V for one token, append K and V to the cache, attend the new Q against the entire cache, and get one row of logits. The matmuls are tiny but we read the *entire* cache from memory each step — **memory-bound**.

This asymmetry — big-batch-of-tokens-once vs. one-token-many-times — is the foundation of nearly every scheduling decision an inference engine makes.

**Math.** Same `forward()` function, different shapes:

```
prefill:  input_ids [P]   positions [0..P-1]   output logits [P, vocab]
decode:   input_ids [1]   positions [t]        output logits [1, vocab]
```

The mask is also different (full triangle vs. trivial all-allowed).

**Connection.** v0 keeps these as two separate Python paths and runs them serially for one request at a time. v1 will let prefill and decode of *different* requests share a single model invocation (continuous batching) — that requires understanding the asymmetry first.

### 4.16 Scheduling and the request lifecycle

**Intuition.** Even with one request at a time, the engine has to track *state*: is this request waiting to start, mid-prefill, mid-decode, or finished? The **scheduler** owns those states and decides what to advance on each engine **step**. In v0 the policy is dumb (one request at a time, FIFO), but the *abstraction* — a pool of `Request` objects with statuses, advanced one step at a time — is the same one v1 will use to do continuous batching across many requests.

**State machine for a v0 request.**

```
NEW ──admit──▶ WAITING ──pick──▶ PREFILL ──run──▶ DECODE ──run──▶ DECODE ──...──▶ DONE
                                                     │
                                       EOS / max_tokens / cache full
                                                     ▼
                                                    DONE
```

**Connection.** Every advanced feature (continuous batching, preemption, prefix caching, fairness, priorities) is a *different policy* over the same state machine. Build the state machine cleanly in v0 and v1 is mostly bookkeeping.

### 4.17 Why an "engine" exists at all

You could write a script that loads a model, runs `forward()` in a loop, and prints text. That works for one request and wastes most of your hardware. An *engine* exists to:

1. **Reuse computation** (the KV cache).
2. **Manage memory** efficiently (paged attention, eviction).
3. **Pack the device** (continuous batching, chunked prefill).
4. **Serve many users at once** with bounded latency (scheduling, admission control).
5. **Hide all of that** behind a simple `generate(prompt)` API.

v0 builds (1) and a tiny piece of (5). Each later version adds one of the others.

---

## 5. Component map

```
                  ┌────────────────┐
   user prompt ──▶│   Tokenizer    │── token ids ──┐
                  └────────────────┘               │
                                                   ▼
                                          ┌────────────────┐
                                          │    Engine      │  (orchestrator)
                                          └────────────────┘
                                                   │
                          ┌────────────────────────┼─────────────────────────┐
                          ▼                        ▼                         ▼
                  ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
                  │  Scheduler   │         │  KV Cache    │          │  Model       │
                  │ (request +   │         │  (per-layer  │          │  Runner      │
                  │  step loop)  │         │   K,V store) │          │  (forward()) │
                  └──────────────┘         └──────────────┘          └──────────────┘
                                                                            │
                                                                            ▼
                                                                  ┌──────────────────┐
                                                                  │     Sampler      │
                                                                  │ (greedy / top-k) │
                                                                  └──────────────────┘
                                                                            │
                                                                            ▼
                                                                       next token
```

### Component summary

| Component    | One-line role                                                  | Concept refs           |
| ------------ | -------------------------------------------------------------- | ---------------------- |
| Tokenizer    | text ↔ token ids                                               | 4.2                    |
| Model        | NumPy implementation of the decoder stack                      | 4.3 – 4.12             |
| KV cache     | per-request contiguous K/V store                               | 4.14                   |
| Model Runner | calls `model.forward` for either prefill or decode             | 4.15                   |
| Scheduler    | owns request state, picks what runs next                       | 4.16                   |
| Sampler      | logits → next token id                                         | 4.13                   |
| Engine       | top-level loop wiring everything together                      | 4.1, 4.17              |

---

## 6. Features to implement in v0

This is the concrete checklist. Each feature links back to the concept it implements (section 4) so you can re-read the relevant background while coding it. Items are listed roughly in build order.

### Model-side

| #   | Feature                          | Concept | What you actually write                                                |
| --- | -------------------------------- | ------- | ---------------------------------------------------------------------- |
| F1  | Weight loading                   | 4.2     | Read `safetensors`/`.npz` into a dict of NumPy arrays                  |
| F2  | RMSNorm                          | 4.4     | A 5-line function: rms-divide and scale                                |
| F3  | RoPE                             | 4.9     | Precompute `cos`/`sin` tables; apply rotation to Q and K               |
| F4  | Attention + KV-cache reads       | 4.5–4.8 | Q/K/V projection, GQA tiling, scaled dot-product, causal mask          |
| F5  | SwiGLU MLP                       | 4.10    | `(silu(x @ W_gate) * (x @ W_up)) @ W_down`                             |
| F6  | Decoder block + full forward     | 4.3, 4.11, 4.12 | Stack N blocks with residuals, final norm, LM head             |

### Runtime

| #   | Feature                          | Concept | What you actually write                                                |
| --- | -------------------------------- | ------- | ---------------------------------------------------------------------- |
| F7  | Naive KV cache                   | 4.14    | Per-request `[max_seq_len, n_kv_heads, head_dim]` arrays + length      |
| F8  | Prefill vs. decode split         | 4.15    | Two code paths through the model runner, different shapes & masks      |
| F9  | Greedy sampler                   | 4.13    | `argmax(logits[-1])`                                                   |
| F10 | FIFO scheduler                   | 4.16    | A queue, a `Request` dataclass with a status field, a `step()` method  |
| F11 | Engine top-level loop            | 4.1, 4.17 | `engine.generate(prompt, max_tokens)` orchestrating everything       |
| F12 | Stop conditions                  | 4.16    | EOS token, `max_tokens`, cache-full → mark request DONE                |

### Plumbing

| #   | Feature                          | Why it exists                                                         |
| --- | -------------------------------- | --------------------------------------------------------------------- |
| F13 | Tokenizer wrapper                | Text in / text out; isolates the engine from any specific tokenizer   |
| F14 | End-to-end example script        | The acceptance test for v0 — if it runs, v0 is done                   |
| F15 | Smoke test                       | Cheap regression net before refactoring toward v1                     |

---

## 7. Request lifecycle in v0 (concrete walkthrough)

1. User calls `engine.generate("Hello", max_tokens=20)`.
2. Tokenizer encodes the prompt → `ids = [101, 5, 88]` (say).
3. Scheduler creates a `Request` holding: prompt ids, generated ids (empty), allocated KV cache, status `WAITING`.
4. Engine `step()` picks the request → status `PREFILL`.
5. **Prefill**: model runs once over all 3 prompt tokens. Cache positions `[0, 3)` are filled. Sampler picks the next token from `logits[2]`. Status → `DECODE`.
6. **Decode loop**: model runs once per generated token, with input length 1, reading positions `[0, t)` from the cache and writing the new K/V at position `t`. Each step calls the sampler, appends to generated ids, checks stop conditions.
7. EOS or `max_tokens` reached → status `DONE`.
8. Tokenizer decodes the generated ids back to text. Engine returns the string.

---

## 8. What v0 deliberately does NOT have

Each of these is intentionally out of scope and will become its own learning milestone:

| Feature              | Why deferred                                                            | Target version |
| -------------------- | ----------------------------------------------------------------------- | -------------- |
| Paged attention      | Need the naive baseline first to feel the memory waste                  | v1             |
| Continuous batching  | Need a working single-request loop to extend                            | v1             |
| Prefix caching       | Small addition on top of paged attention; huge TTFT win for shared prompts | v1          |
| OpenCL backend       | First GPU backend; v0 establishes the CPU baseline to compare against   | v1             |
| Weight + KV quantization | Target OpenCL hardware has limited VRAM — needed to actually run v1's vision | v2       |
| Speculative decoding | Orthogonal to backends/memory; lands right after quantization           | v3             |
| ROCm backend         | Second real backend; validates v1's abstraction on AMD hardware (also ports spec-decoding kernels) | v4 |
| Multi-GPU (TP / PP)  | First run on OpenCL and ROCm together; user wants to test on OpenCL too | v5             |
| CUDA backend         | Explicitly the LAST real backend priority (user preference)             | v6             |
| Vulkan backend       | Optional, not a priority                                                | v7 / optional  |

See [`architecture_v1.md`](architecture_v1.md) for the full plan of what comes after v0.

---

## 9. Directory layout (proposed)

```
vllm-from-scratch/
├── AGENTS.md -> CLAUDE.md
├── CLAUDE.md
├── README.md
├── docs/
│   └── architecture_v0.md          # this file
├── vllm_fs/                        # the python package
│   ├── __init__.py
│   ├── engine.py                   # Engine
│   ├── scheduler.py                # Scheduler + Request
│   ├── model_runner.py             # ModelRunner
│   ├── kv_cache.py                 # naive contiguous KV cache
│   ├── sampler.py                  # greedy sampler
│   ├── tokenizer.py                # tokenizer wrapper
│   └── models/
│       └── tiny_llama.py           # NumPy model implementation
├── examples/
│   └── run_v0.py                   # smallest possible end-to-end script
└── tests/
    └── test_v0_smoke.py
```

---

## 10. Recommended models for v0 testing

For rapid iteration during development, use **small models** that load fast and run on CPU. Three approaches, from smallest to largest:

### Option A: Toy synthetic weights (fastest)
Create random NumPy arrays with correct shapes matching a tiny config:
- `d_model=64`, `n_layers=2`, `n_heads=4`, `vocab_size=100`
- Use for pure unit testing when you just need "something that runs"
- No downloads, instant load, deterministic

### Option B: SmolLM2-135M (v0 target model)
**The official v0 test model.** HuggingFace's 135M parameter model from February 2025:
- ~135M parameters, ~270MB in FP16
- Architecture: Llama-style (RMSNorm, RoPE, GQA, SwiGLU) — matches v0 spec
- Battle-tested: 10M+ downloads, peer-reviewed paper (COLM 2025)
- Apache 2.0 license, actively maintained
- Leaves ~730MB for KV cache on 1GB VRAM — plenty of headroom
- Repository: `HuggingFaceTB/SmolLM2-135M`

Use this for all integration tests and the final v0 acceptance script.

### Option C: SmolLM2-360M (slightly larger)
If you have more VRAM and want better quality:
- 360M parameters, ~720MB in FP16
- Same architecture as 135M variant
- Leaves ~280MB for KV cache — tighter but workable
- Repository: `HuggingFaceTB/SmolLM2-360M`

### Testing strategy
- **Unit tests**: Use Option A (synthetic) — fast, no network
- **Integration tests**: Use Option B (SmolLM2-135M) — validated, lightweight
- **v0 success criteria** (Section 12): Should pass with SmolLM2-135M

---

## 11. SmolLM2-135M architecture details

When implementing v0, use these config values from SmolLM2-135M:

```json
{
  "hidden_size": 576,
  "num_hidden_layers": 30,
  "num_attention_heads": 9,
  "num_key_value_heads": 3,
  "intermediate_size": 1536,
  "rms_norm_eps": 1e-05,
  "vocab_size": 49152,
  "max_position_embeddings": 2048,
  "rope_theta": 10000
}
```

Notes:
- **GQA ratio**: 9 query heads / 3 KV heads = 3:1 grouping
- **Head dim**: 576 / 9 = 64
- **Tie word embeddings**: True (LM head shares weights with embedding)
- **Sliding window attention**: None (global attention only)

---

## 12. Success criteria for v0

- `python examples/run_v0.py` loads a small model, runs a prompt, and prints generated text.
- Each module above exists, is under ~200 lines, and has a docstring explaining the concept it embodies.
- A companion doc walks through one forward pass and lists the exact shapes that flow through the system.

When all of that is true, v0 is "done" and we move on to v1 (paged attention, continuous batching, and the OpenCL backend — together aimed at higher throughput, lower TTFT, and more concurrency).

---

## 11. Further reading

Once a concept feels shaky, go to the source:

- *Attention Is All You Need* — Vaswani et al. (the original transformer paper)
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al. (RoPE)
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al.
- *Root Mean Square Layer Normalization* — Zhang & Sennrich (RMSNorm)
- *GLU Variants Improve Transformer* — Shazeer (SwiGLU)
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* — Kwon et al. (the vLLM paper — read this before starting v1)
- *FlashAttention* — Dao et al. (fused attention kernel; relevant when we replace v0's naive attention)
- Andrej Karpathy's **nanoGPT** — minimal training implementation; very readable Python
