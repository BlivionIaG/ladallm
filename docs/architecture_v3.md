# Architecture v3 — Speculative Decoding

> **Prerequisites:** [`architecture_v0.md`](architecture_v0.md), [`architecture_v1.md`](architecture_v1.md), and [`architecture_v2.md`](architecture_v2.md). v3 assumes paged attention, continuous batching, prefix caching, the OpenCL backend, and weight + KV quantization are all working.

v3 adds **speculative decoding** to the engine: a technique that lets a small "draft" model propose several tokens at once, then has the main "target" model verify them in a single forward pass. When the draft is right, you generate multiple tokens for the price of one decode step. When it is wrong, you fall back to one accepted token and try again. The expected number of tokens per step rises from 1 to typically 2–4× without changing what the target model would have generated.

This is a *different kind* of optimization than v1 and v2:

- **v1** raised throughput by packing many requests into each step (continuous batching) and made memory go further (paged attention, prefix caching).
- **v2** shrank the bytes-per-weight and bytes-per-KV-token so the model fits and decode is bandwidth-cheaper.
- **v3** changes the **per-request decoding loop itself** so each step produces several tokens instead of one. It is the first technique we add that improves *single-request* TTFT and tokens/sec on top of all the systems work below it.

---

## 1. What we are building

A speculative-decoding subsystem layered on top of v1's request lifecycle and v2's quantization. After v3:

- The engine can be configured with a `(target_model, draft_model)` pair instead of a single model.
- For each decode step the draft model proposes `k` tokens (typical `k = 4` or `k = 8`); the target model verifies them in one batched forward pass; the engine accepts the longest matching prefix and emits those tokens.
- Greedy decoding produces **byte-identical outputs** to v2's non-spec greedy path. Sampling decoding produces a *probabilistically equivalent* distribution to v2's plain sampler — this is a real correctness property we will prove.
- Both target and draft model weights can be quantized (typically W4A16 + int8 KV), so the draft model is small *and* compressed. This is the only realistic shape on VRAM-limited hardware.
- A measurement harness reports the **acceptance rate** and the resulting wall-clock speedup, so the technique's value is observed, not assumed.

### Why v3 lands here (between quantization and ROCm)

- **It does not depend on ROCm.** Spec decoding is backend-agnostic. The OpenCL backend from v1+v2 is enough.
- **It depends on v2.** A draft model has to fit in memory next to the target model. On VRAM-limited hardware that requires quantization (typically int4 weights for both). v3 inherits v2's quantized formats unchanged — we don't re-quantize, we just load *two* quantized models.
- **It makes the v4 ROCm port stronger.** Because v3 lands first, the ROCm port in v4 has to handle the multi-token verification path from day one. That's strictly more abstraction validation than porting a 1-token kernel and retrofitting verification later — and it avoids writing the spec-decoding kernels twice.
- **It is a coherent learning unit on its own.** Draft model loader, multi-token verification through the existing forward pass, accept-reject sampler, scheduler tweak for "this request advanced by `m` tokens this step (`1 ≤ m ≤ k+1`)". That's enough to teach in one version, small enough to debug.

---

## 2. Concepts

Same shape as v0/v1/v2: **Intuition → Math/Mechanism → Connections**.

### 2.1 Why speculative decoding works at all

**Intuition.** Decoding one token at a time is *memory-bound*: each step you read all the model weights and the entire KV cache from memory just to compute one token's logits. The arithmetic units are mostly idle. If you could give the model **more work to do** with the same number of memory reads, the per-token cost would drop dramatically.

That is exactly what spec decoding does. The target model is asked to verify `k` candidate tokens at once, which means computing `k` rows of logits in a single forward pass. Reading the weights and the cache once produces `k` tokens of useful output instead of one — *if* the candidates are good enough that most are accepted. The draft model is the source of those candidates: it is a much smaller LLM that runs cheaply, often with the same tokenizer and distribution as the target.

The arithmetic intuition: if a decode step on the target model takes time `T` and produces 1 token, and a verify step on the target model also takes ~`T` (because attention is the dominant cost and `k` extra rows are trivially small) and produces an expected `1 + α·k` accepted tokens (where `α` is the acceptance probability), then the speedup is roughly `1 + α·k`. For `α = 0.7, k = 4`, that's a 3.8× speedup — minus the cost of running the draft model, which is small if the draft is much smaller than the target.

**Connection.** This is *the* technique that improves single-request decoding speed without changing the output distribution. Everything else in v0–v2 was about packing more work into the step or making the step cheaper; v3 is about **producing more tokens per step**.

### 2.2 The draft model

**Intuition.** The draft model is a smaller LLM whose only job is to propose plausible next tokens cheaply. Three things matter about it:

1. It must produce tokens in the **same vocabulary** as the target. Otherwise we cannot compare "what the draft proposed" with "what the target would assign probability to".
2. It must be **much faster** than the target — typically 5–10× smaller in parameter count. A 1B draft for a 7B target is a common ratio.
3. It must be **good enough** that its proposals are accepted reasonably often. If the draft and target disagree most of the time, the draft is just wasted compute and you've made things slower.

Where do drafts come from? Three options, in increasing order of effort:

- **A pretrained smaller sibling.** Llama 3 8B as draft for Llama 3 70B target, etc. Same family, same tokenizer, similar distribution. This is the easy case and what v3 assumes.
- **A distilled draft.** Train a small model specifically to mimic the target. Better acceptance rates, more work to set up. Out of scope for v3.
- **A self-draft.** Use the target model's own early layers, or a single-layer extension (Medusa, EAGLE). State of the art, but too elaborate for v3 — we keep this as a future idea, not a feature.

For v3 we go with option 1 and assume the user provides two checkpoints with matching tokenizers.

**Connection.** Loading a second model is the first time the engine has to manage *two* simultaneous sets of weights and KV caches. v2's quantization is what makes this fit on the target hardware: a 7B target at int4 + a 1B draft at int4 is roughly `3.5 + 0.5 = 4 GB` of weights, plus their KV caches. Without v2 this is uncomfortable on small GPUs.

### 2.3 The propose-verify-accept loop

**Intuition.** Each decode step now does three things instead of one:

1. **Propose.** Run the draft model `k` times to generate `k` candidate tokens, autoregressively. The draft has its own KV cache that grows by `k` per step.
2. **Verify.** Run the target model **once**, on all `k` proposed tokens at once. The target produces `k+1` rows of logits: one for each proposed position, plus one for the position right after the last proposal (which is the target's own next-token prediction if all `k` are accepted).
3. **Accept.** Walk left-to-right through the proposals, comparing each to what the target wanted. Accept the longest prefix that matches (under whatever rule the sampler uses — see 2.4). Emit those tokens. Truncate the draft's KV cache to the accepted length and try again next step.

**Math (greedy case).** Let the draft propose tokens `d_1, d_2, ..., d_k`. Let the target's argmax at each verification position be `t_1, t_2, ..., t_k, t_{k+1}`. Accept rule:

```
accept i tokens, where i = max j such that d_1 = t_1, d_2 = t_2, ..., d_j = t_j
emit d_1, ..., d_i, AND emit t_{i+1}     # always one extra "free" token from the target
```

So the minimum tokens per step is **1** (no draft tokens accepted, but the target's own next-token prediction is still emitted) and the maximum is **k+1**. This is a key property: **even when the draft is wrong on every token, you still produce 1 token per step**, so spec decoding never makes greedy decoding *slower* — only the total wall time of the draft model is wasted.

**Diagram.**

```
draft proposes:    d1   d2   d3   d4
target verifies:   t1   t2   t3   t4   t5      (one big batched forward)
                    ✓    ✓    ✗
emit:              d1   d2   t3                 (3 tokens this step)
draft cache:       keep d1, d2; drop d3, d4
target cache:      append t3; positions for d3, d4 are also written but logically discarded
```

**Connection.** This loop is the new "step" function. The continuous batching scheduler from v1 still drives it — but instead of advancing each request by 1 decode token per step, it advances each request by 1 to `k+1` tokens. The scheduler does not need to know about spec decoding; it just calls the runner, the runner returns "I advanced this request by `m` tokens this step", and the scheduler updates state accordingly. This is one of the big payoffs of v1's clean separation: Python orchestrates, C++ computes. The spec-decoding logic lives in the C++ paged-attention kernel and sampler, not in the Python scheduler.

### 2.4 Acceptance under sampling (the hard correctness bit)

**Intuition.** Greedy is easy: argmax matches or doesn't. Sampling is harder because the draft and target have **different distributions**, and we want the engine's output to be probabilistically equivalent to "just sampled from the target". You cannot just say "if the draft sampled `d`, compare it to a sample from the target" — that would change the output distribution.

The right answer is the **modified rejection sampling** rule from the original spec-decoding paper (Leviathan et al., 2022; Chen et al., 2023). For each draft position with proposed token `d`, target probability `p_target(d)`, and draft probability `p_draft(d)`:

```
accept with probability  min(1, p_target(d) / p_draft(d))
on reject:               sample a replacement from a residual distribution
                         p_residual(x) ∝ max(0, p_target(x) - p_draft(x))
```

The residual distribution exists precisely so that the *unconditional* distribution of the emitted token matches `p_target` exactly. This is provable in a few lines and is the reason spec decoding is mathematically equivalent to plain sampling, not an approximation.

In practice you implement it like this:

1. After verification you have `p_target` (from the target's softmax) and `p_draft` (recorded by the draft when it proposed the token).
2. Sample `u ~ Uniform(0, 1)`.
3. If `u < p_target(d) / p_draft(d)`, accept `d`. Move to the next position.
4. Otherwise, reject. Sample one token from `p_residual` and emit it. Stop the inner loop (the rest of the proposals are now stale).

The "reject + sample from residual" path produces 1 token; the "accept everything plus the bonus target token" path produces `k+1` tokens; the average is somewhere in between, weighted by acceptance rate.

**Connection.** This is the part of v3 that has the most subtle math and the highest risk of silent bugs. The conformance test is critical: we should be able to set `k = 0` (no draft tokens) and recover the v2 sampler exactly, byte-for-byte. We should also be able to seed the RNG, run the same prompt with and without spec decoding, and get the same output **in distribution** (not byte-identical for sampling, but statistically equivalent over many runs).

### 2.5 KV cache management for two models

**Intuition.** With one model, KV cache is simple: each request has a block table, the paged-attention kernel reads it. With spec decoding you now have **two** block tables per request — one for the draft model's cache and one for the target model's cache — and they advance together but get truncated independently when proposals are rejected.

What this means concretely:

- Each `Request` object now holds `target_blocks` and `draft_blocks`.
- The block pool from v1/v2 doesn't need to know there are two of them; both block tables draw from the same pool, refcounted independently.
- When `m` tokens are accepted out of `k` proposed, the draft's block table needs to **truncate** the last `k - m` positions. The target's block table also wrote K/V at positions for the rejected tokens; those slots are now stale and need to be logically discarded (length counter goes back, the bytes can stay — they'll be overwritten next step).
- Prefix caching from v1 still works for the **target** model's prompt prefix. The draft model has its own prefix cache (or shares one — implementation choice).

**Math.** If the request was at target position `t` before the step and accepted `m` tokens, after the step it is at target position `t + m`. The target's logical KV length goes from `t` to `t + m`. The draft's logical KV length also goes from `t` to `t + m` (the draft proposed `k` extra tokens, but only `m` were accepted, so its last `k - m` cache writes are abandoned).

**Connection.** This is where the v1 block manager pays off again. Because positions are logical and managed by a length counter + block table, "truncate by `k - m`" is two integer updates — no memory copying, no kernel work. If the v1 cache had been one contiguous buffer per request, this would be much messier.

### 2.6 The scheduler change (smaller than you'd think)

**Intuition.** The continuous-batching scheduler from v1 advances each request by exactly 1 decode token per step. The only change v3 needs is: each request now advances by **1 to k+1** tokens per step. Everything else — the request pool, the FIFO/priority logic, the chunked-prefill path, the prefix-cache lookups — is unchanged.

The cleanest implementation is to make this a property of the **runner**, not the scheduler. The runner's `decode_step()` returns an integer `m` per request (number of tokens emitted this step) and the scheduler treats it as a black box. Spec decoding then becomes a runner mode (`runner.set_spec_decoding(draft_model, k)`) that swaps the inner loop without touching the scheduler.

This is also why we didn't need to block on v4 (multi-GPU): the multi-GPU work happens *under* the runner, not above it, so the two changes are independent.

**Connection.** Same principle as before: the v1 abstractions are clean enough that an entirely new technique (spec decoding) lands almost entirely in one place (the runner), with one small touch in the request struct (two block tables) and zero changes in the scheduler. If we needed to touch the scheduler to add spec decoding, we'd have a design problem to fix here.

### 2.7 Measuring it honestly

**Intuition.** Spec decoding is famous for not always speeding things up. Common reasons:

- **Acceptance rate is too low** (the draft is bad for this prompt or this domain). The draft cost dominates.
- **The draft is too big** relative to the target. Speedup ratio collapses.
- **The target's verification step is so small that the draft cost is non-negligible.** This happens for short prompts or tiny models.
- **Sampling temperature is high**, which inherently lowers acceptance rate (less peaked distributions disagree more often).

So v3 is not done until it can *measure* what's happening. The measurement pipeline:

1. **Acceptance rate.** Per request, log how many of the `k` proposed tokens were accepted on each step. The mean over the run is the acceptance rate.
2. **Tokens per step.** Mean number of tokens emitted per call to the runner's decode step. Should be `1 + α·k` if the math is right.
3. **Wall-clock speedup.** Measured against the same prompt run with `k = 0`. This is the actual user-visible win.
4. **Draft overhead.** Time spent in the draft model as a fraction of total. If this is > 30% and acceptance is low, the configuration is wrong.
5. **Per-token quality.** Confirm that greedy spec decoding produces byte-identical output to greedy non-spec decoding. Confirm that sampled spec decoding has the same perplexity distribution as sampled non-spec decoding (this is the statistical version of "byte-identical").

**Connection.** Numbers 1–4 are runtime metrics; 5 is a correctness property. v3 is not done until both the speedup is measured *and* the correctness is verified — anything less and we're shipping a potentially wrong optimization.

---

## 3. Features to implement in v3

### Engine + runner

| #    | Feature                                                       | Concept | Notes                                                   |
| ---- | ------------------------------------------------------------- | ------- | ------------------------------------------------------- |
| F301 | Two-model engine config (target + draft, both quantized)      | 2.2     | Extends the loader to take a pair                       |
| F302 | Per-request dual block tables (target + draft)                | 2.5     | Tiny request-struct extension                           |
| F303 | Draft propose loop (`k` autoregressive steps on draft)        | 2.3     | Uses the existing runner unchanged                      |
| F304 | Target verify call (one batched forward over `k` tokens)      | 2.3     | Requires the model runner to accept inputs of length `k+1` |
| F305 | Greedy accept-reject inner loop                               | 2.3     | The simple correctness baseline                         |
| F306 | Sampling accept-reject inner loop (modified rejection sampling) | 2.4   | The mathematically correct sampler                      |
| F307 | Cache truncation on reject                                    | 2.5     | Two integer updates, both block tables                  |
| F308 | Runner mode flag (`spec_k = 0` disables, recovers v2 behavior) | 2.6    | Critical for the conformance test                       |

### Measurement and correctness

| #    | Feature                                                       | Concept | Notes                                                   |
| ---- | ------------------------------------------------------------- | ------- | ------------------------------------------------------- |
| F309 | Per-step acceptance counters and logging                      | 2.7     | Cheap; lives in the runner                              |
| F310 | Tokens-per-step + wall-clock-speedup metrics                  | 2.7     | Extends the v1 benchmark harness                        |
| F311 | Draft-overhead breakdown (time in draft vs target)            | 2.7     | Profiler hooks                                          |
| F312 | Greedy conformance: spec ≡ non-spec, byte-identical           | 2.7     | Hard correctness gate                                   |
| F313 | Sampling conformance: distribution check over many seeds      | 2.7     | Statistical, not byte-identical                         |

### Plumbing

| #    | Feature                            | Notes                                                                  |
| ---- | ---------------------------------- | ---------------------------------------------------------------------- |
| F314 | v3 example script                  | `run_v3.py --target ... --draft ... --spec-k 4`                        |
| F315 | Spec-decoding doc in `docs/perf/`  | Acceptance rates and speedups for representative model pairs           |

**Build order suggestion:** F301 → F302 → F308 (just the flag, default `k=0`, runner unchanged) → F304 (the runner now accepts input length > 1 in decode) → F303 → F305 (greedy spec decoding end-to-end) → F312 (lock in greedy correctness, this is the make-or-break test) → F309 → F310 → F306 → F313 → F311 → F314 → F307 was actually already needed by F305, build it there → F315.

---

## 4. Success criteria for v3

- `python examples/run_v3.py --target <7B> --draft <1B> --spec-k 4` runs end-to-end on the OpenCL backend.
- Greedy spec decoding output is **byte-identical** to greedy non-spec decoding for a fixed prompt (F312).
- Sampling spec decoding output is **statistically equivalent** to sampling non-spec decoding across many seeds (F313).
- Measured wall-clock speedup is at least **1.5×** over the v2 baseline on the test workload, for a target/draft pair where the draft has reasonable acceptance rate (~0.6+).
- Acceptance rate, tokens per step, draft overhead, and wall-clock speedup are all reported by the benchmark and checked into `docs/perf/spec-decoding.md`.
- Zero changes to the scheduler or block manager — all the work lives in the runner and the request struct.

When those hold, v3 is done and v4 (ROCm — now porting fp16 *and* quantized *and* spec-decoding kernels in one go) becomes the next doc.

---

## 5. What v3 still does NOT have

| Feature              | Why deferred                                                                     | Target version |
| -------------------- | -------------------------------------------------------------------------------- | -------------- |
| ROCm backend         | Now ports fp16 + quantized + spec-decoding kernels in one go                     | v4             |
| Multi-GPU (TP / PP)  | v4 preps the hooks; v5 implements real collectives on both OpenCL and ROCm       | v5             |
| Self-speculative decoding (Medusa, EAGLE) | Needs a separate trained head; out of scope                       | later          |
| Tree-of-drafts / multi-draft | More complex acceptance logic; v3 keeps it linear                        | later          |
| CUDA backend         | Explicitly the LAST backend priority                                             | v6             |
| Vulkan backend       | Optional, not a priority                                                         | v7 / optional  |

---

## 6. Further reading

- **Leviathan, Kalman, Matias** — *Fast Inference from Transformers via Speculative Decoding* (2022). The original paper. Read sections 2 and 3 for the rejection-sampling math.
- **Chen et al.** — *Accelerating Large Language Model Decoding with Speculative Sampling* (DeepMind, 2023). Companion paper, very clear writeup of the modified rejection-sampling rule.
- **Spector & Re** — *Accelerating LLM Inference with Staged Speculative Decoding* (2023). Extension to multi-stage drafts.
- **Medusa** — *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (Cai et al., 2024). Self-speculation; out of scope for v3 but the right "what's next" reference.
- **EAGLE** — *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* (Li et al., 2024). The current state of the art for self-speculation.
- **vLLM speculative decoding source** — `vllm/spec_decode/`. Production reference for everything in this doc; especially the worker / proposer / verifier split and the test suite.
