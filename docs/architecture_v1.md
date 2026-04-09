# Architecture v1 — Throughput, TTFT, Concurrency, and OpenCL

> **Prerequisite:** [`architecture_v0.md`](architecture_v0.md). v1 only makes sense once v0 runs end-to-end. Every v1 feature is a *replacement* for a v0 component, motivated by a measurable weakness in v0 that you have already felt.

This document follows the same shape as v0:

1. What we are building (and why this is the right next step)
2. Concepts (the learning section — paged attention, continuous batching, OpenCL)
3. Features to implement
4. What v1 still does NOT have

---

## 1. What we are building

v0 proved we can correctly produce tokens. v1 is about doing it **fast** and **for many users at once**. Concretely, the three success metrics for v1 are:

- **Throughput** — total tokens generated per second across all in-flight requests. v0 maxes this out at "one request at a time", which leaves the device almost entirely idle.
- **TTFT (Time To First Token)** — how long a user waits between submitting a prompt and seeing the first generated token. In v0, TTFT for a queued request is "however long every request ahead of you takes to *finish*". That is unacceptable for any real workload.
- **Concurrency** — how many simultaneous requests the engine can handle without falling over. v0 handles exactly one.

To improve all three we will introduce three big changes — each is a v1 concept and a v1 feature group:

| Change                  | Fixes the v0 weakness…                                                              | Improves            |
| ----------------------- | ----------------------------------------------------------------------------------- | ------------------- |
| **Paged attention**     | KV cache is over-allocated and per-request, so memory limits concurrency           | concurrency         |
| **Continuous batching** | One request at a time leaves the device idle and starves queued requests           | throughput, TTFT    |
| **Prefix caching**      | Shared prompts (system prompts, few-shot examples) are re-prefilled for every request | TTFT, throughput |
| **OpenCL backend**      | NumPy on CPU caps the tokens/sec ceiling regardless of how clever the scheduler is | throughput, TTFT    |

The first three are **systems** changes (how we manage memory and how we schedule work). The last is a **compute** change (where the math actually runs). All four are needed; each addresses a different bottleneck.

---

## 2. Concepts

Same shape as v0: **Intuition → Math/Mechanism → Connections**.

### 2.1 The pain points of v0 (a recap with numbers)

Before introducing solutions, name the problems sharply.

**Pain 1: KV cache waste.** v0 allocates `[max_seq_len, n_kv_heads, head_dim]` per request, per layer, *up front*. If `max_seq_len = 4096` but the request only generates 200 tokens, ~95% of that allocation is wasted. With even modest memory pressure this means you can hold maybe a handful of concurrent requests in memory — most of which sit unused.

**Pain 2: Device idle time.** During **decode**, each step has tiny matmuls (sequence length 1). On CPU it is memory-bound; on GPU it would be embarrassingly underutilized. Running one request at a time means the device spends most of its cycles waiting on memory or doing nothing.

**Pain 3: Prefill blocks decode.** In v0 a long prompt (say 2000 tokens) ties up the engine for the entire prefill. Any request that arrives during that time waits the full duration before its TTFT clock even starts — even though prefill and decode have very different compute profiles and could in principle run together.

**Pain 4: Python is the floor.** The single biggest constant factor on inference performance is "where do the matmuls actually run". NumPy on CPU is fine for v0's correctness goal, but it caps the tokens/sec at a level no amount of scheduling cleverness can recover.

Each v1 concept below targets one (or more) of these.

### 2.2 Paged attention

**Intuition.** Borrow the operating system's playbook. An OS does not give every process one giant contiguous chunk of RAM — it gives out fixed-size **pages** and maintains a per-process page table. Processes can grow without contiguous reservation, free pages get reused, and two processes can share data by pointing at the same physical page.

Paged attention does the same for the KV cache. Instead of allocating one contiguous `[max_seq_len, …]` block per request, the engine owns a global pool of fixed-size **KV blocks** (say, 16 tokens worth of K and V each). Each request has a **block table** mapping its logical positions `0, 16, 32, …` to whichever physical blocks the allocator handed out. When a request needs more space it grabs another block; when it finishes, its blocks return to the pool.

Two consequences:

- **No more over-allocation.** A request only consumes the blocks it actually fills. Internal fragmentation is at most one partly-filled block per request, regardless of `max_seq_len`. In practice this means **several times more concurrent requests** in the same memory budget.
- **Sharing is now possible.** If two requests share a prefix (e.g., the same system prompt in a chat app), their block tables can point at the *same* physical blocks for the shared positions. This is the foundation of **prefix caching** (deferred to v2, but enabled here).

**Mechanism.** A **block table** is just an array of physical block indices, indexed by logical block number. Attention is computed by gathering the relevant blocks via the block table and running the same dot products as v0 — the *math* is unchanged, only the memory layout and the indexing differ.

```
logical positions:  0..15   16..31   32..47   48..63
                      │       │        │        │
block_table[req]:   [ 7,      2,      19,      11 ]
                      │       │        │        │
                      ▼       ▼        ▼        ▼
                  ┌─────┐ ┌─────┐  ┌─────┐  ┌─────┐
   physical pool: │ blk7│ │ blk2│  │blk19│  │blk11│ ...
                  └─────┘ └─────┘  └─────┘  └─────┘
```

A real GPU implementation has a custom kernel that takes `Q`, the block table, and the global pool, and does the gather + attention without ever materializing the request's K/V as one contiguous tensor. v1 will start with a **NumPy reference implementation** of this — the goal is to understand the data structure first; the kernel is a separate concern.

**Connection.** Paged attention does *not* improve per-step latency by itself. What it does is unlock concurrency: the same memory budget now holds many more in-flight requests. That extra concurrency is what continuous batching (2.3) then exploits to raise throughput.

**Read this for the real thing:** *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023) — the vLLM paper. Read it before writing any v1 code.

### 2.3 Continuous batching

**Intuition.** v0 schedules at the granularity of *whole requests*: pick one, run it to completion, pick the next. **Continuous batching** schedules at the granularity of *one model step*: at each step, look at every in-flight request, decide which ones to advance this step, and pack them all into a *single* model invocation.

The key insight: if one request is doing decode (1 token) and another is doing decode (1 token) and a third is doing decode (1 token), you can stack their three input tokens into one batched call to `forward()` and get all three logits back in one shot. The matmuls grow from "1 token" to "B tokens" — far better device utilization for the same wall time.

It is called *continuous* because requests can join and leave the batch *between* steps. There is no fixed "batch", no padding to a common length, no waiting for the slowest member. A new request that arrives mid-step joins the next step's batch immediately — its TTFT clock starts almost instantly instead of waiting in line.

**Mechanism.** Each engine step:

1. Scheduler looks at the request pool and picks a set of requests to advance, subject to a token budget (e.g., "≤256 tokens of work this step").
2. Each selected request contributes either its remaining prefill tokens or its single decode token. The contributions are concatenated.
3. The model runner does **one** `forward()` over the concatenated input.
4. Logits are split back per request; each request samples its next token and updates its state.
5. Newly-finished requests are removed; newly-arrived ones are admitted.

Subtleties you will hit:

- **Mixing prefill and decode in the same batch.** The cleanest design lets a step contain both — a request still prefilling its prompt and another mid-decode share one model call. This is what makes TTFT short *and* throughput high simultaneously.
- **Per-token positions and per-token block tables.** Because the batch contains tokens from different requests at different positions, attention has to know, for each token, which positions in which blocks it should attend to. This is exactly what paged attention's block tables give you — which is why these two features are introduced together.
- **Chunked prefill.** If one request's prompt is enormous (say 8k tokens), processing it all in one step would starve every decode request that step. Solution: split prefill into chunks (e.g., 512 tokens) so the giant prompt cooperates with the rest of the batch instead of monopolizing it. This is a small but important refinement.

**Connection.** Continuous batching is the throughput multiplier. Paged attention is what makes it *possible* (without it, the per-token block-table indexing is impractical and the memory budget for many concurrent requests is unreachable). Together they are the two ideas at the core of vLLM.

### 2.4 Prefix caching

**Intuition.** Real workloads have massive prefix overlap. A chat app sends the same system prompt on every turn. A few-shot classifier sends the same 2000-token preamble followed by a different 20-token query. A RAG pipeline sends the same retrieved context to many follow-up questions. In v0 (and even in v1 with only paged attention) every one of those requests re-runs prefill over the shared prefix from scratch — paying the full compute cost and filling the cache with identical K/V data that the last request already computed.

**Prefix caching** notices that the K and V tensors at a given position depend *only* on the token IDs at that position and earlier. If two requests start with the same token IDs, their K/V for those positions are bit-for-bit identical. So: compute them once, store them, and let both requests point at them.

Paged attention is what makes this practical. Because the KV cache is already organized into fixed-size blocks, "sharing a prefix" is just "two requests' block tables point at the same physical blocks for the first N logical positions". No copying, no special attention kernel — the existing paged-attention path just reads those blocks for both requests.

**Mechanism.** A **prefix cache** is a hash table keyed by "sequence of token IDs that fill a block" and valued by the physical block index that holds their K/V. When a new request arrives:

1. Tokenize the prompt and split it into block-sized chunks.
2. For each chunk, in order from the start, hash the chunk (plus the hash of the previous chunk — so the key is "everything up to and including this block", not just the block in isolation — otherwise identical middle-of-prompt chunks would falsely match).
3. Look up the hash. **Hit**: wire the request's block table to that physical block and skip prefill for those positions. **Miss**: stop looking further; the suffix must be prefilled normally.
4. After prefilling the miss suffix, insert the newly-computed blocks into the hash table so the *next* request with the same prefix gets a longer hit.

Two things to get right:

- **Reference counting.** A shared block cannot be freed until every request using it is done. Each block in the pool gets a refcount; allocation increments, release decrements, block returns to the free pool at zero. The block allocator from plain paged attention already needs a refcount for this — prefix caching just adds more sources of increments.
- **Eviction.** When the pool is full and a new block is needed, evict a block whose refcount is zero (i.e., no request is currently using it, but the hash-table entry still exists). LRU is the standard policy. Evicted blocks' hash-table entries are removed so a future lookup will miss cleanly.

A useful property: prefix caching also turns *speculative retries and multi-turn chat* into cheap operations. When a user sends turn N+1 of a conversation, the first N turns are already cached — the engine only prefills the new user message, which is a huge TTFT win.

**Connection.** Prefix caching is almost pure upside *once paged attention exists*: the code changes are bounded (a hash table, a refcount, an eviction policy, a small scheduler hook) and there is no impact on the attention kernel. It dramatically reduces prefill compute in any workload with repeated prefixes, which is most real workloads. That's why it is worth doing in v1 rather than deferring.

**Read this for the real thing:** the prefix-caching section of the vLLM paper, and vLLM's `vllm/core/block_manager.py` for the production version of the refcount + hash-table dance.

### 2.5 OpenCL backend (the C++ compute layer)

**Intuition.** Up to now everything has been CPU + NumPy. NumPy is great for understanding the math but it caps performance well below what the hardware can do — even on the same CPU, hand-tuned BLAS would be much faster, and a GPU is another order of magnitude beyond that. v1 introduces the **C++ compute layer** so the engine can run its hot paths on a GPU.

The architecture from v1 onward follows a **dual-layer design**:
- **Python layer** (orchestration): Scheduler, request lifecycle, batch builder, KV cache management, tokenizer, engine loop
- **C++ layer** (compute): GEMM, attention kernels, elementwise ops, sampler — compiled for the target backend

The Python layer calls into the C++ layer via a thin **FFI/binding interface**. The model code stays unchanged when you switch backends — that's the whole point of the abstraction.

We use **OpenCL** as the first GPU backend (per the project's backend priority order: CPU → OpenCL → ROCm → CUDA → Vulkan). OpenCL is the right starting point because:

- It is **vendor-neutral** (runs on Intel, AMD, NVIDIA, Apple GPUs and even CPUs as a fallback).
- It is **lower-level than CUDA**, which is good for *learning* — you write kernels in C, you manage buffers explicitly, you launch work explicitly. Nothing is hidden.
- The concepts (kernels, work-groups, global/local memory, command queues) translate directly to CUDA, ROCm HIP, Vulkan compute, and even Metal. Learning OpenCL is learning every GPU programming model at once.

**Mechanism.** OpenCL gives you four primitives we will use repeatedly:

1. **Buffers** — chunks of GPU memory you allocate, write into from the host, and read back.
2. **Kernels** — C functions that run on the device, one instance per *work-item*. Inside a kernel, `get_global_id(0)` tells you which work-item you are.
3. **Work-groups** — groups of work-items that can share fast local memory and synchronize with a barrier. This is where most of the optimization opportunity lives.
4. **Command queues** — ordered streams of "copy this", "run that kernel", "read this back" commands the host submits to the device.

The engine grows a **backend abstraction**: a small interface (`matmul`, `softmax`, `rmsnorm`, `rope`, `attention`, `silu`, `add`, …) that has at minimum a NumPy implementation (the v0 path, kept around for testing) and an OpenCL implementation. Every model module calls into this interface instead of NumPy directly. The model code itself stays unchanged when you switch backends — that is the whole point of the abstraction.

**Designing the backend with multi-GPU in mind.** Even though v1 only runs on a single device, we know from the project roadmap that multi-GPU (tensor parallel + pipeline parallel) is coming later (in v5, on both OpenCL and ROCm). It is much cheaper to design the backend interface correctly now than to rewrite it in v5. Concretely, v1's backend abstraction should already respect the following constraints — all are cheap in v1 and expensive to retrofit:

1. **A `Device` handle is a first-class argument.** Every op, buffer allocation, and kernel launch takes an explicit device. v1 creates exactly one device and passes it everywhere; v5 will create several. No hidden "current device" globals — those are what make multi-GPU code painful.
2. **Buffers are owned by a device, not by the process.** A `Buffer` knows which device it lives on. Cross-device operations must go through an explicit `copy(src_buf, dst_device)` — no implicit migration. In v1 this is a no-op path, but the check is in place.
3. **Command queues, not blocking calls.** All ops enqueue onto a per-device command queue and return immediately; synchronization is explicit (`queue.finish()` or event waits). This is how OpenCL works natively anyway, but it is worth *using* the async-ness in v1 rather than wrapping everything in blocking calls — because in v5 the overlap between compute on one GPU and host↔device or device↔device transfers on another is where the win comes from.
4. **Model weights are a sharded structure, even if the shard count is 1.** Instead of `weights["q_proj"]` returning one tensor, it returns a list of shards (length 1 in v1). Attention/MLP code uses a tiny helper `all_gather`/`reduce_scatter` whose v1 implementation is the identity. In v5 those helpers become real collectives and the rest of the model code does not change. This is how production inference frameworks get tensor parallelism with minimal model-code churn.
5. **The model config carries a parallelism plan.** Even if the plan is `{tp=1, pp=1}`, it is a real object, and every op that would eventually care (Q/K/V projection, output projection, MLP down-projection, embedding, LM head) reads it. In v1 all the branches collapse, but the branch sites exist.

None of this adds runtime cost on v1's single device. What it buys is that v5 is mostly a matter of implementing real collectives and setting `tp > 1`, rather than surgery across every file that touches a tensor.

OpenCL is a good fit for this design because its explicit device / context / queue / buffer model *forces* you to be honest about which device everything lives on. CUDA's "current device" global would let you cheat — OpenCL will not.

**The FFI / binding layer.** Python calls into the C++ kernels through a thin wrapper. The interface is minimal:

```python
# Python side (conceptual)
class Backend:
    def gemm(self, queue, a_buf, b_buf, out_buf, M, N, K): ...
    def paged_attention(self, queue, q_buf, kv_pool, block_tables, ...): ...
    # etc.
```

The C++ side exposes C-compatible entry points that Python calls via `ctypes` or `cffi`. All the complexity (OpenCL context management, kernel compilation, buffer lifecycle) lives in C++. Python just passes handles and shapes.

**Why this split:**
- Python manages the hard systems logic (scheduling, batching, cache management) where debugging and iteration matter
- C++ runs the number-crunching where every memory access pattern matters
- The boundary between them is thin and explicit — easy to profile, easy to test both sides independently

The kernels we will write, in rough order of importance:

1. **GEMM** (matrix multiply). Almost all of inference is GEMMs of various shapes (Q/K/V projection, output projection, MLP, LM head). Even a *naive* GPU GEMM beats CPU NumPy by a wide margin; a tiled one with local memory closes most of the gap to vendor BLAS.
2. **Paged attention kernel.** Takes Q for the current step, the block table, and the K/V pool; produces attention output. This is where the v0 attention loop and the v1 paged-attention data structure meet a real GPU.
3. **Elementwise ops** — RMSNorm, RoPE, SiLU, residual adds. Simple kernels, but you need them all to keep the data on-device between operations (avoiding host↔device copies between layers is critical).
4. **Sampler** — argmax / softmax / top-k can run on-device too, so the only thing crossing the host boundary per step is the sampled token id.

**Mechanism details to learn while implementing:**

- **Memory hierarchy.** Global memory is big and slow; local memory is small and fast; private memory is per-work-item. A naive kernel reads everything from global; a good one stages reused data into local memory once per work-group. The GEMM tile size tutorial is the canonical exercise.
- **Coalesced access.** Adjacent work-items in a work-group should read adjacent memory addresses. Get this wrong and bandwidth collapses by 10–20×.
- **Occupancy.** Pick work-group sizes that keep enough work-items in flight to hide memory latency.
- **Async transfers.** The command queue is ordered but non-blocking by default — overlap host↔device copies with computation when possible.

**Connection.** OpenCL is the *compute* half of v1's improvements. Paged attention + continuous batching make *more* work available to the device per unit time; the OpenCL backend is what lets the device actually consume that work fast. Without continuous batching, the GPU would be underutilized. Without OpenCL, the scheduler would be feeding a slow CPU. Both are needed.

### 2.6 Putting the four together

The four v1 changes interact:

- **Paged attention** unlocks the *memory* ceiling on concurrency.
- **Continuous batching** unlocks the *compute* ceiling on throughput by packing many tokens into each model call.
- **Prefix caching** removes redundant prefill compute entirely for any workload with shared prefixes — the cheapest prefill is the one you don't do.
- **OpenCL** raises the *absolute* compute ceiling so each model call finishes much faster.

Removing any one of them and the others are bottlenecked. Paged attention enables prefix caching enables continuous batching's full TTFT win; OpenCL makes the remaining work fast. This is why v1 introduces them as a single milestone instead of four separate ones.

The conceptual diagram for v1 at the engine level:

```
   incoming requests ──▶ scheduler ──▶ batch builder ──▶ model runner ──▶ backend (OpenCL)
                              ▲              │                  │              │
                              │              │                  ▼              │
                              │              │            paged KV pool        │
                              │              │           (block tables)        │
                              │              ▼                                  │
                              └────── per-request state ◀──── sampled tokens ◀─┘
```

Compared to v0 the boxes are mostly the same — but `scheduler` now considers many requests per step, `model runner` accepts a heterogeneous batch, the KV cache is a *pool* with block tables, and the backend is swappable.

---

## 3. Features to implement in v1

Same template as v0: each feature has a number, a one-line description, the concept it implements, and the v0 component it replaces.

### Memory layer

| #    | Feature                           | Concept | Replaces (v0)                          |
| ---- | --------------------------------- | ------- | -------------------------------------- |
| F101 | KV block pool allocator + refcounts | 2.2, 2.4 | per-request contiguous cache (F7)     |
| F102 | Per-request block table           | 2.2     | (new)                                  |
| F103 | Paged-attention reference (NumPy) | 2.2     | naive attention path of F4             |
| F104 | Paged-attention OpenCL kernel     | 2.2 + 2.5 | the NumPy version above              |
| F104a| Prefix hash table (block hash → physical block) | 2.4 | (new)                         |
| F104b| Prefix-match-on-admission hook    | 2.4     | prefill kickoff in the scheduler       |
| F104c| LRU eviction for unreferenced blocks | 2.4  | (new)                                  |

### Scheduling layer

| #    | Feature                                      | Concept | Replaces (v0)                       |
| ---- | -------------------------------------------- | ------- | ----------------------------------- |
| F105 | Multi-request scheduler with token budget    | 2.3     | FIFO single-request scheduler (F10) |
| F106 | Batch builder (mix prefill + decode)         | 2.3     | prefill/decode split (F8)           |
| F107 | Chunked prefill                              | 2.3     | (new — no v0 equivalent)            |
| F108 | Streaming token output                       | 2.3     | (new — v0 returns the full string)  |

### Compute layer (OpenCL)

| #    | Feature                                     | Concept | Replaces (v0)                          |
| ---- | ------------------------------------------- | ------- | -------------------------------------- |
| F109 | Backend abstraction (NumPy + OpenCL), multi-GPU-aware interface | 2.5 | direct NumPy calls in model modules |
| F109a| Sharded-weights structure + identity `all_gather`/`reduce_scatter` helpers | 2.5 | (new — no-op in v1, real in v5) |
| F109b| Parallelism plan in the model config (v1 defaults `tp=1, pp=1`) | 2.5 | (new)                          |
| F110 | OpenCL device/context/queue/buffer plumbing (explicit device handles) | 2.5 | (new)                    |
| F111 | GEMM kernel (naive → tiled)                 | 2.5     | NumPy `@` in F4, F5, F6                |
| F112 | Elementwise kernels (RMSNorm, RoPE, SiLU, add) | 2.5  | NumPy implementations of F2, F3, F5    |
| F113 | On-device sampler                           | 2.5     | F9 (still keep CPU version for tests)  |
| F114 | Backend conformance test (NumPy ≡ OpenCL)   | 2.5     | (new)                                  |

### Plumbing

| #    | Feature                           | Why                                                                   |
| ---- | --------------------------------- | --------------------------------------------------------------------- |
| F115 | Throughput / TTFT benchmark       | Quantifies the v0 → v1 improvement; the whole point of this milestone |
| F116 | v1 example script                 | End-to-end demo: many concurrent prompts on the OpenCL backend        |

Build order suggestion: F101 → F102 → F103 (paged attention correct in NumPy first) → F105 → F106 → F107 (batching working on the NumPy backend) → F104a → F104b → F104c (prefix caching, still NumPy — it's a pure data-structure change) → F109 → F110 → F111 → F112 → F104 → F113 → F114 → F115 → F116. This way you never debug "is the math wrong" and "is the OpenCL kernel wrong" at the same time, and prefix caching lands on a known-correct batching layer.

---

## 4. Success criteria for v1

- Multiple concurrent requests can be in flight at once; new ones can join without waiting for old ones to finish.
- Measured throughput on a fixed workload is **substantially higher** than v0 (target: ≥10× on the same hardware once OpenCL is in).
- Measured median TTFT under concurrent load is **substantially lower** than v0 (target: bounded by the time to one chunked-prefill chunk + one batched decode step, not by other requests in the queue).
- The OpenCL backend produces token-for-token identical outputs to the NumPy backend on greedy sampling for a fixed prompt (this is what F114 enforces).
- The benchmark in F115 produces a v0-vs-v1 comparison report.

When all of that holds, v1 is "done" and we move to v2 (weight + KV-cache quantization).

---

## 5. What v1 still does NOT have

| Feature              | Why deferred                                                            | Target version |
| -------------------- | ----------------------------------------------------------------------- | -------------- |
| Weight + KV quantization | OpenCL test hardware has limited VRAM — v2 makes v1's wins actually reachable there | v2 |
| Speculative decoding | Orthogonal to backends; lands right after quantization                  | v3             |
| ROCm backend         | Second real backend; validates v1/v2/v3 abstraction on AMD hardware     | v4             |
| Multi-GPU (TP / PP)  | Hooks in v1/v2/v3/v4, real collectives in v5 (OpenCL + ROCm)            | v5             |
| CUDA backend         | Explicitly the LAST real backend priority (user preference)             | v6             |
| Vulkan backend       | Optional, not a priority                                                | v7 / optional  |

---

## 6. Further reading

- *Efficient Memory Management for Large Language Model Serving with PagedAttention* — Kwon et al., 2023 (vLLM). **Required reading before starting v1.**
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* — Yu et al., 2022 (where continuous batching was named and popularized).
- *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* — Agrawal et al., 2023 (the chunked-prefill paper).
- The Khronos OpenCL 1.2 / 2.0 specification — terse but authoritative.
- *OpenCL Programming Guide* (Munshi et al.) — book-length introduction.
- "How to Optimize a CUDA Matmul Kernel for cuBLAS-Like Performance" — Simon Boehm's blog post. CUDA syntax, but every optimization translates directly to OpenCL and is the single best GEMM tutorial available.
- vLLM source code — `vllm/core/block_manager.py` and `vllm/attention/` for the real-world version of what we are building.
