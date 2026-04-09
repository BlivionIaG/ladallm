# LadaLLM

> This file is also available as `AGENTS.md` (symlink). Any coding agent working in this repo should read it before making changes.

## Project goal

This is a **learning project** to understand how vLLM and other AI inference engines work internally by reimplementing their core concepts from scratch.

The user's primary objective is **education**: gaining deep understanding of inference engine internals (paged attention, continuous batching, KV cache management, scheduling, quantization, kernel optimization, etc.). Code that runs but cannot be learned from is a failure of this project.

## Agent role and boundaries

**I do NOT implement features for you.** This is a learning project—you are here to build it yourself and deeply understand how inference engines work.

**My role is to:**
- Explain concepts, math, and mechanisms
- Guide you through design decisions  
- **Write unit tests and test fixtures** (testing is scaffolding, not the learning core)
- Review your code and suggest improvements
- Help debug when you're stuck
- Create feature documentation templates
- Research papers and reference implementations

**You do:**
- **Write the actual implementation code** (this is where learning happens)
- Make implementation decisions
- Run tests and experiments
- Learn by building

**The boundary:**
- **Implementation code** → You write it. I'll explain what to do, but you type it.
- **Tests** → I can generate these for you. Tests validate your understanding without being the learning itself.
- **Documentation** → We collaborate; you decide what concepts need deeper explanation.

When you ask me to "implement" something, I will guide you through the process, explain what needs to be done, and help you understand the concepts—but the keyboard is yours for the core code.

## Language & Architecture

The engine follows a **dual-layer architecture** similar to vLLM: Python orchestrates, C++ computes.

### Python Layer (Orchestration)
- **Scheduler**: Request lifecycle, state machine, batching decisions, admission control
- **Batch Builder**: Continuous batching, chunked prefill, mixing prefill+decode
- **KV Cache Manager**: Block tables, prefix caching, allocation/refcounting
- **Engine Loop**: Top-level step orchestration, streaming output
- **Tokenizer**: Text preprocessing, detokenization

Python is chosen for rapid iteration, readability, and debugging the complex systems logic.

### C/C++ Layer (Compute Kernels)
- **GEMM**: Matrix multiplication (Q/K/V projection, MLP, LM head)
- **Paged Attention**: The core attention kernel with block-table indexing
- **Elementwise**: RMSNorm, RoPE, SiLU, residual adds
- **Quantization**: W8A16/W4A16 dequant-fused matmul, int8/fp8 KV cache quant/dequant
- **Sampler**: Argmax, top-k/p, speculative decoding accept-reject

Kernels are compiled for the target backend and called via a thin FFI/binding layer.

### Backend Targets (in priority order)
1. **CPU** (v0: NumPy reference) → **v1+**: C fallback kernels
2. **OpenCL** (v1: first GPU backend, vendor-neutral)
3. **ROCm/HIP** (v4: AMD native)
4. **CUDA** (v6: NVIDIA native)
5. **Vulkan** (v7/optional: compute shaders)

### Version progression
- **v0**: Pure Python + NumPy (CPU only) — establishes correctness baseline
- **v1+**: Python orchestration + C++ kernels (OpenCL first) — adds performance

## Backend priority

CPU → OpenCL → ROCm → CUDA → Vulkan. CUDA has been explicitly moved to the end of the list and Vulkan is optional; see the roadmap below.

## Performance goals

Maximize: throughput, concurrency, TTFT (time to first token). Quantization (v2) exists in part because the target OpenCL hardware is VRAM-limited.

## Version roadmap

The project is built in versions. Each version has its own architecture document and its own folder of per-feature implementation guides. The current plan:

| Version | Theme                                                          | Architecture doc                             |
| ------- | -------------------------------------------------------------- | -------------------------------------------- |
| v0      | naive correct baseline (Python + NumPy, CPU)                   | [`docs/architecture_v0.md`](docs/architecture_v0.md) |
| v1      | paged attention + continuous batching + prefix caching + OpenCL | [`docs/architecture_v1.md`](docs/architecture_v1.md) |
| v2      | weight + KV-cache quantization (promoted earlier due to VRAM)  | [`docs/architecture_v2.md`](docs/architecture_v2.md) |
| v3      | speculative decoding (draft+target, propose-verify-accept)     | [`docs/architecture_v3.md`](docs/architecture_v3.md) |
| v4      | ROCm backend (HIP), ports fp16, quantized, and spec-decoding kernels | [`docs/architecture_v4.md`](docs/architecture_v4.md) |
| v5      | multi-GPU (TP + PP) on **both** OpenCL and ROCm                | (planned)                                    |
| v6      | CUDA backend (explicitly last real backend priority)           | (planned)                                    |
| v7 / optional | Vulkan backend                                           | (optional)                                   |

Each version's architecture doc is the **plan** for that version; each version's `docs/vN/` folder holds the per-feature implementation guides.

## Documentation layout

All documentation lives under `docs/`. The full convention is in [`docs/README.md`](docs/README.md); the short version:

```
docs/
├── README.md                   # the authoritative explanation of this layout
├── _feature_template.md        # template every feature doc must follow
├── architecture_v0.md          # v0 plan + concepts
├── architecture_v1.md          # v1 plan + concepts
├── architecture_v2.md          # v2 plan + concepts
├── architecture_v3.md          # v3 plan + concepts
├── architecture_v4.md          # v4 plan + concepts
├── v0/                         # per-feature deep-dive docs for v0
│   ├── README.md               # feature index with status
│   ├── f01-weight-loading.md
│   └── ...
├── v1/                         # per-feature deep-dive docs for v1
├── v2/
├── v3/
├── v4/
└── perf/                       # benchmarks, traces, accuracy tables
```

Two kinds of doc, with different jobs:

1. **Architecture docs** (`docs/architecture_vN.md`) are the **plan**. They define what a version is, which concepts it introduces, and which features (F-numbers) it contains. They are written **before** the version is built. Concepts are taught in an **Intuition → Math/Mechanism → Connections** structure so the reader can learn by reading the doc top to bottom.

2. **Feature docs** (`docs/vN/fNN-slug.md`) are the **learning + implementation guides**. Exactly one per feature listed in the corresponding architecture doc. Each follows [`docs/_feature_template.md`](docs/_feature_template.md) and covers:

   - what the feature is,
   - why it exists,
   - the concept refreshed (with shapes and math),
   - **how to implement it, step by step**,
   - edge cases and gotchas,
   - how to test it,
   - where it lives in the codebase,
   - further reading.

   Feature docs are written **when the feature is built** (or immediately before, as a design sketch, and finalized as the code lands). This keeps docs honest about what actually happened.

## Research resources

When implementing features or investigating concepts, agents can leverage the generated LLM wiki located at `~/Projects/Notes/llm-wiki/wiki`. This wiki is accessible via the `qmd` MCP server and contains indexed research on papers, techniques, and implementations relevant to inference engines.

Use `qmd` to search this wiki for:
- Paper summaries and key findings
- Implementation patterns from other projects
- Conceptual explanations of attention mechanisms, quantization methods, etc.
- Benchmark results and performance comparisons

Example queries:
- `qmd_query` with searches for "paged attention" or "continuous batching"
- `qmd_get` to retrieve specific documents by path

This wiki is maintained alongside the project and should be consulted when additional context on unfamiliar techniques is needed.

## Working agreement

### Documentation is not optional (CRITICAL)

Every feature implemented in this project **must** have:

1. A design sketch or finalized feature doc at `docs/vN/fNN-slug.md` (following the template).
2. An entry in the corresponding `docs/vN/README.md` index updated to the correct status.
3. An in-chat walkthrough explaining the feature to the user as it lands.

Code-without-doc is a bug. If you are tempted to skip the feature doc because "the code is obvious", the feature doc should still exist and say so — briefly teaching why it is obvious and linking to the concept section that makes it obvious.

### Teaching style

- **Intuition first, math second, code third.** Always.
- **Contrast with the naive approach.** When introducing a new technique, show the naive version first (even if only in prose) so the motivation for the technique is felt, not asserted.
- **Prefer clarity over cleverness.** This code is meant to be read and learned from. Optimize for understanding, then for speed.
- **Don't skip "obvious" steps.** The reader is here to learn the full picture.
- **Annotate shapes.** Every tensor op should have shapes written down at least once nearby, either in comments, docstrings, or the feature doc.
- **Cite primary sources.** When a technique comes from a paper, link the paper with a one-line annotation.

### When the user asks to reorder features or versions

1. Update the roadmap table in this file.
2. Update the deferred-features table in every affected `docs/architecture_vN.md`.
3. Update `docs/vN/README.md` feature indexes if feature numbers move.
4. Update the roadmap memory file (`project_roadmap.md`).
5. Do not delete old plans silently — if a version's scope shrinks, note what moved and where it went.

### When the user asks to implement a feature

1. Find the feature in the relevant `docs/vN/README.md` index. It should exist there already (if not, propose adding it before implementing).
2. Create or finalize its feature doc from the template **before or alongside** writing code.
3. Write the code. Annotate shapes. Validate it works.
4. Create unit tests for the feature.
5. Mark the feature `◐ in progress` or `✓ done` in the index.
6. Walk the user through what you did and what concepts it embodies.
7. **Commit the feature** (see committing guidelines below).

### Committing completed features

Every time a feature is completed and validated (before starting the next one), **it must be committed**. This keeps the git history clean and tracks progress incrementally.

**Commit includes:**
- Implementation code
- Unit tests for the feature
- Updated documentation (feature doc + index)

**Commit message style:**
- Synthetic and concise
- Straight to the point
- No verbose explanations

**Format:** `F{N}: {Feature name}`

**Examples:**
- `F1: Add weight loading (Safetensors)`
- `F2: Add RMSNorm normalization`
- `F3: Implement RoPE embedding`

Include a brief body explaining what was added and where it lives in the codebase. One commit per feature.

## Project context

- **Target OpenCL hardware is VRAM-limited.** This drives v2 (quantization) to land before v4 (ROCm). Designs that assume plenty of VRAM will not work on the actual test hardware.
- **Multi-GPU on OpenCL matters.** The user wants to test multi-GPU on OpenCL, not just on ROCm. Any v1/v2/v3/v4 work that touches the backend must preserve the multi-device-ready interface (explicit `Device` handles, no hidden globals).
- **CUDA is last.** Do not propose CUDA-first solutions or assume CUDA is available.
