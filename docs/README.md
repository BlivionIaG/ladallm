# `docs/` layout

Documentation for `vllm-from-scratch`. The goal of everything in this folder is **learning**: each file is written so that reading it teaches a concept and, where applicable, shows how to implement it.

## Structure

```
docs/
├── README.md                   # this file
├── architecture_v0.md          # high-level plan + concepts for v0
├── architecture_v1.md          # ... v1 (paged attn, batching, prefix cache, OpenCL)
├── architecture_v2.md          # ... v2 (weight + KV quantization)
├── architecture_v3.md          # ... v3 (speculative decoding)
├── architecture_v4.md          # ... v4 (ROCm backend)
├── backend_abstraction.md      # cross-version: Python/C++ split, FFI, backend interface
├── v0/                         # per-feature implementation guides for v0
│   ├── README.md               # index of v0 features
│   ├── f01-weight-loading.md   # one deep-dive file per feature (F1..F15)
│   ├── f02-rmsnorm.md
│   └── ...
├── v1/                         # per-feature guides for v1 (F101..F116+)
├── v2/                         # per-feature guides for v2 (F201..F216+)
├── v3/                         # per-feature guides for v3 (F301..F315+)
├── v4/                         # per-feature guides for v4 (F401..F413+)
├── tutorial/                   # hands-on exercise-style build-it-yourself tutorials
└── perf/                       # benchmarks, profiler traces, accuracy tables
```

## The two kinds of doc

1. **Architecture docs** (`architecture_vN.md`) are the **plan**. They describe what a version is, what concepts it introduces, which features it contains (F-numbers), and why those features are ordered the way they are. You read them top-to-bottom when you want to understand or revisit the big picture of a version.

   **Cross-version doc**: [`backend_abstraction.md`](backend_abstraction.md) defines the dual-layer Python/C++ architecture, the FFI boundary, and the backend interface that spans v1–v4 (and beyond). Read this to understand how the Python orchestration layer talks to the C++ compute kernels.

2. **Feature docs** (`vN/fNN-slug.md`) are the **learning + implementation guides**. There is exactly one per feature listed in `architecture_vN.md`. Each one:

   - Restates what the feature is (intuition + math/mechanism, refreshed from the architecture doc).
   - Explains *how to actually implement it*, step by step, in the language and stack this project uses.
   - Walks through the shapes, edge cases, gotchas, and tests.
   - Links to the exact file(s) under `vllm_fs/` where the code lives once the feature is built.
   - Links back to the relevant architecture doc and to any prerequisite feature docs.

   The split is deliberate: the architecture doc tells you *what* and *why*, the feature doc tells you *how*. Both are teaching documents; neither is a reference manual. If you can't learn the concept by reading the feature doc, the feature doc is incomplete.

## Feature-doc template

Every feature doc should follow roughly this shape (the template lives at [`_feature_template.md`](_feature_template.md)):

```markdown
# F01 — Weight Loading

> Version: v0  •  Concept: [4.2 Tokens and embeddings](../architecture_v0.md#42-tokens-and-embeddings)
> Depends on: none  •  Depended on by: F06

## What this feature is
One-paragraph summary.

## Why it exists
What problem does it solve? What would happen without it?

## The concept, refreshed
Short re-teach of the underlying concept, with shapes and math.

## How to implement it
Step-by-step. Files to create, functions to write, shapes to respect. Short code sketches where helpful — but the goal is for the reader to *understand*, not to copy-paste.

## Edge cases and gotchas
What can go wrong? What surprised us when we built it?

## How to test it
Unit tests, conformance tests, observable end-to-end effects.

## Where it lives in the codebase
`vllm_fs/path/to/file.py` — function/class names.

## Further reading
Links, papers, reference implementations.
```

## Writing order

We do *not* write all feature docs up front. The rule is: **the feature doc is written when the feature is built** (or immediately before, as a design sketch, and finalized as the code lands). This keeps docs honest about what actually happened, and keeps us from drowning in speculative prose.

The architecture docs, by contrast, are written *before* the version they describe — they are the plan that feature docs implement.
