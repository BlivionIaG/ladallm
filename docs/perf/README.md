# `docs/perf/` — Benchmarks, Traces, Accuracy Tables

This folder holds the *measurements* the project produces. Anything that answers "how fast is it?", "how much memory does it use?", or "how accurate is it?" lives here.

## What goes in here

- **Benchmark reports** — one file per version milestone. Each one lists throughput, TTFT, concurrency, and memory on a fixed workload, compared to the previous version.
- **Profiler traces** — `rocprof` outputs, OpenCL profiling dumps, flamegraphs. Check in the summary/analysis, not raw multi-megabyte dumps.
- **Accuracy tables** — perplexity and task metrics per quantization config (v2 onward).
- **VRAM budget snapshots** — output of the v2 VRAM budget calculator for representative (model, config) pairs.

## Convention

Each file should state, at the top: the hardware, the model, the exact commit, the workload, and the date. A benchmark without those four is not reproducible and not useful.
