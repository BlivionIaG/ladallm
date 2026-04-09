# `docs/tutorial/` — Hands-on exercises

This folder holds **exercise-style tutorials**: you sit down with a blank editor, follow along, and *type the engine into existence yourself*. Unlike the architecture and feature docs (which are reference material), tutorials are linear, opinionated, and assume you are doing the work as you read.

## Available tutorials

| Tutorial | Version | What you build | Time |
| -------- | ------- | -------------- | ---- |
| [`v0-build-it-yourself.md`](v0-build-it-yourself.md) | v0 | A complete naive inference engine for a Llama-style model in pure Python + NumPy | a weekend |

## How to use a tutorial

1. Read the architecture doc for the version first (`docs/architecture_vN.md`). The tutorial assumes you understand *why* you're building each piece — it focuses on *how*.
2. Open a fresh checkout of this repo. The tutorial tells you which files to create and in which order.
3. Type the code yourself. Resist the urge to copy-paste even the snippets that look obvious — muscle memory is part of the point.
4. After each step, the tutorial gives you a "checkpoint" — a tiny test you can run to confirm the piece works in isolation. Do not skip checkpoints. Half the value of building from scratch is catching your bugs early, while the surface area is small enough to debug by inspection.
5. When you get stuck, the **per-feature docs** under `docs/vN/` are the reference. The tutorial is the path; the feature docs are the map.
