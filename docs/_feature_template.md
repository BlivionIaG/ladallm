# F?? — <Feature Title>

> Version: v?  •  Concept: [<section number> <section title>](../architecture_v?.md#<anchor>)
> Depends on: <F??, F??, or "none">  •  Depended on by: <F??, F??, or "none yet">

## What this feature is

One paragraph. Plain language. Describe the feature as a *thing*, not as an implementation step.

## Why it exists

What problem does it solve? What would the engine look like without it? If the answer is "nothing would work", say so and be specific — which call site would break, which test would fail, which concurrent request would starve.

## The concept, refreshed

Short re-teach of the underlying concept, with shapes and math. Assume the reader has read the architecture doc once, some time ago. Repeat the essential formulas here so the reader does not have to flip back and forth.

Include at least:
- What inputs go in (with dtypes and shapes)
- What comes out (with dtypes and shapes)
- The transformation in between, as math or pseudocode
- Any invariant the feature must preserve

## How to implement it

Step-by-step. This is the longest section. For each step:

1. **What you are doing** in one sentence.
2. **Why** that step, in one sentence.
3. **Concrete guidance**: the file(s) to touch, the function(s) to write, the shapes to respect, the library calls available.
4. **A short code sketch** if it clarifies — but **never a full copy-pasteable solution**. The goal is that the reader understands the shape of the code well enough to write it themselves.

Prefer numbered steps over walls of prose. If a step has sub-steps, nest them.

## Edge cases and gotchas

Things that will bite you. Common bugs. Off-by-one errors. Dtype surprises. Platform differences. Performance cliffs.

## How to test it

- **Unit tests**: what to test in isolation, with expected inputs and outputs.
- **Conformance / integration tests**: how this feature is covered end-to-end.
- **Observable effects**: what *should* change in engine behavior (throughput, TTFT, memory) when this feature lands. If there is no observable effect, either the feature is too small to deserve its own doc or you are testing the wrong thing.

## Where it lives in the codebase

- `vllm_fs/<path>/<file>.py` — `<ClassOrFunction>`
- `tests/<path>/<file>.py` — the tests

Update this section when the code lands.

## Further reading

- Paper / blog / reference implementation links, with one-sentence annotations explaining what to look at.
- Links to related feature docs that the reader may want to read next.
