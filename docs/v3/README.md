# v3 — Feature Index (Speculative Decoding)

Plan: [`../architecture_v3.md`](../architecture_v3.md). Status legend: `☐` planned, `◐` in progress, `✓` done.

## Engine plumbing

| #    | Feature                                                          | Doc                                                            | Status |
| ---- | ---------------------------------------------------------------- | -------------------------------------------------------------- | ------ |
| F301 | Two-model engine config (target + draft, both quantized)         | [`f301-two-model-config.md`](f301-two-model-config.md)         | ☐      |
| F302 | Per-request dual block tables (target + draft)                   | [`f302-dual-block-tables.md`](f302-dual-block-tables.md)       | ☐      |

## Propose / verify loop

| #    | Feature                                                          | Doc                                                            | Status |
| ---- | ---------------------------------------------------------------- | -------------------------------------------------------------- | ------ |
| F303 | Draft propose loop (`k` autoregressive steps on draft)           | [`f303-draft-propose.md`](f303-draft-propose.md)               | ☐      |
| F304 | Target verify call (one batched forward over `k` tokens)         | [`f304-target-verify.md`](f304-target-verify.md)               | ☐      |
| F305 | Greedy accept-reject inner loop                                  | [`f305-greedy-accept.md`](f305-greedy-accept.md)               | ☐      |
| F306 | Sampling accept-reject inner loop (modified rejection sampling)  | [`f306-sampling-accept.md`](f306-sampling-accept.md)           | ☐      |
| F307 | Cache truncation on reject                                       | [`f307-cache-truncate.md`](f307-cache-truncate.md)             | ☐      |
| F308 | Runner mode flag (`spec_k = 0` disables, recovers v2 behavior)   | [`f308-spec-k-flag.md`](f308-spec-k-flag.md)                   | ☐      |

## Measurement and conformance

| #    | Feature                                                          | Doc                                                            | Status |
| ---- | ---------------------------------------------------------------- | -------------------------------------------------------------- | ------ |
| F309 | Per-step acceptance counters and logging                         | [`f309-acceptance-counters.md`](f309-acceptance-counters.md)   | ☐      |
| F310 | Tokens-per-step + wall-clock-speedup metrics                     | [`f310-speedup-metrics.md`](f310-speedup-metrics.md)           | ☐      |
| F311 | Draft-overhead breakdown (time in draft vs target)               | [`f311-draft-overhead.md`](f311-draft-overhead.md)             | ☐      |
| F312 | Greedy conformance: spec ≡ non-spec, byte-identical              | [`f312-greedy-conformance.md`](f312-greedy-conformance.md)     | ☐      |
| F313 | Sampling conformance: distribution check over many seeds        | [`f313-sampling-conformance.md`](f313-sampling-conformance.md) | ☐      |

## Example and writeup

| #    | Feature                                                          | Doc                                                            | Status |
| ---- | ---------------------------------------------------------------- | -------------------------------------------------------------- | ------ |
| F314 | v3 example script                                                | [`f314-example.md`](f314-example.md)                           | ☐      |
| F315 | Spec-decoding writeup in `docs/perf/`                            | [`f315-perf-writeup.md`](f315-perf-writeup.md)                 | ☐      |

## Convention

Feature docs follow [`../_feature_template.md`](../_feature_template.md) and are written when the feature is built.
