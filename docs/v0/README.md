# v0 — Feature Index

Plan: [`../architecture_v0.md`](../architecture_v0.md). Each feature below has (or will have) its own deep-dive file in this folder. Status legend: `☐` planned, `◐` in progress, `✓` done.

## Model-side

| #   | Feature                      | Doc                                                   | Status |
| --- | ---------------------------- | ----------------------------------------------------- | ------ |
| F1  | Weight loading               | [`f01-weight-loading.md`](f01-weight-loading.md)       | ✓      |
| F2  | RMSNorm                      | [`f02-rmsnorm.md`](f02-rmsnorm.md)                     | ✓      |
| F3  | RoPE                         | [`f03-rope.md`](f03-rope.md)                           | ✓      |
| F4  | Attention + KV-cache reads   | [`f04-attention.md`](f04-attention.md)                 | ✓      |
| F5  | SwiGLU MLP                   | [`f05-swiglu-mlp.md`](f05-swiglu-mlp.md)               | ✓      |
| F6  | Decoder block + full forward | [`f06-decoder-forward.md`](f06-decoder-forward.md)     | ✓      |

## Runtime

| #   | Feature                    | Doc                                                  | Status |
| --- | -------------------------- | ---------------------------------------------------- | ------ |
| F7  | Naive KV cache             | [`f07-kv-cache.md`](f07-kv-cache.md)                 | ✓      |
| F8  | Prefill vs. decode split   | [`f08-prefill-decode.md`](f08-prefill-decode.md)     | ☐      |
| F9  | Greedy sampler             | [`f09-greedy-sampler.md`](f09-greedy-sampler.md)     | ☐      |
| F10 | FIFO scheduler             | [`f10-fifo-scheduler.md`](f10-fifo-scheduler.md)     | ☐      |
| F11 | Engine top-level loop      | [`f11-engine-loop.md`](f11-engine-loop.md)           | ☐      |
| F12 | Stop conditions            | [`f12-stop-conditions.md`](f12-stop-conditions.md)   | ☐      |

## Plumbing

| #   | Feature                    | Doc                                                  | Status |
| --- | -------------------------- | ---------------------------------------------------- | ------ |
| F13 | Tokenizer wrapper          | [`f13-tokenizer.md`](f13-tokenizer.md)               | ☐      |
| F14 | End-to-end example script  | [`f14-example-script.md`](f14-example-script.md)     | ☐      |
| F15 | Smoke test                 | [`f15-smoke-test.md`](f15-smoke-test.md)             | ☐      |

## Convention

Feature docs follow [`../_feature_template.md`](../_feature_template.md) and are written when the feature is built (or just before, as a design sketch).
