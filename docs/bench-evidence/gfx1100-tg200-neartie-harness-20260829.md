# GFX1100-TG200 — the near-tie adjudication harness: teacher-forced logprob bands over the campaign reference (2026-08-29)

## Question

T35 closed red because the one thing its merged-GEMV divergence could not show
was whether it was a near-tie: the campaign owed the teacher-forced
logprob-band ceremony (.agents/specs/rocm-m4-oracle.md) and had no harness for
it. This file lands that harness and proves it end-to-end on the current head:
reference capture, self-test, and a live negative demo against a real
reduction-order flip.

## What landed

| file | content |
|---|---|
| `examples/tg200_neartie/main.cpp` | the adjudicator — thin C-ABI client (`include/vllm.h` only, links `vllm::shared`, vllm-cli shape) |
| `examples/CMakeLists.txt` | `tg200-neartie` target |
| `tools/tg200-neartie.sh` | gpu-ctl-locked container wrapper; `@levers` expands to the adopted-lever block |
| `tools/tg200-reference.ids.i32` | the campaign reference continuation: 256 greedy token ids on the gate prompt |
| `tools/tg200-reference.body.txt` | the reference body bytes (`md5sum` == `783cea1790ae7ebc4a0105fd309a6712`) |
| `tools/tg200-reference.meta.json` | binding: model/prompt sha256, git head, env block, band, capture provenance |
| `docs/bench-evidence/gfx1100-tg200-neartie-TEMPLATE.md` | the per-lever adjudication record stub |

Mechanism: ABI v8's `vllm_logits_processor` hands a host callback the request's
logits row once per decode step, before sampling. `adjudicate` teacher-forces
the reference ids: at step n it scores
`gap = max_j logits[j] - logits[ref[n]]` on the raw row (the softmax
normalizer cancels, so this IS logprob(argmax) − logprob(ref), the m4 lane's
neartie gap), then masks every non-ref entry to −inf so the greedy walk
appends exactly the reference token. Verdict: FAIL iff any gap exceeds 500
mnats; in-band divergences are the near-ties the band exists to admit.
`capture` records the reference ids as the per-step argmax of an untouched
greedy gate run and binds the body md5.

## The exact command a future lever runs

```
tools/tg200-neartie.sh adjudicate @levers VT_MY_LEVER=1 -- \
  --model /models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt-file /repo/tg200/tools/tg200-prompt.txt \
  --ref-ids /repo/tg200/tools/tg200-reference.ids.i32 \
  --json /repo/tg200/t<NN>-my-lever-neartie.json \
  --expect-md5 783cea1790ae7ebc4a0105fd309a6712 \
  --note "T<NN> my lever ON"
```

Exit 0 = PASS (record with the TEMPLATE), 1 = FAIL (a gap over band), 3 =
runtime, 4 = integrity (the forced walk failed to reproduce the reference
body md5 — the report would be fiction, so none is issued).

## Demonstration (all legs under the gpu-ctl lock, container `rocm-dev:10.0.0`, gate prompt, 256 tokens, greedy)

### Reference capture

`capture` at HEAD `54e40a850`, adopted-lever env: body md5
`783cea1790ae7ebc4a0105fd309a6712` — the campaign reference of T34/T36,
reproduced bit-for-bit and bound via `--expect-md5`; 256 ids recorded. The
observer run's body equals the untouched run's byte-for-byte (the processor is
numerics-neutral, checked).

### Self-test (arm == reference build)

`adjudicate` of the committed ids at the same head/env: **verdict=PASS,
divergent=0, over_band=0, max_gap=0.000 mnats**, forced body md5
`783cea17…`. Raw: `agent-artifacts/tg200-neartie/selftest-neartie.json`.

### Negative demo — a real reduction-order flip, adjudicated

`VT_ATTN_DECODE_GQA4=0` (routes the d=128 f32-Q GQA decode through
`PagedAttnOnline` instead of `DecodeGqa4` — src names the differing reduction
order explicitly; harmless, quality-neutral, the lever class the ceremony
exists for). Free walk: body md5 `cca91f3f4d9921bc8fc6a8bca2db728e` — the
256-token body genuinely diverges from the reference. Forced adjudication vs
the reference ids:

| rank | pos | ref | argmax | gap_mnats |
|---|---|---|---|---|
| 1 | 205 | 4962 | 3437 | 250.000 |
| 2 | 253 | 9019 | 23926 | 125.000 |

**verdict=PASS** — divergent=2, over_band=0, max gap 250.0 of the 500-mnat
band; forced body still `783cea17…` (integrity held). This is the T35
ceremony, demonstrated live: a reduction-order lever whose token divergence is
entirely in-band near-ties now HAS its adjudication, in one command. Raw:
`agent-artifacts/tg200-neartie/negdemo-gqa4-off-neartie.json`.

## Engine finding recorded, not repaired (outside this row's authority)

On this engine build the processor's per-step token view
(`token_ids`/`n_token_ids`) arrives EMPTY at every decode step on the real
GGUF-Qwen3.5 ROCm path (five callbacks, every `n_token_ids == 0`; the
synthetic-engine capi test documents the async-feedback residual). The harness
therefore takes nothing the engine merely asserts: capture reads the ids from
the raw logits row (argmax IS the emitted token under greedy), and adjudicate
proves the walk end-to-end by binding the forced body to the reference md5.
The ABI v8 contract gap itself is upstream's to fix; it does not affect this
ceremony's soundness.

## Boundaries

- No `src/` or `include/` change; the engine surface used (ABI v8 processor,
  v1 completion) shipped in the base.
- The committed reference is bound to the adopted-lever env + this checkpoint;
  `tools/tg200-reference.meta.json` carries the full fingerprint. A lever that
  LANDS within band does NOT re-mint the reference — the pre-campaign body
  stays the reference (acceptance-gate token-identity clause).
- T35's merged-GEMV arm itself remains reverted; re-landing it now has the
  ceremony one command away, per that file's bounded next step.
