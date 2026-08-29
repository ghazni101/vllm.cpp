# GFX1100-TG200 — near-tie adjudication record: <LEVER> (<DATE>)

> Copy this file for every reduction-order lever that owes the teacher-forced
> logprob-band ceremony (.agents/specs/rocm-m4-oracle.md: band <= 500 mnats,
> teacher-forced on the exact reference prefix). Fill every `<>`; attach the
> raw JSON under `agent-artifacts/tg200-neartie/`. A lever whose divergence is
> adjudicated lands ONLY with this record complete; a raw divergence count is
> never presented as a quality score (spec ## Correctness policy).

## Lever

`<VT_LEVER=1 / change summary>` — mechanism in one sentence: which reduction
order changed (kernel, geometry, accumulation), and why token identity cannot
be claimed bit-exact.

## Reference

| item | value |
|---|---|
| reference ids | `tools/tg200-reference.ids.i32` (256 tokens, sha256 `2dcda0e4…`) |
| reference body md5 | `783cea1790ae7ebc4a0105fd309a6712` |
| model | Qwen3.5-4B-Q4_K_M (sha256 `00fe7986…`) |
| prompt | `tools/tg200-prompt.txt` (sha256 `e2b801cc…`) |
| band | 500 mnats (rocm-m4-oracle.md) |
| ARM build | `<git head / commit under adjudication>` |
| ARM env | `<levers: @levers + deltas>` |

## Command (verbatim)

```
tools/tg200-neartie.sh adjudicate @levers <EXTRA_LEVER=...> -- \
  --model /models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt-file /repo/tg200/tools/tg200-prompt.txt \
  --ref-ids /repo/tg200/tools/tg200-reference.ids.i32 \
  --json /repo/tg200/<lever>-neartie.json \
  --expect-md5 783cea1790ae7ebc4a0105fd309a6712 \
  --note "<lever, arm, window>"
```

## Result

| metric | value |
|---|---|
| verdict | `<PASS / FAIL>` |
| positions | 256 |
| divergent (argmax != ref) | `<n>` |
| over band (> 500 mnats) | `<n>` |
| max gap | `<n>` mnats at position `<n>` |
| forced body md5 | `783cea1790ae7ebc4a0105fd309a6712` (integrity bound) |
| ARM free-walk body md5 | `<md5 of the arm's own unforced 256-token body>` |

Top divergent positions (worst first; copy from the tool's table):

| rank | pos | ref | argmax | gap_mnats |
|---|---|---|---|---|
| 1 | | | | |

## Disposition

`<Adopted / closed>` — if PASS: the lever's token divergence is entirely
in-band near-ties; record the perf A/B beside it and land with the lever env
documented. If FAIL: the lever changes the model's preference beyond the band
at `<n>` positions; it does not land on token coherence, whatever its tok/s.

Raw JSON: `agent-artifacts/tg200-neartie/<lever>-neartie.json`.
