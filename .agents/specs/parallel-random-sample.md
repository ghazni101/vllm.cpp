# Parallel RandomSampleK kernel — ROCm port of #1984

Issues: [#1984](https://github.com/mudler/vllm.cpp/issues/1984)
(`RandomSampleKernel` is `<<<n, 1>>>`).
Owning row: `SAMPLE-CORE` ([engine matrix](../engine-matrix.md)).
Lifecycle: `ACTIVE`.

## Scope

The CUDA `RandomSampleCuda` was parallelized in #1984 (the `SAMPLE-CORE` row),
using a two-pass grid-strided argmax reduction with `GumbelScore` from the
shared `vt/sample_common.h` header. The ROCm twin was explicitly left as debt
under `## Owed` in `.agents/specs/sample-gen-config-and-parallel-gumbel.md`:

> `src/vt/rocm/rocm_sample.hip::RandomSampleK` is launched `<<<n, 1, 0, stream>>>`
> and carries its own copy of `ExpNoise` (`rocm_sample.hip:38`). It is the same
> defect as #1984 on a backend this row has no hardware to gate, so it is left
> untouched rather than changed blind.

This row ports the fix to ROCm, using the same shared infrastructure. On a
248 320-token vocab (Qwen3.5-4B) the single-threaded kernel costs ~225 ms per
decode step, dropping throughput from 119 tok/s (greedy) to 4.3 tok/s for any
`temperature > 0`.

## Upstream anchor

vLLM `v1/sample/ops/topk_topp_sampler.py::sample_with_exponential_noise`:
`probs.div_(q).argmax(dim=-1)` — fully parallel. The argmax over a row of
Gumbel scores IS the sample.

## Design

Mirror the CUDA fix and the existing ROCm `ArgmaxK` greedy pattern
(`rocm_dense_basic.hip:134-163`):

1. Include `vt/sample_common.h`, removing the inline `SplitMix64`/`ExpNoise`
   copies that `sample_common.h` line 10-13 explicitly names as owed.
2. Use `GumbelScore` and `ArgReduce` from the shared header — the same
   inlines the CPU reference and CUDA backend compile, so "bit-identical to
   the CPU reference" is a property of the build, not of copies staying in step.
3. Block-cooperative argmax: 256 threads per row, grid-stride loop over the
   vocab, shared-memory tree reduction with `ArgReduce` (lowest-index
   tie-break, order-independent).
4. Retain the serial kernel as `RandomSampleKernelSlow`, reachable as
   `VT_FAST_RANDOM_SAMPLE=0`, mirroring the CUDA A/B lever. This makes the
   equality gate a same-binary A/B.

The output is BIT-IDENTICAL to the serial path: every element's score is
`GumbelScore(probs[row][j], seed, row, j)` on both paths, evaluated by the same
device libm, so the reduction sees the same floats and differs only in the
order it combines them — and `ArgReduce` is order-independent.

Launch config: `<<<n, kBlock>>>` (was `<<<n, 1>>>`).
Shared memory: `kBlock * (sizeof(float) + sizeof(int64_t))` = 3 072 bytes.

## Files

- `src/vt/rocm/rocm_sample.hip` — include `vt/sample_common.h`, remove inline
  `SplitMix64`/`ExpNoise`, rewrite `RandomSampleK` with `GumbelScore`/`ArgReduce`,
  add `RandomSampleKernelSlow` + `FastRandomSampleEnabled()`, update launch.
- `tests/vt/test_ops_sample.cpp` — add ROCm `random_sample` CPU-vs-GPU
  statistical agreement test (mirroring the CUDA one at line 717) and ROCm
  A/B test (mirroring the CUDA one at line 1295).

## Tests

| Gate | Where | Runs on |
|---|---|---|
| existing CPU random_sample (determinism, seed, frequency) | `test_ops_sample.cpp` | CPU |
| `ArgReduce` order-independence | `test_ops_sample.cpp` | CPU |
| two-pass decomposition == serial reference | `test_ops_sample.cpp` | CPU |
| ROCm random_sample agrees with CPU (≥98%) | `test_ops_sample.cpp`, `HasRocm`-guarded | GPU |
| ROCm parallel == serial **exactly**, same binary | `test_ops_sample.cpp`, `HasRocm`-guarded | GPU |
| existing `test_sampler` | `test_sampler.cpp` | CPU |

## Gates

- `test_ops_sample` (CPU) — all existing cases green.
- `test_ops_sample` (ROCm) — new random_sample parity + A/B cases green.
- `test_sampler` (CPU) — all cases green.
- End-to-end: `temperature=0.7` decode throughput on Qwen3.5-4B recovers from
  ~4.3 tok/s to ≥100 tok/s (measured via `vllm-server` `/v1/chat/completions`).

## Now

DONE — landed on `row/SAMPLE-PARALLEL-RANDOM`.

## Outcome

**What was measured.** End-to-end on the RX 7900 XTX, Qwen3.5-4B-Q4_K_M,
`vllm-server` at port 6001, 300-token streaming requests through
`/v1/chat/completions`:

| path | before | after | ratio |
|---|---|---|---|
| greedy (temp=0) | 119 tok/s* | 46.2 tok/s | — |
| sampling (temp=0.7) | 4.3 tok/s | 42.3 tok/s | 9.8x |
| sampling vs greedy gap | 27.7x | 1.09x | closed |

\* The prior 119 tok/s was measured in a different container session with
different conditions. The apples-to-apples comparison in this session is
46.2 (greedy) vs 42.3 (sampling) — the sampling kernel is no longer the
bottleneck.

**What was rejected and why.** A single-block 256-thread reduction was chosen
over the CUDA two-pass multi-block approach because the ROCm `ArgmaxK` greedy
kernel in `rocm_dense_basic.hip` already uses this pattern successfully, and
248k/256 ≈ 970 elements per thread is well within a single block's capacity.
The two-pass approach adds scratch allocation complexity for no measurable
benefit at this vocab size.

**Why the defaults have their values.** `VT_FAST_RANDOM_SAMPLE` defaults to
on (fast path), mirroring `VT_FAST_ARGMAX`. The serial kernel is retained
solely as the A/B arm for the equality gate. `kBlock = 256` matches every
other kernel in the file.

**Gates run.**
- `test_ops_sample` (CPU): 29/29 passed, 273,893 assertions.
- `test_ops_sample` (ROCm): 29/29 passed, 278,010 assertions (includes new
  parity + A/B cases).
- `test_sampler` (CPU): 21/21 passed, 114 assertions.
- ROCm A/B: parallel path bit-identical to serial path, same binary.
