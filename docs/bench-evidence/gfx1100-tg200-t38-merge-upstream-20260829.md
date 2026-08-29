# GFX1100-TG200 T38 — upstream sync (74 commits) + post-merge correctness gates (2026-08-29)

## Question

Does `row/GFX1100-TG200` stay correct after syncing `upstream/main`
(`e551cf8e4` -> `3015aad08`, 74 commits) into the campaign branch, and is
the campaign reference preserved bit-for-bit under the adopted-lever block?

## Change

- `4262858c4` merge upstream/main into row/GFX1100-TG200. Three content
  conflicts, all in the op seam, all additive: `kPermuteVHeads` (ours, T25)
  vs `kCastF16` (upstream, QUANT-EXL3 W1a #2181) inserted at the same
  anchor of `include/vt/ops.h`, `src/vt/cpu/cpu_ops.cpp`, `src/vt/ops.cpp`.
  Resolved keep-both. No campaign lever rebased; evidence-commit SHAs intact.
- `69d4243b1` fix: the shared closer had been unified by git across both
  conflict sides, leaving `PermuteVHeadsKernel` and `PermuteVHeads`
  unclosed; adds the two missing `}`. Caught by the build (red-first).

## Inherited upstream break (NOT introduced by the merge)

`tests/vllm/model_executor/test_placed_moe_roundtrip.cpp` references
`vllm::RunMoeBlockPlaced`, which exists NOWHERE on upstream/main either
(git grep across include/ and src/: only the test, the spec, and the issue
index). Upstream's own `hybrid-placement.md` (:442-444) records that W3c
removed the helper as dead code; the W3b test (:550 evidence claim) was
left behind. Upstream `1029f201e` lineage does not compile its test suite;
the merged branch inherits that verbatim. Build uses `-k 0`; this is the
ONLY failing target. Fixing it here would mean deleting upstream's test,
which is upstream's call, not this row's.

## Gates (container `rocm-dev:10.0.0`, HIP 7.15, gpu-ctl lock held)

Build: `cmake --build build-hip-docker -j 16 -- -k 0` in-container over
`/repo/tg200` (the configured source root), Release, `-Wall -Wextra
-Werror` — all targets green except the inherited break above.

`ctest -R 'rocm|quant' --output-on-failure`: 24 tests, 22 passed, 2 failed:

- `test_rocm_quant_dot`: 841/841 assertions, 2 case-level throws at
  `test_rocm_quant_dot.cpp:1603` (`matmul_bt_quant`, iq2_xxs) and `:1699`
  (`matmul_bt_quant_grouped`, iq2_xxs) — the documented UNPORTED-dtype
  throws (ported: Q8_0/Q4_K/Q5_K/Q6_K; owed: Q4_0/Q2_K/Q3_K/IQ*), unchanged.
- `test_gguf_keep_quant`: 34/42 cases, 18 assertion failures, ALL in
  `RouteGgufTensor(...) == kKeepQuant` host-routing checks for
  IQ2_XS (id 17) / IQ4_XS (id 23) / IQ4_NL (20u) roles. Inherited from
  upstream verbatim: `git diff upstream/main HEAD` on
  `gguf_keep_quant.{h,cpp}` is EMPTY, and `8e2f56cb1..upstream/main`
  changed ONLY the test file (+101/-34), not the router — upstream landed
  stricter expectations without the router change.

`test_rocm_prefill_tile` PASSED (the 720/720 byte-identity gate);
`test_rocm_skinny_f32`, `test_rocm_fp8_kv_cache`, `test_gemma4_rocm_fp8_seams`
passed. Both reds are the documented pre-existing set, now traced to their
upstream origin rather than assumed.

## Acceptance identity (the row's own token gate)

Command: `build-hip-docker/examples/vllm-cli --model
/models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf --prompt "$(cat
tools/tg200-prompt.txt)" --max-tokens 256 --temperature 0 --seed 0
--repeat 5`, all 15 adopted levers exported (verified `env | grep -c ^VT_`
= 15 in-container), gpu-ctl held, single process.

- 109 prompt tokens, 256 completion tokens, finish=length on every rep.
- body md5 `a0fa1c4aa8cc5de086006111dad7a7bf` ==
  `tools/tg200-reference.body.txt`; `diff` byte-identical. The re-minted
  reference survives the 74-commit merge bit-for-bit.
- tok/s by rep: 86.19 (warm-up) / 95.19 / 95.21 / 95.05 / 95.16; median of
  5 = 95.05. CAVEAT: window was not pristine-idle (loadavg 4.3 decaying
  from the build; T37's windows started at 0.42). The identity claim is
  load-independent; the +11% vs the documented 85.8 position is an
  OBSERVATION, not a claim — it owes a clean idle-window A/B before any
  attribution (candidate: upstream decode-path changes riding the merge).

## Verdict

Correctness PASS: focused gates at the documented baseline, campaign
reference byte-identical, branch synced to upstream tip `3015aad08` at
`69d4243b1`. Owed upstream: the `test_placed_moe_roundtrip` /
`RunMoeBlockPlaced` removal belongs on mudler/vllm.cpp, not here.
