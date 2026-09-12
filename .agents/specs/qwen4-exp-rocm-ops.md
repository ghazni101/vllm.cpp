# Spec: qwen4_exp on ROCm — the nine operations with no arm

- Issue: `ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V`
- Row: `MODEL-MM-QWEN4-EXP` (`.agents/model-matrix.md`, `ACTIVE`). Owning
  backend row: `BACKEND-ROCM` ([#41](https://github.com/mudler/vllm.cpp/issues/41)).
- Claim: `CLAIM-MODEL-MM-QWEN4-EXP` (row-level, already held)
- Base: `97cb6964b` (`origin/main`, 2026-09-12)

## Contract

| Field | Value |
|---|---|
| Scope | IN: a `kROCM` arm for each of the nine `vt::` operations the `qwen4_exp` forward calls and ROCm does not register — `kQwen4ExpPleConv`, `kQwen4ExpPleGate`, `kQwen4ExpGatedResidual`, `kQwen4ExpGatedResidualWriteBack`, `kQwen4ExpQsaCompress`, `kQwen4ExpQsaGatherAttention`, `kRmsNormGroup`, `kIndexSelect`, `kIndexCopy` — each with its cross-device gate against the CPU oracle, and the device-fit and refusal records that change with them. OUT: the MTP head (still unimplemented on every device); the vision tower (no ROCm-specific work, and the only published GGUF is text-only); multi-sequence decode (the engine clamps `--max-num-seqs` to 1 for this model on every device); any token-exactness claim; any throughput, latency or memory number. |
| Upstream chain | vLLM is the primary oracle and at the current pin `e126687a9a` it ships a FIRST-PARTY AMD backend for this architecture: `vllm/models/qwen4_exp/amd/`. Its divergence from `nvidia/` is about 840 lines confined to the Triton op layer, with `model.py`, `model_state.py` and `mtp.py` BYTE-IDENTICAL, and it carries `amd/ops/qsa.py` (6 `@triton.jit` kernels including `_compress_qsa_groups_kernel` and `_qsa_sparse_paged_gqa_splitk_kernel`), `amd/ops/hc.py`, `amd/ple_layer.py`, `amd/hyperconnection.py` and `amd/indexer_qsa.py`. `vllm/models/qwen4_exp/__init__.py` states the support surface in its own words: "Qwen4Exp currently supports CUDA and ROCm only". So the AMD arm is a MIRROR, not an extension, and not a transliteration of the CUDA arm. |
| Our baseline | Every one of the nine has BOTH a CPU and a CUDA implementation in this tree, which is what bounds the work: `cpu_qwen4_exp{,_ple,_qsa}.cpp` and `cuda_qwen4_exp{,_ple,_qsa}.cu`, `cuda_rms_norm_group.cu`, and `kIndexSelect`/`kIndexCopy` in `cpu_ops.cpp` and `cuda_gdn.cu`. About 1709 lines of CUDA across the four qwen4_exp files. `grep -rn 'Qwen4Exp' src/vt/rocm/` returns nothing. |
| Port map | Each CUDA kernel -> a `.hip` sibling under `src/vt/rocm/`, mirroring the vLLM file structure as AGENTS.md requires: `cuda_qwen4_exp.cu` -> `rocm_qwen4_exp.hip`, `cuda_qwen4_exp_ple.cu` -> `rocm_qwen4_exp_ple.hip`, `cuda_qwen4_exp_qsa.cu` -> `rocm_qwen4_exp_qsa.hip`, `cuda_rms_norm_group.cu` -> `rocm_rms_norm_group.hip`; `kIndexSelect`/`kIndexCopy` join an existing ROCm TU rather than earning a file. **Transcribe from the CPU bodies where the CUDA one uses a primitive RDNA lacks, and from the CUDA one otherwise** — measured, the four qwen4_exp CUDA files use only `__shfl_down_sync`, `atomicAdd` and bf16/fp16 types, with no `__dp4a` and no CUDA-only intrinsic, so most port directly. Read vLLM's `amd/` Triton kernels for the ALGORITHM where ours and theirs disagree, because that is the mirror source. |
| Tests to port | vLLM has no C++ test to port. The gate is INHERITED and must not be re-authored: `tests/vt/test_backend_cross_device.cpp` already holds any registered backend to NMSE <= 5e-4 against the CPU oracle, and the sibling cases REQUIRE-prove registration rather than skipping. Each op gains a case there in that shape. The existing `tests/vllm/models/test_qwen4_exp_*_device.cpp` suites are the per-op golden surface. |
| Gates | G1 per-op cross-device NMSE on `strix:gpu0`. G2 the model LOADS and the forward completes on ROCm with `VT_OP_PROVIDER_STATS=1` showing ZERO reference-tier hits — see D2, on this board a hit is impossible, so a non-zero count means the tier was somehow installed and the run is void. G3 first tokens. G4 token-exactness against an oracle — see `## Owed`, not this row. **No throughput, latency or memory number is admissible until G4.** |
| Dependencies | The IQ4_NL ROCm GEMM (`QUANT-GGUF-IQ4_NL`, PR #3149) for the `ffn_down_exps`, and #3097's ROCm gather for the n-gram table — the gather has landed; the GEMM is reviewed PASS and awaiting merge. Hardware: `strix:gpu0`, the only AMD fleet device, reachable ONLY through an `rc` lease. No CI lane has an AMD runner, so a green CI is not evidence for any arm here. |
| Work breakdown | `W0` this spec -> `W1` the two elementwise-shaped ops (`kQwen4ExpGatedResidual`, `kQwen4ExpGatedResidualWriteBack`) plus `kRmsNormGroup`, which are the cheapest and prove the file and registration shape -> `W2` `kIndexSelect`/`kIndexCopy` -> `W3` the PLE pair -> `W4` the QSA pair, the hardest, and the one where vLLM's `amd/ops/qsa.py` is the reference rather than the CUDA arm -> `W5` first load and forward on `strix:gpu0` -> `W6` first tokens. Each wave lands with its cross-device case; no wave lands unreached. |
| Risks/decisions | R1 a missing op HARD-REFUSES on this board rather than degrading (D2), so a partial port is not a slow model, it is the same refusal with a different name. R2 wave size: nine ops in one pull request would be unreviewable, and the waves above exist to keep each reviewable. R3 the QSA pair is a gather consumer and not a mask, so a fixture under 2048 tokens of context cannot distinguish a correct port from one attending pooled keys — that bound is stated in `.agents/specs/qwen4-exp-flash-next.md` and applies here. R4 `strix:gpu0` is a single shared device and every gate here needs it. R5 no AMD runner in CI. |

## D1. Why this row exists now rather than later

Two of the three things that stopped `qwen4_exp` reaching an AMD device are
already gone. [#3097](https://github.com/mudler/vllm.cpp/pull/3097) landed the
ROCm quantized gather, so `DeviceQuantGatherSupported(kROCM)` is true and the
loader no longer refuses the device by name at `qwen4_exp_weights.cpp:665`. The
IQ4_NL keep-quant GEMM gives the 48 `ffn_down_exps` a device arm. **The weights
are now loadable and multipliable on ROCm and the forward still cannot run**,
and these nine operations are the whole of the difference.

The artifact also fits, which it did not a day ago. `strix:gpu0` was re-carved
to 96 GiB on 2026-09-11 and plain `hipMalloc` reaches at least 76 GiB there,
measured by a bounded probe under an `rc` lease; the released `UD-IQ1_S` is
67.56 GiB.

## D2. A missing op on gfx1151 is a refusal, not a slow path

**This is the single most important fact in this spec and it is
counter-intuitive**, because Strix Halo is an APU and this tree does carry a
portable CPU reference tier. It does not apply here. The chain, each link read
in the tree rather than inferred:

- `ReferenceTierEligible` (`src/vt/op_provider.cpp:906`) gates on
  `DeviceMemoryIsHostAddressable()`, deliberately NOT on `UnifiedMemory()`.
  That narrowing cost two crashes, #844 and #1435.
- `RocmBackend::DeviceMemoryIsHostAddressable()` returns `unified_memory_`
  (`rocm_backend.hip:484`).
- `ResolveMemoryPolicy` (`include/vt/rocm/rocm_arch.h:192`) computes
  `unified_memory = managed_alloc || (pageable_memory_access && integrated)`.
- gfx1151 reports `pageableMemoryAccess = 0`, MEASURED on the device, and since
  the #2511 narrowing `managed_alloc = pageable_memory_access` under `kUnset`.

So `unified_memory_` is false, the tier is never installed, and `GetOp` refuses
by name. `rocm_backend.hip:313` says as much: "a discrete AMD board never
installs the tier in the first place" — post-#2511 this APU behaves like one for
this purpose.

**W1 MEASURED THIS RATHER THAN LEAVING IT ARGUED.** On `strix:gpu0`, with
`vt::RmsNormGroup`'s ROCm registration neutralised in a scratch copy and the
kernel itself left in place, a direct call on a ROCm queue throws
`vt: no kernel for op RmsNormGroup (id 142) on device rocm (type 5), and the
portable CPU reference tier is NOT eligible: ... this ROCm device reports
hipDeviceAttributePageableMemoryAccess = 0`. The same call on the unmutated
tree returns with `GetReferenceTierHits()` unchanged. So the chain above is not
a reading of four files, it is an observed refusal on the board, and the tier is
not merely unused there — it cannot be reached. Restored byte-for-byte
afterwards.

**The coupling is worth stating on its own, because nothing records it:** the
repair that stopped the gfx1151 GPU hang (#2511), by disabling managed
allocation, also removed the CPU reference tier on that board. Spec prose
elsewhere in `.agents/specs/` still asserts host-addressability is true on
gfx1151; that is pre-#2511 and stale.

Two consequences for how this row is planned. A partial port buys nothing
runnable, so the waves are ordered to reach a forward as early as the
dependencies allow rather than to maximise ops landed. And `VT_OP_PROVIDER_STATS`
showing zero reference-tier hits is not a target to work toward here; it is a
precondition that holds by construction, and a non-zero count means something
installed a tier that should not exist and the run is void.

## D3. vLLM's AMD backend is the mirror, not the CUDA arm

AGENTS.md requires mirroring vLLM wherever it defines behavior. For this
architecture vLLM defines an AMD behavior explicitly, so "port our CUDA kernel
to HIP" is the wrong default for any op where the two disagree. The AMD backend
is pure Triton over the standard vLLM linear methods — `amd/low_latency_gemm.py`
is a 376-byte no-op that keeps them — with no AITER and no MFMA dependency,
which also makes it plausible on RDNA where MFMA does not exist at all.

Where our CUDA arm and vLLM's AMD arm agree, port ours: it is already gated
against our CPU oracle and carries this tree's conventions. Where they disagree,
vLLM wins and the spec records the difference.

## D3a. What W1 read in vLLM's AMD backend, and the three differences

D3 says the mirror decides wherever it defines behaviour, so W1 read
`vllm/models/qwen4_exp/amd/ops/hc.py` at the pin `e126687a9a` before porting.
The mirror covers all three of this wave's ops:
`_grouped_gemma_rmsnorm_kernel` is `vt::RmsNormGroup`, `_hc_silu_kernel` and
`_hc_gate_mix_kernel` are stages 2 and 3 of `vt::Qwen4ExpGatedResidual`, and
`_hc_combine_kernel` is `vt::Qwen4ExpGatedResidualWriteBack` with its injection
scaling fused in.

**The algorithms agree.** The division by `hc_count` sits inside the SiLU and
not after it, the mix is a mean over the hc branches and not a sum, the gate
multiplies the NORMED stream and not the raw one, the injection is
`2 * sigmoid(logits / hc_count)`, and the grouped norm puts eps inside the
rsqrt over the mean square. Every one of those is what our CPU oracle and our
CUDA arm already compute, so D3's "where they agree, port ours" applies and the
ROCm arms are transcriptions of `cuda_qwen4_exp.cu` and `cuda_rms_norm_group.cu`.

Three differences exist and none of them is an algorithm difference. They are
recorded here rather than resolved silently:

1. **The Gemma affine's spelling.** vLLM writes `y = x*rrms; y += y*w` and says
   in its own comment that the form exists "to lower to an FMA". Ours is
   `(x*rrms) * (1 + w)`, which is what transformers writes at
   `modeling_qwen4_exp.py:177` and what the CPU arm computes. The two agree in
   exact arithmetic and are NOT bit-identical: once `|w| < 2^-24`, `fl(1 + w)`
   rounds `w` away entirely, so the difference is a real one and not only
   associativity. **Ours is kept, and it is also the mirror.** vLLM's own
   PyTorch reference for this norm writes
   `return (normalized * (1.0 + self.weight.float())).to(input_dtype)` at
   `vllm/models/qwen4_exp/common/hyperconnection.py:87`, in the `common/` file
   that both the `nvidia/` and `amd/` layers import. Our spelling is therefore
   what upstream DEFINES as the behaviour, and the Triton `y += y*w` is an FMA
   lowering of that same behaviour rather than a second one. The measurement
   agrees: the CPU arm is the only oracle that can fail this arm — G1 is NMSE
   against it — and adopting the lowering hint would move the ROCm arm away from
   that oracle AND away from upstream's own reference while matching nothing. A
   lowering hint is not a behaviour, and D3 is about behaviour.
2. **A shared `[GROUP_DIM]` norm affine.** vLLM's kernel admits one
   (`W_SHARED`) beside the full `[DIM]` layout. `vt::RmsNormGroup`'s contract
   requires the full-row weight (`src/vt/ops.cpp:1228`), so the shared form is
   not expressible at this op at all. It is OWED, not implemented: no
   `qwen4_exp` checkpoint this tree loads stores one, and widening an op
   contract to match a capability nothing exercises would be a second, untested
   arm.
3. **The injection fusion boundary.** vLLM computes
   `2*sigmoid(logits/hc)` inside `_hc_combine_kernel`, keeping it in f32
   registers; this tree computes it in the MIXER, stores it to the caller's
   `injection` tensor, and the write-back reads it back. Where that tensor is
   bf16 the two differ by one narrowing. The boundary is this tree's op
   contract, shared by the CPU oracle and the CUDA arm, and moving it for one
   device would make the ROCm arm answer a different op. Recorded as a known
   narrowing, owed to whichever wave revisits the op split.

## D3b. Two W1 guarantees were unmeasured, and what it took to measure them

A fresh review mutated the three arms and found TWO guarantees that the kernel
comments assert and the fixtures could not see. The kernels were correct. The
fixtures were wrong, and they are recorded here because a fixture that cannot
fail is indistinguishable from one that passes, and the next reader has no other
way to learn this axis was once blind.

**The mixer could not see DIVISION 1.** `rocm_qwen4_exp.hip:257-261` says the
division by `hc_count` sits INSIDE the SiLU and that the placement is
load-bearing because SiLU is not homogeneous. Deleting it measured NMSE
`0.000490056` against the `5e-4` bar and PASSED with 2% of margin. The cause was
the fixture's projection scale, not the tolerance: at `+/-0.2`, `down(normed)`
landed at `|a| ~ 0.5` inside SiLU's near-linear part, AND `up()` then left
`sigmoid(gate)` spanning only `0.488..0.512`, so the entire low-rank branch was
a near-constant `0.5` that could not move the output whatever it computed.
`mix_down` at `+/-2.0` and `mix_up` at `+/-0.5` put the pre-activation in the
knee and open the gate to `0.27..0.81`, still nowhere near saturation. The same
deletion now measures **`0.160672`** on both `use_combine` arms and FAILS — 321x
the bar, against 0.98x before.

**The MUTATION ITSELF was a false one the first time, exactly as W1's earlier
registration mutation was.** Writing `const float a = low[i];` drops the last
use of `hc_f`, and this tree builds HIP with `-Wall -Wextra -Werror`, so
`rocm_qwen4_exp.hip:262: error: unused parameter 'hc_f' [-Werror,-Wunused-parameter]`
fails the compile and the STALE binary then passes with the unmutated number.
The measurement above was taken with
`const float a = __fmul_rn(low[i], (hc_f > 0.0f) ? 1.0f : 2.0f);`, which deletes
the division, keeps the parameter live, and multiplies by an exact `1.0f`. The
binary was sha256-proven changed (`5b6117f8..` against the baseline
`69a716d6..`) and sha256-proven restored.

**The grouped norm could not see the eps PLACEMENT.** `rocm_rms_norm_group.hip:175-178`
says eps is inside the rsqrt and is added to the MEAN SQUARE. On `O(1)` data with
`eps = 1e-6` that is unmeasurable, and both ways of breaking it survived at NMSE
`2.39943e-12`. The fixture now rescales two rows, and RMS norm returns every row
to `O(1)` whatever its input scale, so neither row distorts what the others
contribute:

| Mutation | Before | After (`gemma=false` / `true`) |
|---|---|---|
| eps added to the ROOT, `1/(sqrt(ms) + eps)` | 2.39943e-12, PASS | **0.00910165 / 0.0106572**, FAIL |
| eps added OUTSIDE the reciprocal, `1/sqrt(ms) + eps` | 2.39943e-12, PASS | **0.250049 / 0.292251**, FAIL |

Row 0 is scaled by `1e-3`, which puts its mean square at the same order as eps
and makes the first mutation shift that row by ~32%. Row 1 is scaled by `1e6`,
which puts `sqrt(ms)` at `~1.2e6` and makes the second mutation more than double
that row. The unmutated arm stays BIT-EXACT against the CPU oracle at both
polarities with those rows present, so the widening cost no margin.

**One instrument defect was found and repaired alongside.** doctest stringifies
a `const char*` operand through its `bool` overload, so every W1
`MESSAGE(... << DeviceName(dt) << ...)` recorded its NMSE beside the text `1`
and never named the device it measured; `CAPTURE` takes the same path. A
`DeviceTag()` helper returning `std::string` repairs the three W1 cases, and the
gate now prints `qwen4_exp mixer NMSE ROCM combine=true mixed = 6.98127e-16`.
The roughly forty older sites in the same file belong to other rows and are
owed.

## Tests

Red first, per op, each failing for the intended reason before the arm exists:

1. A cross-device case in `tests/vt/test_backend_cross_device.cpp` in the
   sibling shape — REQUIRE-proven registration, never `if (!OpAvailable)
   continue`, which that file states twice in comments at `:3939` and `:4048`
   and which a fresh review already corrected once on the IQ4_NL row.
2. The per-op golden suites under `tests/vllm/models/` extended to the ROCm
   device where they are device-parameterised.
3. For each op, the reachability mutation `.agents/reachability.md` requires:
   delete the `RegisterOp` line in a scratch copy and confirm the focused gate
   reds rather than silently falling back. On this board it must red by
   REFUSAL, which is also the executable proof of D2.

## Gates

- **G1 per-op.** Cross-device NMSE <= 5e-4 against the CPU oracle on
  `strix:gpu0`, every case REQUIRE-proving registration.
- **G2 load and forward.** `UD-IQ1_S` loads on `--device rocm` and the forward
  completes, with `VT_OP_PROVIDER_STATS=1` showing zero reference-tier hits.
- **G3 first tokens.** Real text, recorded as a load-and-decode result and NOT
  as a token gate.
- **G4 token-exactness.** Owed, not this row. See `## Owed`.
- **No throughput, latency or memory number is admissible from this row.**
  AGENTS.md admits no performance result from an arm whose token gate has not
  passed. The `gfx1151` token gate currently reads `TOKEN_GATE=FAIL` at 3 of 6
  on a simpler model. A baseline captured before G4 is recorded with the word
  INADMISSIBLE beside it, which is how the withdrawn 2.71x should have been
  recorded.

## Owed

- **W1 LANDS UNREACHED, and this bullet is the record AGENTS.md "Nothing lands
  dead" requires.** The three arms `kQwen4ExpGatedResidual`,
  `kQwen4ExpGatedResidualWriteBack` and `kRmsNormGroup` are registered on ROCm
  and are reached by NO production entry point on that device. The weights load
  since #3097, but `ModelRegistry::Forward` cannot complete a `qwen4_exp` step
  on `rocm`: it throws at the first of the six arms that is still missing
  (`kQwen4ExpPleConv`, `kQwen4ExpPleGate`, `kQwen4ExpQsaCompress`,
  `kQwen4ExpQsaGatherAttention`, `kIndexSelect`, `kIndexCopy`). Their only
  caller today is the cross-device suite. **The row that owns the wiring is
  `MODEL-MM-QWEN4-EXP`, this row, through waves W2 to W4**; the issue that
  tracks it is this row's own, named in the commit and pull request bodies
  rather than here, because a row-owned issue is not an owed reference. D2 is
  why the slice is staged rather than held back: on a board with no reference
  tier a partial port
  refuses by name, so there is no half-working forward to ship and no way to
  reach these three from production until the sixth arm lands.
- **The remaining `CAPTURE(DeviceName(dt))` sites in
  `tests/vt/test_backend_cross_device.cpp`**, tracked by
  `ISSUE-LOCAL-01M2A7P3C95W3PBAVT9SC6KKY5`. doctest stringifies a `const char*`
  operand through its `bool` overload and renders the device name as `1`, so a
  failing cross-device assertion cannot say which device failed. The three W1
  cases are repaired in place with a `DeviceTag()` helper; roughly forty older
  sites belong to other rows and are not swept here.
- **G4 and its oracle.** vLLM's own AMD `qwen4_exp` backend at the current pin
  is the natural denominator on gfx1151, and it is UNMEASURED: nobody has built
  or run it on this board. **The platform half of that question is already
  answered from evidence committed in this repository**, and the answer narrows
  what is still owed rather than closing it. Triton itself WORKS on gfx1151 at
  3.8.0 — `docs/bench-evidence/oracle-vllm-gfx1151-20260903/job-phase3.txt:69-70`
  and `job-phase2.txt:64` — conditional on `python3-dev`, because Triton's AMD
  driver compiles `hip_utils.c` at import time; without it the same probe reads
  `TRITON_JIT_ON_GFX1151 = FAIL ValueError`
  (`job-phase1b.txt:281`), and installing the package turned that FAIL into the
  PASS. Since vLLM's RDNA paths ARE Triton kernels, that precondition applies to
  any run of them. **What remains open is narrower:** whether vLLM's SPECIFIC
  `qwen4_exp` AMD Triton kernels compile and run on this board. That is now a
  build-and-run question, not a platform-viability one.
- **The oracle recipe is committed and must be followed rather than
  improvised.** A bare `rc` worker on `strix` carries no torch and no triton
  (probed: `ModuleNotFoundError: No module named 'torch'`), so any future oracle
  wave installs through `docs/bench-evidence/oracle-vllm-gfx1151-20260903/`
  (`phase1b.sh`, `phase2.sh`, `phase3.sh`, `phase4.sh`). That recipe also
  records that `HSA_OVERRIDE_GFX_VERSION` is never set, because it makes the
  runtime report a different device and no oracle measurement survives it.
- **The MTP head**, unimplemented on every device.
- **The vision tower**, which has no ROCm-specific work and no artifact.
- **The CPU comparison arm no longer fits `strix:gpu0`.** It peaked at 73.9 GiB
  `VmHWM` and the host side of that box is now 31 GiB after the re-carve. A
  CPU-versus-ROCm comparison must run on `thor` or `dgx`, or on-box against an
  oracle instead.

## Stop conditions

- An op cannot be expressed on RDNA without a primitive the target lacks. Stop
  and record which, rather than substituting a different algorithm silently.
- A cross-device case cannot be made to red by deleting its `RegisterOp`. The
  case is measuring nothing; stop and fix the case before landing the arm.
- `strix:gpu0` is unavailable or unhealthy. Arms stay `PENDING` on a named
  lease. Never convert an unrun gate into a pass.
- The dependency PR #3149 does not land. W1 through W4 are unaffected, but G2
  cannot run, because the `ffn_down_exps` have no device arm without it.

## Now

`ACTIVE`, 2026-09-12. **W1 landed three of the nine arms**:
`kQwen4ExpGatedResidual`, `kQwen4ExpGatedResidualWriteBack` and
`kRmsNormGroup`, in `src/vt/rocm/rocm_qwen4_exp.hip` and
`src/vt/rocm/rocm_rms_norm_group.hip`, each with a cross-device case that
REQUIRE-proves its ROCm registration. Six arms remain owed and the forward
still refuses on ROCm, because a partial port buys nothing runnable on a board
with no reference tier (D2). Next action is W2, `kIndexSelect` and
`kIndexCopy`.

**A fresh review returned FAIL on the GATES, not on the kernels**, and the
findings are repaired here. The three arms are unchanged; the two fixtures that
could not fail are widened (D3b), the eps and DIVISION-1 placements are now
pinned by measurement, and W1's staged, unreached slice is recorded under
`## Owed` as AGENTS.md requires. Re-measured on `strix:gpu0` at the widened
fixtures: write-back 0, `rmsnorm_group` 0 at both polarities, mixer injection 0,
mixer `mixed` 6.98127e-16, full ROCm cross-device suite 50/50 with 84102
assertions.

No throughput, latency or memory number is admissible from this row, and W1
took none.
