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

- **G4 and its oracle.** vLLM's own AMD `qwen4_exp` backend at the current pin
  is the natural denominator on gfx1151, and it is UNMEASURED: nobody has built
  or run it on this board. Whether Triton compiles those kernels for gfx1151 is
  the open question and it should be answered EARLY, because it decides whether
  this model can ever have a token gate on AMD hardware.
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

`ACTIVE`, 2026-09-12. Spec only, no product code. This is a scoping wave: it
opens the work, it does not do it. Next action is W1.
