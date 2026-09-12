# Spec: QUANT-GGUF-IQ4_NL — native IQ4_NL compute on ROCm and CUDA

- Issue: `ISSUE-LOCAL-01M29ECVDVH5A89TBKA8K72YRD`
- Row: `QUANT-GGUF-IQ4_NL` (`.agents/quantization-matrix.md`). Owning feature
  rows: `BACKEND-ROCM` ([#41](https://github.com/mudler/vllm.cpp/issues/41))
  and `QUANT-CUDA-GATES`.
- Claim: `CLAIM-QUANT-GGUF-IQ4_NL`
- Base: `18f39771c` (`origin/main`, 2026-09-12)
- Pull request shape: **one pull request** carrying the spec and the
  implementation, spec committed first. Developer decision, 2026-09-11.

## Contract

| Field | Value |
|---|---|
| Scope | IN: the IQ4_NL keep-quant dot and device admission on ROCm; the CUDA Q8_0-activation GEMM variant plus `DotIQ4_NL` and its `IsCudaKeepQuantSupported` arm; the `QUANT-GGUF-IQ4_NL` record repair and the `strix:gpu0` rows in `.agents/environment.md`. OUT: the ROCm quantized GATHER, which #3097 landed on 2026-09-12; every other missing ROCm format (IQ2_XS, IQ1_M, Q5_0, Q4_0, IQ3_S, MXFP4 -- #1940); any tensor-core tile for IQ4_NL; the seven `qwen4_exp` operations with no ROCm arm; every throughput, latency and memory number. |
| Upstream chain | Numerical authority is the pinned [`llama-cpp`](../oracles/llama-cpp.md) oracle at `10bf611e5` (`b10451`), because vLLM defines no GGUF k-quant dot kernel and therefore has nothing to mirror here: `ggml/src/ggml-cpu/quants.c:1254` `ggml_vec_dot_iq4_nl_q8_0_generic` (the dot), `ggml/src/ggml-common.h:447-452` `block_iq4_nl` (18 B, `QK4_NL = 32`), `ggml/src/ggml-common.h:1120` `kvalues_iq4nl` (the 16-entry codebook), `ggml/src/ggml-cpu/ggml-cpu.c:379-384` (`.vec_dot_type = GGML_TYPE_Q8_0`, the activation pairing, read off upstream and NOT inherited from IQ4_XS which pairs `Q8_K`). |
| Our baseline | Reader, dequantizer and CPU dot all landed under #1989 and are the transcription source: `gguf_reader.cpp` case 20; `src/vt/cpu/cpu_quant_dequant.cpp:92` `DequantIQ4_NL`; `src/vt/cpu/cpu_quant_dot.cpp:140` `VecDotIQ4_NLQ8_0`; `src/vt/cpu/cpu_quant_blocks.h:62` `BlockIQ4_NL`; traits `src/vt/cpu/cpu_quant_traits.cpp:52` pairing `kIQ4_NL` -> `kQ8_0`. Absent everywhere else: ten `WType` entries in `rocm_quant_dot.hip:580` exclude it, `rocm_grouped_gemm.hip:1783` and `:1855` throw naming it, `IsCudaKeepQuantSupported` omits it, and `kEmbeddingQuant` is registered for `kCPU` and `kCUDA` only. |
| Port map | `cpu_quant_dot.cpp:140` `VecDotIQ4_NLQ8_0` -> a `DotIQ4_NL` device body in `src/vt/rocm/rocm_grouped_gemm.hip` (scalar, transcribed from the CPU body and NOT from CUDA: gfx1100 has no hardware `__dp4a`, and the codebook lookup is per nibble so there is nothing for a four-way byte dot to multiply until the values are gathered) feeding TWO kernels, `IQ4NLGemmK` (single matrix) and **`GroupedIQ4NLK` (expert towers, the load-bearing one, because the 48 `ffn_down_exps` reach the GROUPED provider)**, plus a matching `DotIQ4_NL` in `src/vt/cuda/cuda_quant_dot.cu`. `kvalues_iq4nl` -> the ROCm IQ codebook header #3029 introduces, sealed. `cpu_ops.cpp:4282` `kEmbeddingQuant` registration -> a `kROCM` registration decoding IQ4_NL block rows, mirroring the CUDA codec `cuda_quant_dequant.cuh:156` `DqIQ4_NL`. The ROCm Q8_0 activation quantizer already exists (`rocm_grouped_gemm.hip` `QuantizeQ8_0Kernel`); CUDA needs one. |
| Tests to port | vLLM has no test to port here, because it has no GGUF k-quant dot. Goldens come from the pinned oracle's OWN kernel over real `UD-IQ1_S` checkpoint bytes, in the shape `tests/vllm/test_gguf_dequant.cpp:528` already uses for the IQ4_NL dequantizer and `tests/vt/test_ops_quant_dot.cpp:869` uses for the IQ4_XS dot: per block and in total, never a synthetic tensor. Newly authored: the IQ4_NL cases in `tests/vt/test_rocm_quant_dot.cpp`, `tests/vt/test_ops_quant_traits.cpp`, `tests/vllm/test_gguf_keep_quant.cpp`, `tests/vt/test_backend_cross_device.cpp`, and a device gather case. |
| Gates | See `## Gates`. G1 unit suites on CPU, on `strix:gpu0` for the ROCm arms and on a CUDA box for arm 3; G2 admission of `UD-IQ1_S` with zero reference-tier hits for the IQ4_NL GEMM and gather. NO throughput, latency or memory number is admissible from this row. |
| Dependencies | **None blocking.** #3029 was recorded as a hard dependency and is not one: #3097 landed `src/vt/rocm/rocm_quant_iq_tables.h` on `main` carrying `d_kvalues_iq4nl[16]` at `:1198`. #3029 remains a CONFLICT surface (eight files, measured by `git merge-tree`), and its two `sanitize-cpu` reds are a repository-wide pre-existing failure in `dots3` tests it does not touch. Hardware: `strix:gpu0` (`gfx1151`), a fleet device reachable only through an `rc` lease. No CI lane has an AMD runner, so a green CI is not evidence for arms 1 and 2. |
| Work breakdown | `W0` this spec -> `W1` ROCm `DotIQ4_NL` + `WType` entry + dispatch + device admission -> `W2` CUDA Q8_0-activation variant + `DotIQ4_NL` + `IsCudaKeepQuantSupported` arm -> `W3` record repair. The gather wave this spec originally planned is struck: #3097 landed it. W1 alone is what stands between `UD-IQ1_S` and a fully device-resident weight set on gfx1151. |
| Risks/decisions | R1 a codebook decoder pointed at a sibling table still decodes (IQ4_NL shares its codebook with IQ4_XS, so this is live); R2 reassociation passes a small golden; R3 the CUDA Q8_0-activation variant is a new dispatch path beside the Q8_K one it can regress; R4 `main` moves and #3029 also edits `.agents/quantization-matrix.md`; R5 no AMD runner in CI. D1 keep upstream's association order; D2 IQ4_NL is a 32-element block on a Q8_0 activation, which is why arm 3 is a variant rather than a table entry; D3 the gather and the dot are both required and are not the same code. Detail in `## Design` and `## Risks`. |

## Why this row exists, in one paragraph

IQ4_NL is the one encoding that every published `Qwen3.8-Flash-Next` GGUF
stores its two largest structures in, and it is the only encoding in those
files that no device can compute on. The reader opens it, the CPU dequantizes
it and the CPU dots it. Every accelerator refuses it. On ROCm the refusal is
terminal because the backend is discrete and cannot fall back to a host kernel
that would follow device pointers.

## Scope

Three arms and one record repair, in one change:

1. **ROCm keep-quant dot.** An `IQ4_NL` arm for the ROCm quantized GEMM, with
   device admission, so `MatmulBTQuant` and the grouped expert GEMM stop
   throwing on it.
2. ~~**ROCm quantized gather.**~~ **LANDED ELSEWHERE ON 2026-09-12, while this
   spec was being written, and struck rather than deleted so the change of
   scope is auditable.**
   [#3097](https://github.com/mudler/vllm.cpp/pull/3097)
   (`feat(BACKEND-ROCM-QUANT-GATHER)`, `82de418e8`) registers
   `OpId::kEmbeddingQuant` for `kROCM` at `src/vt/rocm/rocm_ops.hip:191` and
   lists `X(kIQ4_NL, DqIQ4_NL)` in `src/vt/rocm/rocm_embedding_quant.hip:88`.
   `DeviceQuantGatherSupported` is exactly
   `vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev)`
   (`gguf_keep_quant.cpp:220`), so it is now true for ROCm and the
   `qwen4_exp_weights.cpp:665` refusal no longer fires on this device. This row
   does not reimplement it.
3. **CUDA keep-quant dot.** The Q8_0-activation GEMM variant the CUDA file is
   already written against, plus `DotIQ4_NL` and its entry in
   `IsCudaKeepQuantSupported`.
4. **Record reconciliation** of the `QUANT-GGUF-IQ4_NL` row, which understates
   the tree: `R`, `M` and CPU `C` all landed under #1989 while the row still
   read `INVENTORIED` with every stage unset. It moves to `ACTIVE`.

### Out of scope, named rather than dropped

- **IQ2_XS, IQ1_M, Q5_0, Q4_0, IQ3_S and MXFP4 on ROCm.** Tracked by
  [#1940](https://github.com/mudler/vllm.cpp/issues/1940). `DotMXFP4` already
  exists on CUDA and only waits on arm 3; wiring it is still a separate row
  because it changes what `IsCudaKeepQuantSupported` admits.
- **Any tensor-core tile for IQ4_NL.** The scalar tier lands first, exactly as
  `KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT` decided for IQ4_XS. A 16-entry codebook
  format does not map onto the linear-scale k-quant WMMA tile.
- **The seven `qwen4_exp` operations that have no ROCm arm.** See `## Owed`.
  This row makes the model's weights computable on ROCm. It does not make the
  model run, and no part of this spec should be read as claiming that.
- **Any throughput, latency or memory number.** See `## Gates`.

## Dependency on #3029: DOWNGRADED on 2026-09-12

**This spec first recorded #3029 as a hard dependency. It is not one any more,
and the correction is kept rather than quietly applied.** The premise was that
`src/vt/rocm/rocm_quant_iq_tables.h` existed only on that branch, so IQ4_NL's
codebook had nowhere to live. That was true when written and false hours later:
#3097 landed the header on `main`, and it carries `d_kvalues_iq4nl[16]` at
`:1198`, transcribed from stock `ggml-common.h:1120`. The codebook this row
needs is therefore already on `main`.

What remains is a **conflict surface, not a dependency**. #3029 edits
`src/vt/rocm/rocm_grouped_gemm.hip` and `.agents/quantization-matrix.md`, which
this row also edits. Measured with `git merge-tree` against `origin/main`
`18f39771c` on 2026-09-12, #3029 carries **eight conflicts**:
`.agents/quantization-matrix.md`,
`.agents/specs/kernel-quant-ciq-gemm-rocm-iquant.md`, `docs/USAGE.md`,
`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp`,
`src/vt/rocm/rocm_grouped_gemm.hip`, `src/vt/rocm/rocm_quant_iq_tables.h`
(add/add, because #3097 landed the same path), and two test files.

**Its two `sanitize-cpu` reds are NOT its own**, and that was measured rather
than assumed: the failing cases are `test_dots3_note_vision`,
`test_dots3_note_audio` and `test_openai_api_server_dots3_mm_forward`, #3029
touches no `dots3` file, and the same two lanes are red on
[#3140](https://github.com/mudler/vllm.cpp/pull/3140),
[#3133](https://github.com/mudler/vllm.cpp/pull/3133) and
[#3036](https://github.com/mudler/vllm.cpp/pull/3036) as well. They are a
repository-wide pre-existing red and they do not belong to this row either.
Its `windows-msvc-*` reds are the known
[#584](https://github.com/mudler/vllm.cpp/issues/584) crash baseline.

## Upstream anchors

Primary oracle is vLLM wherever it defines behavior. It does not define a GGUF
k-quant dot kernel, so the numerical authority here is the pinned llama.cpp
oracle [`llama-cpp`](../oracles/llama-cpp.md) at `10bf611e5` (`b10451`), exactly
as every sibling format in this tree used.

| What | Where |
|---|---|
| `ggml_vec_dot_iq4_nl_q8_0_generic` | `b10451` `ggml/src/ggml-cpu/quants.c:1254` |
| `block_iq4_nl` layout | `b10451` `ggml/src/ggml-common.h:447-452` |
| `kvalues_iq4nl` codebook | `b10451` `ggml/src/ggml-common.h:1120` |
| activation pairing | `b10451` `ggml/src/ggml-cpu/ggml-cpu.c:379-384`, `.vec_dot_type = GGML_TYPE_Q8_0` |

Our existing ports, which are the transcription source for the device arms:

| What | Where |
|---|---|
| CPU dot | `src/vt/cpu/cpu_quant_dot.cpp:140` `VecDotIQ4_NLQ8_0` |
| CPU block | `src/vt/cpu/cpu_quant_blocks.h:62` `BlockIQ4_NL`, 18 bytes |
| CPU dequant | `src/vt/cpu/cpu_quant_dequant.cpp:92` `DequantIQ4_NL` |
| reader trait | `src/vllm/model_executor/model_loader/gguf_reader.cpp` case 20 |
| CUDA gather codec | `src/vt/cuda/cuda_quant_dequant.cuh:156` `DqIQ4_NL` |

**Transcribe from the CPU bodies, not from CUDA.** That is what
`rocm_quant_dot.hip:226` records for the existing formats, and its reason still
holds: gfx1100 has no `__dp4a`, so the CUDA integer-core shapes do not port.

## Design

### D1. The association order is load-bearing

`VecDotIQ4_NLQ8_0` forms the scale product **before** folding in the integer
sum, `d * (sumi1 + sumi2)`, which is the opposite association from the adjacent
`q4_0` kernel. `cpu_quant_dot.cpp:136-139` already says so in a comment, and
says why: it is what makes the GEMM bit-reproducible against upstream. Both
device arms keep that order. A kernel that reassociates is not this kernel, and
the goldens below will not detect the difference on small inputs, which is
precisely why this is stated as a design decision rather than left to review.

### D2. IQ4_NL is a 32-element block on a Q8_0 activation

This is the structural fact that makes arm 3 a variant rather than a table
entry. The resident device path on both backends quantizes activations to
`BlockQ8_K` over 256-element super-blocks. IQ4_NL pairs with `BlockQ8_0` over
32. **The two backends are asymmetric here, and W1 measured the asymmetry rather
than inheriting this section's first guess.** ROCm turned out to carry a
COMPLETE Q8_0-activation path already: `QuantizeQ8_0K` beside
`QuantizeQ8KKernel`, `DotQ8_0(BlockQ8_0*, BlockQ8_0*)`, and both a single-matrix
and a grouped GEMM consuming `const BlockQ8_0* act`. The ROCm arm is therefore a
new dot slotted into existing machinery, not new machinery, and it is
correspondingly small.

CUDA has no such path, and `cuda_quant_dot.cu:695-698` records the gap in its
own words, marking `DotMXFP4` `[[maybe_unused]]` because it "awaits the
Q8_0-activation GEMM variant above". The CUDA arm is that variant. Q5_0, Q4_0
and MXFP4 all queue behind it, which is this row's leverage and also why the
CUDA half is the larger of the two.

### D3. The gather and the dot are both required, and they are not the same code

`unsloth/Qwen3.8-Flash-Next-GGUF` stores **91,465,564,160 elements** in IQ4_NL
in every published quant. Two structures account for it: the 20M-entry n-gram
embedding table, which is read by `kEmbeddingQuant` and never multiplied, and
the 48 `ffn_down_exps`, which are multiplied and never gathered. Landing only
the dot leaves `DeviceQuantGatherSupported(kROCM)` false and the loader still
refuses at `qwen4_exp_weights.cpp:665`, before any tensor I/O. Landing only the
gather leaves the expert GEMM throwing at `rocm_grouped_gemm.hip:1783`. This is
why the row's scope is both.

### D4. Reachability

Per `AGENTS.md` "Nothing lands dead", each arm enters through a production
entry point at this commit, and the smallest failing test enters the same way:

| Arm | Production entry | Mutation that must red the gate |
|---|---|---|
| ROCm dot | `vt::MatmulBTQuant` / `kMatmulBTQuantGrouped` dispatch | delete the `IQ4_NL` dispatch case; the focused gate must fail rather than fall back |
| CUDA dot | `IsCudaKeepQuantSupported` plus the Q8_0-activation GEMM | delete the `IsCudaKeepQuantSupported` arm; the keep-quant routing case must red |

A unit test that constructs the kernel by hand proves the arithmetic and not
the capability. Each arm therefore also carries a routing assertion by name, in
the shape `tests/vllm/test_gguf_keep_quant.cpp` already uses for IQ3_S and
IQ4_XS.

## Evidence measured for this spec

All read on 2026-09-11, recorded here because three of them contradict what the
tree currently says.

### E1. What the shipped artifacts actually store

Read from the three GGUF shard headers of each quant over HTTP range requests,
not from prose. `general.architecture = qwen4exp`, `split.tensors.count = 1224`.

| Quant | Bytes | Formats with no ROCm arm |
|---|---|---|
| `UD-IQ1_S` | 72,546,461,344 (67.56 GiB) | **IQ4_NL only** |
| `UD-IQ1_M` | 74,538,755,776 (69.42 GiB) | IQ4_NL, IQ1_M |
| `UD-Q2_K_XL` | 78,869,128,864 (73.45 GiB) | IQ4_NL, IQ2_XS |

`UD-Q2_K_XL` type histogram: IQ4_NL 49 tensors / 91,465,564,160 elements;
IQ2_XS 94 / 78,852,915,200; Q5_K 189 / 2,823,946,240; IQ3_XXS 2 /
1,677,721,600; Q8_0 244 / 747,110,400; Q6_K 64 / 642,908,160; Q4_K 1 /
635,699,200; F32 and BF16 581 / 98,034,560. The two formats ROCm lacks are
about 97 percent of its weight elements.

`UD-IQ1_S` is therefore the cheapest ROCm target of the family, on two
independent counts: one missing format instead of two, and the smallest file.

### E2. `strix:gpu0` carve, measured under an `rc` lease

Job `a8111ff8-3ce8-42f6-9034-36bdd2cacfe4`, `rc run -d strix:gpu0`.

| Probe | Value |
|---|---|
| `mem_info_vram_total` | 103,079,215,104 B = **96.00 GiB** |
| `mem_info_vram_used` | 154,816,512 B, idle |
| `mem_info_gtt_total` | 16,635,248,640 B = 15.49 GiB |
| host RAM total / available | 33,270,497,280 / 29,304,037,376 B |
| device | `gfx1151`, AMD RYZEN AI MAX+ 395 w/ Radeon 8060S |

### E3. `hipMallocManaged` on gfx1151 is bounded by HOST memory, not the carve

Job `c30dc437-bf12-4a23-ab8b-89b88fe767dd`, same box, HIP 7.2.53211, a bounded
probe that stops at the first failure and never walks past its target.

```text
hipMemGetInfo: free=95.848 GiB total=96.000 GiB
integrated=1 managedMemory=1 pageableMemoryAccess=0 gcn=gfx1151
hipMalloc,        target 76 GiB:  => 76 GiB          (target reached, ceiling is >= 76)
hipMallocManaged, target 76 GiB:  => 27 GiB          (out of memory)
```

27 GiB against 29.3 GiB host-available is the match that identifies the source.
This **resolves the ambiguity [#2518](https://github.com/mudler/vllm.cpp/issues/2518)
could not**: its 58.000 GiB managed ceiling was measured when the carve was
64.00 GiB and host RAM was 62 GiB, and 58 sits below both, so that measurement
alone never said which bound it hit. The current split separates them.

Three consequences, each of which outlives this row:

1. **The shipped artifact fits, on the plain path.** 67.56 GiB and 73.45 GiB
   both sit inside the proven 76 GiB of `hipMalloc`. Neither fits the 27 GiB
   managed ceiling.
2. **`VT_ROCM_MANAGED_ALLOC=1` is now actively harmful on this board**, and
   more so than before the carve change: it moves the ceiling from >= 76 GiB to
   27 GiB. The default is already correct without it. `ResolveMemoryPolicy`
   (`include/vt/rocm/rocm_arch.h:164`) sets `managed_alloc =
   pageable_memory_access` under `kUnset`, and this board reports that 0, so
   the #2511 narrowing already selects plain `hipMalloc`.
3. **`.agents/environment.md` does not carry this box at all.** Its fleet
   table is dated 2026-08-17 and lists `dgx:gpu0`, `thor:gpu0` and
   `orin:gpu0` only, while `strix:gpu0` has been leasable for weeks and every
   ROCm row leases it. The absence is worse than a stale number, because a
   reader sizing a model for this board finds nothing and falls back to the
   64 GiB and 62 GiB figures scattered through
   `q4km-limb3-kquant-vehicle.md` and `rocm-expert-lane-guard.md`, both of
   which the carve change has now falsified. Repaired in this change by adding
   the box with the values measured above.

The `hipMalloc` ceiling above 76 GiB is **not measured**. The probe stopped at
its target by design. Do not quote 96 GiB as an allocatable figure.

## Risks

- **R1. A codebook decoder pointed at a sibling table still decodes.** This is
  the exact failure `rocm_iq_table_seal.h` exists for, and IQ4_NL shares its
  16-entry codebook with IQ4_XS, so it is live here rather than theoretical.
  Mitigation: seal `kValuesIq4nl` by digest and by its lane alphabet, in the
  shape `tests/vt/test_ops_quant_dot.cpp:717` already uses.
- **R2. Reassociation passes a small golden.** See D1. Mitigation: goldens
  taken from the oracle's own kernel over real checkpoint bytes, per block and
  in total, not a synthetic tensor.
- **R3. The CUDA Q8_0-activation variant is a new dispatch path, not an
  entry.** It can regress the Q8_K path it sits beside. Mitigation: the Q8_K
  formats' existing cases must stay byte-identical; assert that rather than
  assume it.
- **R4. `main` moves under this row.** #3029 touches
  `.agents/quantization-matrix.md`, which this row also edits. Per
  `AGENTS.md` "Records", take the complete target-branch version and re-apply
  the scoped edit; never accept a three-way merge of a keyed record.
- **R5. No AMD runner in CI.** Every ROCm device test self-skips, so a green CI
  is not evidence for arms 1 and 2. Their evidence is a named lease.

## Tests

Red first, in this order, each failing for the intended reason before any
kernel exists:

1. `tests/vt/test_rocm_quant_dot.cpp` — IQ4_NL dot against oracle-produced
   goldens over real `UD-IQ1_S` `ffn_down_exps` bytes, per block and total.
2. `tests/vt/test_ops_quant_traits.cpp` — reader and vt geometry agree for
   IQ4_NL, 18-byte block, `QK4_NL = 32`, against the oracle's own
   `sizeof(block_iq4_nl)` printed by the harness.
3. `tests/vllm/test_gguf_keep_quant.cpp` — routing by name: IQ4_NL routes
   `kKeepQuant` on ROCm and CUDA GEMM arms, and on the gather.
4. `tests/vt/test_backend_cross_device.cpp` — IQ4_NL GEMM within NMSE <= 5e-4
   of the CPU oracle, on a real launch shape rather than a toy one. The Q6_K
   precedent is explicit about why: its gate ran `M=3, N=8, K=512` against a
   production launch of 6400 blocks and never tested a launch.
5. A gather case that decodes an IQ4_NL block row on device and matches
   `DequantIQ4_NL`.
6. The three D4 mutations, each restoring the tree byte for byte afterwards.

## Gates

- **G1 (unit).** The suites above, green on CPU, on `strix:gpu0` for the ROCm
  arms, and on a CUDA box for arm 3.
- **G2 (admission).** `UD-IQ1_S` opens and its IQ4_NL tensors keep their blocks
  on ROCm, with `VT_OP_PROVIDER_STATS` showing zero reference-tier hits for the
  IQ4_NL GEMM and gather.
- **G3 (correctness).** Not owned by this row. See `## Owed`.
- **No throughput, latency or memory number is admissible from this row.**
  `AGENTS.md` Gates admits no performance result from an arm whose token gate
  has not passed, and the `gfx1151` token gate currently reads
  `TOKEN_GATE=FAIL` at 3 of 6 on a simpler model
  (`rocm-gfx1151-q4k-token-gate-v2.md`). Numbers captured while this stands are
  recorded as inadmissible, with that word, or they repeat the withdrawn 2.71x.

## Owed

- **The seven `qwen4_exp` operations with no ROCm arm**, which are what stands
  between this row and a model that runs: `kQwen4ExpPleConv`,
  `kQwen4ExpPleGate`, `kQwen4ExpGatedResidual`,
  `kQwen4ExpGatedResidualWriteBack`, `kQwen4ExpQsaCompress`,
  `kQwen4ExpQsaGatherAttention`, `kRmsNormGroup`. `grep -rn Qwen4Exp
  src/vt/rocm/` returns nothing. Needs its own row and spec.
- **The ROCm token gate for `qwen4_exp`**, and the oracle it runs against.
  vLLM at the current pin `e126687a9a` ships a first-class AMD backend for this
  model (`vllm/models/qwen4_exp/amd/`), whose divergence from `nvidia/` is
  about 840 lines confined to the Triton op layer, with `model.py`,
  `model_state.py` and `mtp.py` byte-identical and no AITER or MFMA dependency.
  That makes it the natural denominator on gfx1151, and it is unmeasured.
- **The CPU comparison arm no longer fits this box.** It needed 73.9 GiB
  `VmHWM` and the host side is now 31 GiB. A CPU-versus-ROCm token comparison
  has to run on `thor` or `dgx`, or on-box against an oracle instead.
- IQ2_XS, IQ1_M, Q5_0, Q4_0, IQ3_S and MXFP4 on ROCm (#1940). Wiring the
  already-written CUDA `DotMXFP4` onto arm 3's variant.
- The `hipMalloc` ceiling above 76 GiB on this board, unmeasured by design.

## Stop conditions

- #3029 cannot be landed or its ROCm IQ table headers change shape. Stop and
  re-scope arms 1 and 2; arm 3 and the record repair remain independent.
- The oracle goldens cannot be produced from real checkpoint bytes. Stop rather
  than substitute a synthetic tensor, per R2.
- A device arm disagrees with the CPU oracle beyond NMSE 5e-4 and the cause is
  not reassociation. Stop and report; do not widen the bound.
- `strix:gpu0` is unavailable or unhealthy. Arms 1 and 2 stay `PENDING` on a
  named lease. Never convert an unrun gate into a pass.

## Now

`ACTIVE`, 2026-09-12. Spec committed, no product code, base pinned to
`origin/main` `18f39771c`.

**The scope shrank between drafting and committing, and the record says so
rather than pretending it was always this size.** #3097 landed the ROCm
quantized gather, which was arm 2, and it also landed the ROCm IQ codebook
header that made #3029 look like a hard dependency. What remains is the ROCm
IQ4_NL dot, the CUDA IQ4_NL dot, and the record repair.

Next action is W1, the ROCm dot, which needs no other pull request to land
first.
