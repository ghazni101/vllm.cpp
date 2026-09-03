# QUANT-EXL3-MUL1 — the `mul1` codebook and the bit widths a 3.5bpw EXL3 artifact actually ships

Row: `QUANT-EXL3-MUL1`
Issues: [#2495](https://github.com/mudler/vllm.cpp/issues/2495) (primary),
[#2574](https://github.com/mudler/vllm.cpp/issues/2574) (slice F, the recount
and the `(3, 2)` arm),
[#2756](https://github.com/mudler/vllm.cpp/issues/2756) (slice G, the fused MoE
widths), [#2762](https://github.com/mudler/vllm.cpp/issues/2762) (the device
case that could never run)
Base SHA: `11fed3ba5`
Parent row: [`QUANT-EXL3`](quant-exl3-shared.md)
Matrix: [`.agents/quantization-matrix.md`](../quantization-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** at the pin, so the format is mirrored from the registered secondary
oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT). The seam is vLLM's; exllamav3
supplies the trellis format and its kernels only.

## Now

`ACTIVE`. Slices A and B have landed: the host decoder implements codebook 2 and
the loader accepts a `mul1` marker. Slices C, D and E are described below with
what each one still owes. **Slice F is the recount** (#2574): slice D was
specified against a tensor census that omitted 137 modules, so `(3, 2)` was
never instantiated and the checkpoint could not run on CUDA at all. **Slice G is
the fused MoE** (#2756): `kExl3MoeMlp` was still the tree's narrowest EXL3
surface at one arm, `(3, mcg)`, and a mcg expert tower at any other width could
not run on a CUDA queue by any path.

## The gap

`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (#2495) cannot be loaded at all, and two
independent by-name refusals stand in the way:

1. **The `mul1` codebook (cb 2).** Every quantized linear in that checkpoint
   ships a `.mul1` marker. `dense_weight_loaders.h` refused it by name and
   `cuda_exl3.cu`'s `decode_3inst_2` `static_assert`ed it out.
2. **Bit widths 3, 4 and 5, at codebook 2.** The device arm instantiated
   `(3,0)`, `(3,1)` and `(6,0)` only. The census slice D worked from --- "270
   tensors at 4 bpw, one at 5, one at 6" --- was an UNDERCOUNT, and the
   corrected one is in slice F below: **409** trellis modules, **137** of them
   at bits 3.

Neither refusal was wrong. Decoding cb 2 as cb 0 or cb 1 is the failure this
family's records already document at length: the wrong multiplier yields a
weight with the RIGHT DISTRIBUTION and no correlation to the true one, so every
shape check passes and the model emits fluent nonsense.

## Why cb 2 is a port and not a table entry

Codebooks 0 and 1 differ only in the scramble. Both then mask, xor, and sum the
two fp16 halves of the 32-bit product:

```
x = cw * M ;  x = (x & 0x8fff8fff) ^ 0x3b603b60 ;  v = fp16(x.lo) + fp16(x.hi)
```

Codebook 2 (`codebook.cuh:82-89`) shares only the multiply:

```
x = cw * 0x83DCD12D
s = 0x6400 + (byte0(x) + byte1(x) + byte2(x) + byte3(x))     // __dp4a
v = fma_fp16( bitcast_fp16(s), 0x1eee, 0xc931 )
```

Three facts make that port exact rather than approximate, and each is asserted:

- `0x6400` is chosen so the reinterpretation is exact. fp16 has an ULP of
  exactly 1.0 across the whole `[1024, 2048)` binade, and the byte sum is at
  most `4*255 == 1020`, so `bitcast_fp16(0x6400 + sum)` is the integer
  `1024 + sum` for every reachable sum. Measured over the full 16-bit codeword
  domain, the reachable sums are `[0, 1005]`.
- The two constants are BIT PATTERNS, not the decimals in upstream's comments.
  `0x1eee` is `887/131072 == 0.00676727294921875`, not "0.00677"; `0xc931` is
  `-1329/128 == -10.3828125`, not "-10.39".
- **`__hfma`'s single rounding is reproducible in f32 without a fused op.** `h`
  is an integer in `[1024, 2044]`; `k_inv` is `887 * 2^-17`, so `h * k_inv`
  needs at most 21 significant bits and is exact in f32. `k_bias` is
  `-1329 * 2^-7`, an integer multiple of the same `2^-17` quantum, so the sum
  is too and its magnitude stays under `2^2` — every reachable value lands
  exactly on an f32. The only rounding is the final one to fp16, which is where
  `__hfma` rounds as well. Verified over all 1021 reachable byte sums.

## Scope

IN: the codebook-2 decode on the host and on the device; the loader's resolution
of a `mul1` marker; the `(bits, codebook)` arms `(3,2)`, `(4,2)`, `(5,2)` and
`(6,2)` on the device; the `dq_dispatch` routes for widths 4 and 5; and the
records those changes make stale.

OUT: the `m <= 8` GEMV, which stays specialized to `bits == 3`; the fused MoE
mgemm, which stays `(3, 1)`; every codebook-0 and codebook-1 decode path, which
must stay byte-identical; and any benchmark of the #2495 checkpoint.

## Upstream chain

vLLM implements no EXL3 at the parity pin, so the chain is the registered
secondary oracle's, read at `2398c05635fbbad01a0a51dce63c85c6c8a8450e`:

- `exllamav3_ext/quant/codebook.cuh:25-41` `decode_mul1_product_2` — the
  two-at-a-time cb 2 decode, which the device arm ports verbatim.
- `exllamav3_ext/quant/codebook.cuh:76-89` `decode_3inst<2>` — the same
  arithmetic one codeword at a time, which the host arm ports.
- `exllamav3/modules/quant/exl3.py:74-77,197,223` `LinearEXL3` — the codebook is
  derived from tensor PRESENCE and passed as two booleans.
- `exllamav3/modules/quant/exl3_lib/quantize.py:18-19,1417-1424` — the two
  multipliers and the marker tensors, each written as
  `torch.tensor(<mult>, uint32).view(torch.int)`.
- `exllamav3_ext/quant/exl3_dq.cuh:254-293` `dq_dispatch` — the per-width route:
  `dq8` for 3 and 4, two `dq4`s for 5, 6 and 8, `dq2x2` for 7.
- `exllamav3_ext/quant/exl3_dq.cuh:164-185` `dq8_aligned_4bits` — upstream's
  hand-written 4-bit form. The generic `dq8<4, cb, 4>` this tree already has
  computes the SAME eight windows; verified analytically (`s2 == 0`,
  `i2 % 32 == t >> 3` and `i0 % 32 == ((t >> 3) + 31) & 31` for every
  `t = lane << 3`) and numerically over 200 random tiles by 32 lanes.
- `exllamav3_ext/quant/exl3_gemv.cu:74-90,107-121` — the GEMV instantiation grid
  and its hard gate, which is what decides Slice E.
- `exllamav3_ext/quant/exl3_kernel_map.cuh:81-110` and `comp_units/` — the
  per-`(K, cb)` compilation-unit split, 24 TUs of 16 instantiations each.

## Our baseline

`QUANT-EXL3` W3 left the device arm at `(3,0)`, `(3,1)`, `(6,0)`, the host
decoder at codebooks 0 and 1, and `vt::Exl3Gemm` refusing any other codebook at
the seam. `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` therefore refused at LOAD time,
before any kernel was reached, on the `mul1` marker of its first quantized
linear.

## Port map

| Upstream | Here |
|---|---|
| `codebook.cuh:76-89` `decode_3inst<2>` | `Exl3DecodeCodeword(cw, 2)`, `src/vt/cpu/cpu_exl3_dequant.cpp` |
| `codebook.cuh:25-41` `decode_mul1_product_2` | `decode_mul1_product_2`, `src/vt/cuda/cuda_exl3.cu` |
| `util.cuh:83-90` `half_uint16` | the same union, `src/vt/cuda/cuda_exl3.cu` |
| `exl3.py:74-77` presence rule | `r.codebook = has_mul1 ? 2 : (has_mcg ? 1 : 0)`, `dense_weight_loaders.h` |
| `exl3_dq.cuh:254-293` `dq_dispatch` | `dq_dispatch<bits, cb>`, widths 3-6 |
| `exl3_kernel_map.cu:93-130` the `[K][cb]` table | `Exl3ArmInstantiated` + `GemmKernelForShape` |

## Tests to port

Upstream ships no unit test for the codebook itself — it is exercised only
through `reconstruct` against a reference tensor — so there is no upstream test
to preserve here. What replaces it is stronger for this purpose and is stated as
a deliberate adaptation: the codebook is gated against values computed
independently in exact rational arithmetic from `codebook.cuh:82-89`, so the
gate does not depend on any implementation of it.

## Dependencies

- `QUANT-EXL3` (the seam, the loader and the device arm this row widens).
- The `exllamav3` oracle pin, which must not move under this row.
- An `rc` lease for the device half. There is no local nvcc.

## Work breakdown

- **A — the host reference.** `Exl3DecodeCodeword(cw, 2)` and the seam's
  codebook range. Gated against hand-computed literals.
- **B — the loader refusal.** `r.codebook = has_mul1 ? 2 : (has_mcg ? 1 : 0)`,
  the `mul1` dtype check, and the test that pinned the refusal, CHANGED rather
  than deleted.
- **C — the device decoder.** `decode_mul1_product_2` and the `cb == 2` arm.
- **D — bit widths 4 and 5.** `dq_dispatch`, `Exl3ArmInstantiated`,
  `GemmKernelForShape` and the refusal message.
- **E — the GEMV.** Decide, and record the decision with upstream's own
  envelope rather than an assumption.
- **F — the recount, and `(3, 2)`.** Re-derive the census from the LOCAL
  artifact, instantiate the width the census adds, and correct every place the
  old number was written down.

## Risks/decisions

- **Decoding cb 2 as another codebook is invisible.** The wrong multiplier
  yields a weight with the right distribution and no correlation to the true
  one; every shape check passes and the model emits fluent nonsense. This is why
  the refusal existed and why the gate is against independent literals.
- **A token gate cannot see this.** The decision to gate on hand-computed fp16
  bit patterns, and to assert the BIT PATTERN as well as the value, is what
  catches a result that is numerically close but not an fp16.
- **Widening one seam is not widening the other.** `vt::Exl3Gemm` validates the
  codebook; the device launcher validates the `(bits, codebook)` PAIR. Both are
  asserted, in both directions, so a future widening of one cannot be mistaken
  for the other.
- **Seven arms, not twenty-four.** Upstream's dense table is affordable because
  it splits per `(K, cb)` into one TU each; this tree has one TU in a fat build
  over ten architectures. This line said SIX and said the seventh pair should
  carry the split; slice F added a seventh that does not, because it is not a
  new artifact's width but the same artifact's, missed by the census. Widening
  for a NEW artifact still carries the split.

## Slices

- **A — the host reference.** `Exl3DecodeCodeword(cw, 2)` in
  `src/vt/cpu/cpu_exl3_dequant.cpp`. Gated against HAND-COMPUTED literals, not
  against this tree.
- **B — the loader refusal.** `r.codebook = has_mul1 ? 2 : (has_mcg ? 1 : 0)`
  in `include/vllm/model_executor/models/dense_weight_loaders.h`. The
  both-markers refusal stays; so do the `had` and packed `su`/`sv` refusals.
- **C — the device decoder.** `decode_mul1_product_2` and the `cb == 2` arm of
  `decode_3inst_2` in `src/vt/cuda/cuda_exl3.cu`.
- **D — bit widths 4 and 5.** `Exl3ArmInstantiated`, `GemmKernelForShape` and
  `dq_dispatch`.
- **E — the GEMV.** `exl3_gemv_kernel` is specialized to `bits == 3`.
- **F — the recount, and `(3, 2)`.** `Exl3ArmInstantiated`,
  `GemmKernelForShape` and the refusal message again, plus the corrected census
  in `docs/FEATURES.md` and in this spec. See the section below.
- **G — the fused MoE widths.** `Exl3MoeArmInstantiated` and `MoeKernel` in
  `src/vt/cuda/cuda_exl3.cu`. The fused arm alone, at codebook 1; the codebook
  half is a LOADER slice and is owed. See the section below.

## Slice F — the recount, and the `(3, 2)` arm (#2574)

### The census, re-derived

Slice D was specified against "270 of the checkpoint's quantized tensors are
4 bpw, one is 5 and one is 6". That count is 272. The artifact has **409**
trellis modules.

Re-derived from the LOCAL artifact rather than from range requests over the
HuggingFace API, because a range-read census is what produced the undercount:
`/mnt/nas_share/models/Qwen3.8-27B-EXL3/target-3.5bpw`, revision
`19441ac874c4018295da848e250f23511361cda4`, both `model-0000N-of-00002.safetensors`
shards complete (each shard's largest declared `data_offsets` end equals its
file size). A module is a trellis module when it owns a `.trellis`; its width is
`shape[-1] / 16` (`[k/16, n/16, 16*bits]` on disk as `I16`), and its codebook is
2 when it owns a `.mul1`, 1 for `.mcg`, 0 for neither.

```
2426 tensors, 409 trellis modules, 409 carrying mul1, 0 carrying mcg
bits histogram: {3: 137, 4: 270, 5: 1, 6: 1}
```

The 137 at bits 3 are **entirely** the language model's MLP, and there is not
one of them anywhere else in the artifact:

| module | bits 3 | bits 4 |
|---|---:|---:|
| `model.language_model.layers.<N>.mlp.gate_proj` | 46 | 18 |
| `model.language_model.layers.<N>.mlp.up_proj` | 46 | 18 |
| `model.language_model.layers.<N>.mlp.down_proj` | 45 | 19 |

That is the per-layer mixed-precision split which averages to 3.5 bpw: 64
language-model layers, each MLP projection quantized at 3 or at 4 bits. The
remaining bits-4 population is the GDN tower (144: `linear_attn.in_proj_qkv`,
`in_proj_z`, `out_proj` over 48 layers), the 16 dense-attention layers (63
projections), and the 8-module `mtp` head. `layers.63.self_attn.o_proj` is the
lone bits-5 module and `lm_head` the lone bits-6 one.

### What this changes

`(3, 2)` was absent from `Exl3ArmInstantiated`, so **every one of those 137
projections refused by name** and the checkpoint could not execute a single
decoder layer on a CUDA device. The three instantiations slice D added were
correct and stay; this is the fourth pair the same artifact needs, missed
because the census under-reported it.

It is an INSTANTIATION and not a port. `decode_3inst_2<2>` and
`decode_mul1_product_2` are slice C's, already compiled and already gated at
widths 4, 5 and 6; `dq_dispatch` already routes `bits == 3` through `dq8`,
because `(3, 0)` and `(3, 1)` have used that route since QUANT-EXL3 W3. No new
decode is written here, and none may be: a second decode for the same codebook
is the way two ports of one mistake agree with each other.

### The confusable pair

`(3, 1)` and `(3, 2)` are now BOTH instantiated at the same width, and they are
the pair a threading mistake would silently swap: same `bits`, adjacent `cb`,
same `dq8` route, same tile shapes. A wrong thread does not fail to compile and
does not change any shape --- it decodes with `mcg`'s multiplier and yields a
weight with the right distribution and no correlation to the true one, which is
the failure mode this row's `## Risks/decisions` already names. The mutation
table therefore carries that exact swap, and the device case must red on it.

### The mutation margin, measured before the device run

`(3, 1)`-for-`(3, 2)` is only a useful mutation if the two decodes disagree by
much more than the bound. That is a property of the CODEBOOKS and not of the
device, so it is measurable on the CPU arm alone, and it was measured before
the lease came:

| bits | cb 1 vs cb 2, relative RMS | against the 1.0e-3 bound |
|---|---:|---:|
| 3 | 1.70865 | 1709x |
| 4 | 1.52897 | 1529x |

Same fixture, same seeds and same shapes the device case uses
(`MakeFixture(256, 256, bits, 0x1D0C0DE + bits)`, m = 4), both arms run through
`vt::Exl3Gemm` on a CPU queue, scratch program, not committed. A relative RMS
above 1 is the SATURATED error of two uncorrelated results, which is the
"right distribution, no correlation" this row's `## Risks/decisions` describes,
and it is what makes the M2 row below a bimodal mutation rather than a
tolerance nudge. If the device ever reports an M2 number NEAR 1.0e-3 rather
than near 1.7, the mutation did not apply and the row is not evidence.

### What the gate does NOT reach: three of the four shapes

Each arm is FOUR kernel instantiations, one per entry of the shape table
(`GemmKernelForArm`: `(16,16,128)`, `(16,32,128)`, `(16,32,256)`,
`(16,16,512)`), and `Exl3SelectGemmShape` picks one at runtime from `cc`, `k`
and `n`. The device case runs `m = 4, k = 256, n = 256`. On a Blackwell-class
`cc` with `K = 3` that path is `mod_256 && size_n <= 4096` and `size_k` is not
`> 8192`, so it returns **shape 2** and shape 2 alone launches. Shape 4 could
not be reached from that call at all: it tiles `n` by 512 and
`Exl3GemmShapeCompat` refuses `n = 256`.

So the gate proves the ARM is reachable and decodes correctly, not that all four
of its shapes do. This is the pre-existing coverage of that table --- it is the
same for `(3, 0)`, `(6, 0)`, `(4, 2)`, `(5, 2)` and `(6, 2)` --- so slice F
neither improves nor worsens it, and it is recorded here rather than left to be
discovered. Closing it is a loop over `force_shape_idx` gated by
`Exl3GemmShapeCompat`, with a second `n` for shape 4; it belongs to the table
and to every arm in it, not to this one arm.

### Not in scope

- **The `m <= 8` GEMV** (#2570). `exl3_gemv_kernel` carries
  `static_assert(bits == 3, ...)` and `LSTRIDE` is hardcoded to 24 there; its
  `(3, 2)` and `(4, 2)` arms are a separate and larger job, and
  `Exl3GemvArmInstantiated` stays `bits == 3 && cb == 1`. Upstream's `SEL_GRID`
  does list `(3, 2)`, so this is owed rather than refused as impossible.
- **The loaders, the DFlash2 draft, and the `lm_head`/`mtp` heads.** Other rows
  own those files.
- **ROCm and Vulkan.** Both transcribe the portable CPU reference and decode
  every width already, so the CUDA instantiation list does not bound them.

### Slice F evidence — dgx:gpu0, 2026-09-02

Inside `rc` lease `7503b585-85d4-43ea-8eb8-6f1e7173af73` on `dgx:gpu0`. No
`ssh`, and the job was submitted with `setsid nohup` so a dying client could not
cancel it from the queue. It queued for about 3h50m at depth 3 and ran 31m.

- **Device**: NVIDIA GB10, compute capability 12.1, driver `580.173.02`.
- **Toolchain**: `nvcc` 13.0, apt-installed per run; `-DVLLM_CPP_CUDA_ARCHITECTURES=121a`,
  `-DVLLM_CPP_BUILD_TESTS=ON`, `-DVLLM_CPP_CUTLASS_FETCH=ON`, Ninja, `-j 4`,
  ccache, built in `/tmp`.
- **Tree**: `row/QUANT-EXL3-MUL1-BITS3` `da8d1f420`, staged as a `git archive`
  tarball and pinned inside the job by its sha256
  (`3ddf3514c0848966cad139b02ae17aa67a7fd035255a2479bf5bbd08b0b99fcf`), so the
  job cannot silently build a different tree.
  `src/vt/cuda/cuda_exl3.cu` sha256 `ad532b34cb34f802832060e89da3dc3cba1f5486bc1522b9b9bf5799a6734872`.

**The gate.** `test_exl3_gemm`, whole binary, no `-tc` filter: `rc=0`, 14 of 14
cases, 215 assertions. The device arm RAN rather than skipping, which is the
thing a skip would have hidden inside a green summary line — the CPU-only run of
the same binary reports 202 assertions, and the 13 extra are the device ones.

| arm | device vs CPU, relative RMS | bound |
|---|---:|---:|
| (3, 0) | 3.02751e-07 | 1.0e-3 |
| (6, 0) | 3.06153e-07 | 1.0e-3 |
| **(3, 2) — new** | **3.02544e-07** | 1.0e-3 |
| (4, 2) | 3.06479e-07 | 1.0e-3 |
| (5, 2) | 2.99241e-07 | 1.0e-3 |
| (6, 2) | 2.98355e-07 | 1.0e-3 |

The new arm sits inside the band the five already-gated arms occupy
(2.98e-07 to 3.06e-07), about 3300x under the bound. That is the signature of a
decode that is exact on both sides with only the reduction order differing,
which is what the tier was chosen to measure.

**The cost of the added arm.** One TU, `cuda_exl3.cu`, recompiled with
`CCACHE_DISABLE=1` so the number is a compile and not a cache lookup, on the
same box in the same job:

| variant | compile | object |
|---|---:|---:|
| with `(3, 2)` | 57.837 s | 9,424,816 B |
| without `(3, 2)` | 51.577 s | 8,276,544 B |
| with `(3, 2)`, restored, repeat | 57.055 s | 9,424,816 B |

**+1,148,272 bytes (+13.9%) and about +6 s (+11%)**, for ONE architecture. The
repeat gives the only spread available (1.4% between the two "with" runs), so
the compile-time figure is n=2 against n=1 and should be read as "about 11%",
not as 12.1%. The object figure is exact and repeatable. The fat build compiles
this TU for ten architectures, which is the multiplier that makes the per-K
split in `## Owed` worth more than it was at six arms.

**Mutations.** Each row rebuilt from the same build directory, and each asserts
four things rather than only the verdict: the file's sha256 CHANGED, it
compiled, the test binary's mtime MOVED, and the tree restored byte-for-byte.

| # | mutation | sha256 changed | built | mtime moved | gate | how it failed |
|---|---|---|---|---|---|---|
| M1 | drop `bits == 3` from the cb 2 clause of `Exl3ArmInstantiated` | yes | rc 0 | yes | **RED** | the by-name refusal, at the `vt::Exl3Gemm` entry |
| M2 | `GemmKernelForArm<3, 2>` → `<3, 1>` in `GemmKernelForShape` | yes | rc 0 | yes | **RED** | `CHECK(1.70865 <= 0.001)`, `arm.bits := 3`, `arm.codebook := 2` |
| M3 | DELETE the `(3, 2)` line from `GemmKernelForShape` | yes | rc 0 | yes | **RED** | `exl3_gemm shape 2 has no instantiation` |

M1 is also the RED-BEFORE: for the guarantee under test it is exactly the tree
as it stood before this change, and it fails for the intended reason.

M2 is the one worth reading twice. **The device produced 1.70865, and the CPU
prediction recorded above this section, taken before the lease and on different
hardware, was 1.70865.** Six significant figures. Two things follow: the
mutation certainly applied, and the device kernel certainly reads its `CB`
template parameter rather than deriving the codebook some other way.

M3 is the REACHABILITY row. Deleting the production call site in the shape table
while leaving the predicate alone makes the dispatch return `nullptr` and throw,
so the gate is entering the new arm through `vt::Exl3Gemm` and the shape table,
not by naming a kernel.

**One leg of that job is VOID, and it is an instrument defect rather than a
result.** The job's last phase restored `cuda_exl3.cu` from the tarball, rebuilt,
and re-ran the gate as a "the tree is back and still green" check. It reported
RED. It is void because `tar -x` restores the ARCHIVE's mtime, which is older
than the mutated file, so Ninja saw nothing to do and the binary was never
relinked: `gate-final.log` is BYTE-IDENTICAL to `gate-mut-M3-callsite.log`, and
the phase re-ran M3's mutant. The clean-tree green is phase E's, which built
from the pristine tarball before any mutation touched it, and the restore itself
is verified by sha256 on content. A job that mutates and restores must `touch`
the file after restoring; this one did not. This is the mtime-restore skip that
this repository has already recorded once; a mutation harness that restores from
an archive re-acquires it every time, so it belongs in the harness rather than
in a reader's memory.

That phase also demonstrated the trap it was written against: it reported
`assertions: 207 | 207 passed | 0 failed` beside `Status: FAILURE!`, because the
case THREW rather than failing a `CHECK`. The exit code is what caught it.

### The gate, and why this tier

`tests/vt/test_exl3_gemm.cpp`'s "the widened (bits, codebook) arms agree with
the CPU arm" case gains one row. The reference is the CPU arm, which decodes
every width and every codebook and is itself tied to real exllamav3 data by
`test_exl3_real_decode`; the bound is `rel_rms <= 1.0e-3`, which is the SAME
tier `(4, 2)`, `(5, 2)` and `(6, 2)` use and the same one the `(3, 1)` f64 case
states. It is the right tier because the quantity being bounded is the same one:
the tensor-core reduction order. The decode is exact on both sides, so a
disagreement larger than that bound is a decode disagreement, not rounding.

A device that is absent SKIPS this case, and a skip that asserts nothing reports
`assertions: 0`, which reads as a pass. The case already asserts its own
precondition on the skip path, and this row does not weaken that.

## Slice G — the fused MoE widths (#2756)

### The gap, and its two independent halves

`Exl3MoeMlpKernelCuda` instantiates one arm. `kMoeBits = 3`, `kMoeCb = 1`, and
every other pair refuses by name. That is a much narrower surface than the
regular GEMM beside it in the same file, which slice F took to seven pairs, and
narrower than the CPU arm, which threads `args.bits` and `args.codebook` into
`Exl3DecodeTile` unchanged and therefore decodes all eight widths over all three
codebooks.

The narrowness has two halves and they are NOT the same problem.

**The widths are reachable and refuse.** `deepseek_v4.cpp:1546-1548` reads the
bit width per projection off the checkpoint — `e0.w1.bits`, `e0.w3.bits`,
`e0.w2.bits` — and assigns it straight into `Exl3MoeArgs`. Nothing between the
loader and the launcher clamps it. A DeepSeek-V4 EXL3 expert tower quantized at
4, 5 or 6 bits therefore reaches `Exl3MoeMlpKernelCuda` with that width and is
refused there.

**That refusal is not caught, and there is no second path.** `MoeBlock` calls
`Exl3FusedMoePass` unguarded (`deepseek_v4.cpp:1811`), and the fused arm is
default-ON — `Dsv4Exl3FusedMoeFlagIsOn` reads an unset `VT_DSV4_EXL3_FUSED_MOE`
as on — so the exception leaves the forward rather than degrading to the loop.
Setting `VT_DSV4_EXL3_FUSED_MOE=0` does not rescue it either: the per-expert loop
calls `vt::Exl3Gemm`, and `Exl3ArmInstantiated` has no `(4, 1)`, `(5, 1)` or
`(6, 1)`. A 4-, 5- or 6-bit mcg expert tower cannot run on a CUDA queue by ANY
path today. Widening the fused launcher restores the default path, which is the
one production takes.

**The codebook is NOT reachable and would land dead.** Four sites pin it to 1
before the kernel is chosen, and they have to widen together or not at all:

| Site | What it does |
|---|---|
| `deepseek_v4_weights.cpp:1017` | `VT_CHECK(codebook == "mcg", ...)` — refuses any other marker at load |
| `deepseek_v4.cpp:1336` | `args.codebook = 1;` hardcoded |
| `deepseek_v4.cpp:1371` | `args.codebook = 1;` hardcoded |
| `deepseek_v4.cpp:1549` | `args.codebook = 1;` hardcoded, on the fused MoE call |
| `src/vt/ops.cpp:5609` | `VT_CHECK(args.codebook == 1, ...)` in the SHARED seam, refusing for every backend including the CPU reference |

`vt::Exl3MoeMlp` has exactly two production callers, both inside
`Exl3FusedMoePass` (`deepseek_v4.cpp:1659` and `:1693`), and no second
architecture reaches the op — `qwen3_5_moe.cpp` contains zero EXL3. So a
codebook-2 MoE instantiation added in this slice would be unreachable from any
production entry point at its own merge commit, which "Nothing lands dead"
forbids. It is owed below, as a LOADER slice that happens to end in a kernel.

### What upstream instantiates, and the two bounds it already carries

`exl3_moe.cu:22-33` builds the table and `:226` indexes it:

```
fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];
```

`K` in 0..8, `cb_idx` in {0, 1}, `N_off` in {0, 1} — 36 instantiations, split one
file per (K, cb) in `comp_units/exl3_moe_inst_k*_cb*.cu`. Two of upstream's own
bounds are load-bearing and neither appears in our refusal message:

- **Codebook 0 is not a MoE arm upstream either.** `exl3_moe.cu:184` is
  `TORCH_CHECK(gate_mcg != gate_mul1, "MoE kernel: Only mcg and mul1 codebooks
  are supported")`, and `:185` derives `cb_idx = gate_mul1 ? 1 : 0`. A 3INST
  checkpoint is refused before the table is indexed. So the MoE's reachable
  codebook set is at most `{1, 2}` — it is NOT the GEMM's `{0, 1, 2}`, and a
  refusal message that implies otherwise is wrong about upstream.
- **`K == 0` is upstream's RUNTIME-width instance**, for a tower whose gate, up
  and down widths differ (`exl3_moe_kernel.cuh:139-149`). Our port takes
  `K_gate`, `K_up` and `K_down` as kernel arguments and then discards them —
  `cuda_exl3.cu:1779` is `(void)K;` — so it serves only `Kg == Ku == Kd`. That
  is a separate owed item from the width set, and `deepseek_v4.cpp:1450` already
  refuses a disagreeing tower by name, so nothing silently decodes one expert
  with another's width.

### The width set this slice takes, and why it is four and not eight

`dq_dispatch` (`cuda_exl3.cu:453-454`) static_asserts `bits == 3 || 4 || 5 || 6`
and the reason is arithmetic rather than taste: the eight-window span
`16 + bits*7` leaves the 64-bit funnel once the start shift is added at bits 5,
which is why 5 and 6 route through two `dq4`s, and 1, 2, 7 and 8 have no route
in this tree at all. Widths outside that set are a DECODER question the GEMM
shares, not a MoE question, and this slice does not open it.

So the set is `bits` in {3, 4, 5, 6} at `cb == 1`: four pairs, eight kernels
against the current two, since each pair carries the `MOE_TILESIZE_N` 128 and 256
forms upstream carries.

**Shared memory admits every one of them, and the guard is already in the tree.**
For the MoE shape (`TILESIZE_M` 16, `TILESIZE_K` 32, `SH_STAGES` 3) the
`static_assert` in `exl3_gemm_kernel_inner` resolves to
`kSmemMax >= 3*(2*512 + 2*512*bits) + 4*4096` at `MOE_TILESIZE_N == 256`, which
is 28672 bytes at bits 3 and 37888 at bits 6 against a `kSmemMax` of 92160. The
128 form is smaller still. A width that did not fit would fail to COMPILE on
that assert rather than mis-stage silently, which is what makes widening here
safe to attempt.

### The dead predicate this slice also repairs

`cuda_exl3.cu:2323-2325` refuses on `args.codebook != kMoeCb`. It cannot fire.
`vt::Exl3MoeMlp` is the only route to the registered op and `ops.cpp:5609`
refuses `codebook != 1` first, so the codebook half of that condition is
unreachable and its refusal text can never print. The launcher keeps a codebook
predicate — it is the right place for one — but the message now states
upstream's `{1, 2}` bound rather than implying the GEMM's `{0, 1, 2}`.

### Design

Mirroring `Exl3ArmInstantiated`/`GemmKernelForShape`, which slice F left in the
shape this needs:

- `Exl3MoeArmInstantiated(int bits, int cb)`, `constexpr`, the single source of
  truth for the arm set, carrying upstream's cb bound in its comment.
- `MoeKernel(int bits, int cb, bool n256)` returning `const void*`, a dense
  switch over the instantiated pairs, `nullptr` for anything else.
- The launcher asks the predicate, refuses by name with both bounds stated, then
  asks `MoeKernel`. A `nullptr` from `MoeKernel` on a pair the predicate admitted
  is a `std::runtime_error` and not a launch, so the two can never disagree
  silently.

`Exl3MoeArgs`, `ops.cpp`'s shared validation and the CPU arm are UNCHANGED by
this slice. The CPU arm already serves every width; that is what makes it the
reference the device arm is gated against.

### Tests

`tests/vt/test_exl3_moe.cpp`:

- `MakeMoeFixture` gains a `bits` parameter. It hardcoded `MakeFixture(..., 3,
  ...)` three times while carrying an unused `MoeFixture::bits = 3` field, so
  the fixture could not build a tower the widened arm needs.
- The device case loops `bits` in {3, 4, 5, 6} and gates each against the CPU arm
  at the SAME tier-4 bound the `(3, 1)` arm already carries, `rel <= 2.0e-2`
  (`test_exl3_moe.cpp:585`). The bound is not widened for any new width.
- The case asserts the arm RAN. A device case that skips reports its skip
  through a `CHECK_FALSE` on the registration, exactly as the existing case
  does, so `assertions: 0` can never read as a pass.
- **The fixture UPLOADS** (#2762). The case guarded on
  `CudaBackend::DeviceMemoryIsHostAddressable()` and returned when it was false,
  and that predicate is false on EVERY CUDA device permanently — `CudaBackend`
  declares no override and `cuda_backend.cu` holds it at the inherited `false`
  with a `static_assert` that fires on any override, because `Alloc` is
  `cudaMalloc`. So the case had never executed anywhere and could not, which is
  why `model-dsv4-exl3.md` records W2d tier 4 as "STILL OWED, and it cannot be
  taken on this code". The guard was an honest reading of the fixture:
  `FusedCall` labels HOST vectors with the queue's device, so the nine pointer
  tables carry host addresses while the kernel dereferences them on the device.
  `FusedCallDevice` allocates and copies every operand and fills the tables with
  DEVICE addresses, which is what `Exl3FusedMoePass` builds in production out of
  `ResidentWeight`, and the upload pattern is `tests/vt/test_exl3_rocm.cpp`'s.
  Each upload synchronizes, because `Copy` is a `cudaMemcpyAsync` and an async
  copy out of pageable memory does not promise its source is consumed before it
  returns. **This is what makes the widened arm gateable at all**, and it closes
  a debt older than this slice rather than one it opened.
- A host case pins the refusal: bits 7 refuses BY NAME and the message names
  both the width and upstream's codebook bound.

### Gates

```sh
ctest --test-dir build -R '^test_exl3_moe$' --output-on-failure
ctest --test-dir build -R '^test_exl3_gemm$' --output-on-failure
ctest --test-dir build -R '^test_cuda_deepseek_v4$' --output-on-failure
scripts/agent-preflight.sh --staged
```

The device half needs an `rc` lease and a CUDA build. A CPU-only green says
nothing about an arm that only exists in `cuda_exl3.cu`, and is never reported
as one.

### Mutations required

| # | Mutation | Must |
|---|---|---|
| M1 | Drop bits 4 from `Exl3MoeArmInstantiated` | RED — the 4-bit device case refuses |
| M2 | Drop bits 5 from `Exl3MoeArmInstantiated` | RED |
| M3 | Drop bits 6 from `Exl3MoeArmInstantiated` | RED |
| M4 | Swap one `MoeKernel` entry's `cb` template argument from 1 to 2 | RED — the wrong multiplier yields a correctly distributed and completely wrong weight, which a shape-checking gate cannot see |
| M5 | Swap one `MoeKernel` entry's `bits` template argument to its neighbour | RED |
| M6 | Delete the `vt::Exl3MoeMlp` call at `deepseek_v4.cpp:1659` | RED on the DSV4 forward case — the reachability proof |

Each records the sha256 before and after, that the object rebuilt and its mtime
moved, and that the tree was restored byte-for-byte.

### Stop conditions for this slice

- A new width's device result exceeds `2.0e-2` against the CPU arm → the port is
  wrong. Never raise the bound.
- The `static_assert` on `kSmemMax` fires for a width → record the number and
  drop that width from the set; never raise `kSmemMax`.
- The compile cost of `cuda_exl3.cu` at one architecture more than doubles →
  the per-K translation-unit split this spec already owes stops being deferrable
  and this slice carries it. The number is recorded with the evidence, measured
  and not estimated.

## Tests

- `tests/vt/test_exl3_dequant.cpp` — the hand-computed cb 2 table, a
  whole-domain distinctness sweep against cb 0 and cb 1, a whole-domain
  unchanged-ness sweep for cb 0 and cb 1, and a refusal case for a codebook
  upstream does not define.
- `tests/vllm/model_executor/layers/test_exl3_native_loader.cpp` — the case
  that pinned the `mul1` refusal is CHANGED, not deleted: it now asserts the
  marker is accepted and reports `codebook == 2`, and a sibling case keeps the
  both-markers refusal.
- `tests/vt/test_exl3_gemm.cpp` — the device cb 2 cross-check against the host
  decoder, and the new `(bits, cb)` arms.
- `tests/vt/test_exl3_gemm.cpp` — slice F adds the `(3, 2)` row to that same
  device arms table. It is one row and not a new case, because the question is
  identical to the one the table already asks of the other cb 2 widths.

## Gates

```sh
ctest --test-dir build -R '^test_exl3_dequant$' --output-on-failure
ctest --test-dir build -R '^test_exl3_native_loader$' --output-on-failure
ctest --test-dir build -R '^test_exl3_linear_method$' --output-on-failure
ctest --test-dir build -R '^test_exl3_gemm$' --output-on-failure
scripts/agent-preflight.sh --staged
```

The device arms additionally need an `rc` lease and a CUDA build; a CPU-only
green is not a device result and is never reported as one.

## Owed

- **Slice C is UNVERIFIED ON A DEVICE until a CUDA build runs it.** A CPU-only
  gate cannot compile `cuda_exl3.cu` at all, so a green CPU preflight says
  nothing about the device arm. Named here so it is visible debt rather than an
  assumed pass.
- **Slice D's fat-build cost.** Each new `(bits, cb)` pair is a full kernel set
  compiled for every architecture in the fat build. Upstream's own answer is a
  per-K compilation-unit split (`comp_units/exl3_comp_unit_K_cbX.cu`, 24 TUs of
  16 instantiations each); this tree has one TU.
- **The shape table is gated at ONE of its four shapes, for every arm.** See
  "What the gate does NOT reach" above. `force_shape_idx` already exists, so
  this is a test loop rather than a port, and it is owed by the table.
- **Slice F's arm is VERIFIED on `dgx:gpu0`** (see its evidence section), which
  also closes slice C's device debt for cb 2 at four widths. What is still
  unverified is any REAL tensor of the artifact: the gate is synthetic
  fixtures, and no shape, `k`, `n` or trellis from the 137 modules was fed to
  it.
- **The arm labels in the device arms table never print**
  ([#2587](https://github.com/mudler/vllm.cpp/issues/2587)). `MESSAGE`
  stringifies a `const char*` as a bool, so every row logs `(1)` where its
  description should be, in the CUDA, ROCm and Vulkan arm tables alike.
  Cosmetic, pre-existing, and NOT fixed in this flow because
  `tests/vt/test_exl3_gemm.cpp` is one of the two translation units the lease
  evidence was measured on and editing it would put the evidence and the head
  out of step for no correctness gain.
- **Slice F adds a seventh pair to a table this spec argued should stay at
  six.** The argument stands and the reason it is overridden is that the seventh
  pair is not a new artifact's width, it is the SAME artifact's width that the
  census missed --- six was never the right number for #2495, seven was. The
  per-K translation-unit split upstream uses is now more owed, not less, and the
  compile-cost measurement is recorded with this slice's evidence.
- **Slice E: the GEMV has no arm for these widths.** Upstream's own envelope
  refuses `K < 2 || K > 4` (`exl3_gemv.cu:107-121`) and its instantiation list
  is `(4,0) (4,1) (4,2) (2,1) (2,2) (3,1) (3,2)` — so **bits 5 and 6 have no
  GEMV upstream either**, and falling to the GEMM there is upstream's behaviour
  rather than a gap this row opened. Bits 4 cb 2 IS in upstream's list, and
  reaching it is a kernel port: `LSTRIDE` is `bits == 3 ? 24 : 32`, `LOADS` is
  `WNT`, and a `dq8_regs_4bits` register extractor does not exist in this tree.
- **The fused MoE at codebook 2 is a LOADER slice, not a kernel slice** (slice G,
  #2756). Five sites pin the codebook to 1 before the kernel is chosen —
  `deepseek_v4_weights.cpp:1017` refuses any marker but `mcg`,
  `deepseek_v4.cpp:1336`, `:1371` and `:1549` each assign `args.codebook = 1`,
  and `ops.cpp:5609` refuses `codebook != 1` in the SHARED seam for every
  backend including the CPU reference. They have to widen together; a cb-2
  instantiation added alone would be dead code. Upstream's own bound is
  `{mcg, mul1}` (`exl3_moe.cu:184-185`), so cb 0 is never owed here.
- **Upstream's `K == 0` runtime-width MoE instance is not ported.** A tower whose
  gate, up and down widths differ takes upstream's runtime switch
  (`exl3_moe_kernel.cuh:139-149`); this port discards `K_gate`/`K_up`/`K_down`
  (`cuda_exl3.cu`, `(void)K;`) and refuses such a tower by name.
- **`Exl3ArmInstantiated` has no `(4, 1)`, `(5, 1)` or `(6, 1)`, so after slice G
  the fused MoE serves widths its own fallback does not.** The
  `VT_DSV4_EXL3_FUSED_MOE=0` rollback lever therefore cannot serve a 4-, 5- or
  6-bit mcg tower on a CUDA queue, and the fused arm is the only device path for
  it. Named rather than fixed, because widening the GEMM's cb-1 set costs three
  more pairs in the same translation unit for an artifact this tree has not
  seen; the fused widening is driven by the reachability of `args.bits_gate`,
  which the GEMM's own set does not share.
- **`kExl3MoeMlp` on ROCm and on Vulkan.** Both are owed on their own rows with
  their own reasons and neither is a transcription:
  [`backend-rocm-exl3.md`](backend-rocm-exl3.md) `## Owed` ("unreached,
  unwritten, and unmeasurable on this fleet" — no AMD board here holds the
  ~99.5 GiB artifact that reaches it) and
  [`backend-vulkan-exl3.md`](backend-vulkan-exl3.md) `## Owed` ("a rewrite and
  not a transcription"). The Vulkan blocker is structural: the fused MoE is a
  persistent cooperative launch whose group barrier is legal only because
  `cudaLaunchCooperativeKernel` guarantees co-residency, and Vulkan has no
  equivalent guarantee. Their `kExl3Gemm` arms are NOT narrowed — both decode
  every width the host does — so only the fused op is missing.
- **No end-to-end run of the #2495 checkpoint, and no benchmark.** This row
  ports the format. It does not produce the number #2495 asks for, which is why
  the commits say `Refs` and not `Closes`.
- **`docs/USAGE.md` owes the checkpoint's file names, sizes, repo and
  REVISION** once an arm of it actually loads end to end.

## Stop conditions

- A hand-computed cb 2 literal disagrees with the implementation → stop and
  re-derive from `codebook.cuh:82-89`. Never adjust the literal to green.
- A new `(bits, cb)` arm changes any existing arm's output → the change is not
  additive; stop.
- The fat build stops fitting or a shape's `static_assert` on shared memory
  fires → record the number and propose which pairs are essential, rather than
  raising a limit.
- `(3, 2)` and `(3, 1)` agree on the device case → the case is not
  discriminating, because those two codebooks must produce different numbers at
  the same width. Stop and fix the fixture before reading the tolerance.
