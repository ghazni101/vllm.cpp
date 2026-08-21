# Spec: ROCM-QUANT-GEMM-BW

## Scope

Raise the effective weight-streaming rate of Qwen3.5-4B Q4_K_M greedy
decode on gfx1100 (RX 7900 XTX, ROCm 7.14 container) to at least 60% of
peak DRAM bandwidth (~576 GB/s of ~960), by optimizing the W1 keep-quant
GEMM (`src/vt/rocm/rocm_quant_dot.hip`) memory path and scheduling with
zero numeric change. Owned under issue #1586. Success is measured, not
argued: the fixed workload in `## Gates` must reach the rate with the
existing bit-exactness gate unchanged.

## Upstream anchors

- vLLM pin `555967922` (0.26.0.dev0). vLLM defines no keep-quant RDNA3
  GEMM, so behavior parity does not constrain the internals; only our
  CPU reference (`src/vt/cpu/cpu_quant_dot.cpp`) pins the numerics.
- The CUDA sibling `src/vt/cuda/cuda_quant_dot.cu` is the structural
  mirror. It stays untouched; any improvement found here that would also
  help CUDA is recorded as owed, never ported silently into this row.

## Baseline evidence

`rocprofv3 -r true` capture of the gate workload at tree `6236e9e55`
(144 tokens, 97,721 dispatches, results db parsed from
`rocpd_kernel_dispatch`):

| Fact | Value |
|---|---|
| GPU busy fraction | 0.83 |
| `QuantDotGemm*` share of busy | 48.3% |
| hipBLASLt `Cijk_*` share | 26.2% |
| GDN family share | 16.9% |
| Effective weight-streaming rate | ~163 GB/s (~17% of peak) |
| `QuantDotGemmKernel` decode geometry | grid up to 7,946,240 blocks x 256 threads, avg 1.9 ms |

Diagnosis: the kernel occupies the chip but streams bytes narrowly.
Each lane reads one byte per super-block step; q-weight rows are walked
with lane-strided single-byte loads, so every 256-thread wavefront
touches scattered addresses and the memory system delivers far below
its burst width.

## Design

Attempt ladder, one attempt = change + rebuild + both-gate verify:

1. **Vectorized weight loads.** Give each lane a contiguous 16-byte
   load (`ulonglong2`) covering four lanes' worth of q-weight payload
   per super-block step where the block layout allows it, keeping the
   CPU accumulation order exactly (sum over nibbles/bytes in reference
   sequence). Bit-exactness is preserved because reassociation is not
   introduced; only the load width changes.
2. **Wave/block reshaping.** Reduce grid size by assigning each warp
   multiple output elements along N; improves L2 reuse of activation
   rows and drops launch count. Output mapping stays N-major within a
   super-block so partial sums remain per-output.
3. **hipBLASLt algo-policy A/B** for the bf16 arms (26.2% share):
   measurement-only lever from the #1586 attribution table; adopt a
   pinned algo policy if a variant wins at decode shapes.
4. Optional: `VT_*` env knob parity with the CUDA side for any new
   scheduling switch, defaulting to the fast path.

Numerics guardrail for every attempt: no hardware dot instructions
(gfx1100 has no signed byte dot; recorded in the W1 spec), no change to
scale application order, no fp reassociation beyond what the reference
already fixes.

## Risks

- R1: Vector loads misaligned at odd N*K offsets -> guard with
  alignment checks falling back to the scalar path (same kernel,
  selected per-tensor, still bit-exact).
- R2: Register pressure rise kills occupancy and negates the win ->
  measure VGPR count from the code object before accepting.
- R3: Prefill arms regress while decode improves -> gates measure both;
  accept only when neither arm regresses beyond noise on the gate test.

## Tests

- `tests/vt/test_rocm_quant_dot.cpp` runs UNCHANGED as the correctness
  gate: 132,094 assertions, integer core bit-exact vs CPU, NMSE <= 1e-6.
- Bandwidth gate: the workload in `## Gates`, parsed from rocprofv3
  results db, must show >= 576 GB/s steady-state decode.
- End-to-end smoke: deterministic decode across two identical runs.

## Gates

Fixed workload (identical to the baseline capture):

```
rocprofv3 -r true -- examples/vllm-cli \
  --model /models/Qwen3.5-4B-Q4_K_M.gguf --device auto \
  --temperature 0 --seed 0 --max-tokens 48 --repeat 3 \
  --prompt "Write a detailed explanation of how a transformer neural network works."
```

Rate = 2,740,937,888 bytes x 144 tokens / total GPU-busy seconds of
steady-state decode runs (run 1 warmup excluded). Pass at >= 576 GB/s.

## Owed

- Any improvement applicable to the CUDA sibling: record in the W1
  spec's owed list rather than editing `cuda_quant_dot.cu` here.
- Kernel-matrix family row updates ride the landing commit.

## Stop conditions

20 failed attempts without meeting the bandwidth gate: stop, report
findings and the measured ceiling hypothesis. Ambiguity needing a user
decision or an operation outside recorded authority: halt and surface.
