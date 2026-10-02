// EXL3 small-m GEMV — ROCm arm. Row BACKEND-ROCM, spec
// .agents/specs/rocm-exl3-gemv.md, issue
// .agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3QJSFATGWXMMQKTVDG6BG5G.md.
//
// THE DONOR IS UPSTREAM'S OWN KERNEL, NOT THE CPU REFERENCE. rocm_exl3.hip
// documents why the m<=8 GEMM could not be a CUDA transcription; this file is
// the OTHER arm of the same op, where a CUDA-shaped port DOES exist on this
// hardware, proven in exllamav3-rocm @ 4a19d7a (which serves ~11 tok/s on
// gfx110x). Source, exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh  the kernel (geometry, decode,
//                                              prefetch ring, MMA, reduction)
//   exllamav3_ext/quant/exl3_gemv.cu          the try-launch + shape heuristic
//   exllamav3_ext/ptx_hip.cuh + hip_compat_hip.cuh  exllamav3-rocm's MMA and
//                                              intrinsic shims (same tree,
//                                              f16-accumulate emu and WMMA)
//
// THREE ADAPTATIONS FROM THE UPSTREAM KERNEL, each named in the spec:
//
//   1. NO COOPERATIVE LAUNCH. Upstream wraps an input-Hadamard stage and an
//      output-Hadamard stage around the GEMM body inside ONE kernel and orders
//      them with grid.sync(), which is why its launch is cooperative and its
//      grid is co-residency capped. rocm_exl3.hip's Exl3GemmKernelRocm already
//      runs those transforms as separate LaunchHad launches on one stream —
//      kernel boundaries ARE the grid.sync. The kernel below is therefore the
//      GEMM body only, launched normally between the same two Had calls, and
//      `locks`/`suh`/`A_had`/`svh` are not kernel arguments. The strided
//      `group` loop is kept, so the upstream occupancy-derived grid survives
//      verbatim.
//   2. ACCUMULATE PATH SELECTABLE AT LAUNCH (WMMA template flag). The emulated
//      `mma_m16n8k16_f16` (shuffle-gather + packed __hfma2) is the DEFAULT — it
//      reproduces the CUDA arm's fp16-fragment accumulation order exactly, so
//      the CUDA arm's 6.0e-3 relative-RMS bound (tests/vt/test_exl3_gemv.cpp,
//      spec quant-exl3-perf tier 3c) transfers to this arm unchanged. The gfx11
//      WMMA path (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) is instantiated
//      for the 4 bpw arms and selected by VT_ROCM_EXL3_WMMA=1; exllamav3-rocm
//      carries it behind its own EXL3_WMMA flag with a recorded NaN issue on
//      m>=3 paths, so it is evaluation-only until its own gate clears it.
//   3. STREAMING LOADS. `__ldcs` is `__builtin_nontemporal_load`, the
//      exllamav3-rocm shim: on AMD it emits .slc evict-first loads, which is
//      what a trellis you touch exactly once wants.
//
// THE REST IS VERBATIM: the (WK, WNT, PF, FOLD) config pair, the k-split
// across warps, the register prefetch ring, the per-lane extraction constants
// for bits 3, the two-word funnel windows for bits 4, the FragB decode order
// (w0..w7 as upstream decodes them), the ch -> acc0 fold cadence, and the
// sh_red cross-warp reduction. The epilogue writes f32 into the same `raw`
// scratch Exl3GemmK writes, so LaunchHad post-scale sees an identical operand
// on both arms.
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_f16_codec.h"
#include "vt/rocm/rocm_exl3_decode.h"  // Exl3DecodePairCb2Dp4a

namespace vt::rocm {
namespace {

constexpr int kGemvMaxM = 8;  // EXL3_GEMV_MAX_M (exl3_gemv_kernel.cuh:31)

// 1/sqrt(128), the same literal rocm_exl3.hip's kInvSqrt128 spells (transcribed;
// recomputing 1/sqrt(128) can differ in the last f32 bit). Prefixed kGemv*
// because both TUs sit in vt::rocm's anonymous namespace scope when a bench
// translation unit includes them together.
constexpr float kGemvInvSqrt128 = 0.088388347648f;

inline void CheckG(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm exl3 gemv: ") + what + ": " +
                             hipGetErrorString(err));
}

// ─── fragment types (ptx.cuh:16, util.cuh:83-90) ─────────────────────────────

template <typename T, int n>
struct GVec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
  __device__ const T& operator[](int i) const { return elems[i]; }
};
using FragA = GVec<half2, 4>;
using FragB = GVec<half2, 2>;
using FragC_h = GVec<half2, 2>;  // fp16-accumulate fragment — the GEMV arm
using FragC = GVec<float, 4>;    // f32 fragment — the WMMA converter needs it

union half2_uint32 {
  uint32_t as_uint32;
  half2 as_half2;
  __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
};
union half_uint16 {
  uint16_t as_uint16;
  half as_half;
  __device__ half_uint16(uint16_t val) : as_uint16(val) {}
};

// ─── the codebook decode (codebook.cuh:25-116) ───────────────────────────────
//
// decode_3inst_2 spelled as the donor's HIP file spells it — the lop3 LUT 0x6a
// `(a & b) ^ c` as plain ops, which is bit-identical (integer arithmetic) to
// cuda_exl3.cu's asm.


template <int cb>
__device__ __forceinline__ half2 decode_mcg_or_3inst_2(uint32_t x0, uint32_t x1) {
  if constexpr (cb == 0) {
    x0 *= 89226354u;
    x0 += 64248484u;
    x1 *= 89226354u;
    x1 += 64248484u;
  } else {
    x0 *= 0xCBAC1FEDu;
    x1 *= 0xCBAC1FEDu;
  }
  x0 = 0x3b603b60u ^ (x0 & 0x8fff8fffu);  // lop3 LUT 0x6a: c ^ (a & b)
  x1 = 0x3b603b60u ^ (x1 & 0x8fff8fffu);
  half2_uint32 xu0(x0);
  half2_uint32 xu1(x1);
  half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
  half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
  return __hadd2(d0, d1);
}


// ─── tail-biting window (exl3_dq.cuh) ────────────────────────────────────────

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
  // __funnelshift_r lowers to v_alignbit_b32 (one VALU op) on gfx11; the
  // (a<<32|b)>>s idiom lowers to v_lshrrev_b64, a multi-cycle 64-bit shift.
  // Same bits either way — for shift < 32. dq8_regs_3bits calls this with
  // s2+12 which reaches 42, and v_alignbit wraps its shift mod 32 while the
  // 64-bit shift yields a >> (s-32). The guard keeps both halves correct.
  return shift < 32 ? __funnelshift_r(b, a, shift) : (a >> (shift - 32));
}

template <int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
                                        uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7,
                                        FragB& f0, FragB& f1) {
  if constexpr (cb == 2) {
    f0[0] = Exl3DecodePairCb2Dp4a(w0, w1);
    f0[1] = Exl3DecodePairCb2Dp4a(w2, w3);
    f1[0] = Exl3DecodePairCb2Dp4a(w4, w5);
    f1[1] = Exl3DecodePairCb2Dp4a(w6, w7);
  } else {
    f0[0] = decode_mcg_or_3inst_2<cb>(w0, w1);
    f0[1] = decode_mcg_or_3inst_2<cb>(w2, w3);
    f1[0] = decode_mcg_or_3inst_2<cb>(w4, w5);
    f1[1] = decode_mcg_or_3inst_2<cb>(w6, w7);
  }
}

// Window extraction from two already-loaded words, same order as
// dq8_aligned_4bits (exl3_dq.cuh:164-180). VT_EXL3_FSHF/BFE16 spell it in
// cuda_exl3.cu:1316-1358; shifts and masks are the same arithmetic.
template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1) {
  uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
  s = fshift(b, a, 20);  // FSHF_IMM(s, b, a, 20) — funnelshift_r(b, a, 20), args named (lo,hi)
  w7 = b & 0xffff;
  w6 = (b >> 4) & 0xffff;
  w5 = (b >> 8) & 0xffff;
  w4 = (b >> 12) & 0xffff;
  w3 = (b >> 16) & 0xffff;
  w2 = s & 0xffff;
  w1 = (s >> 4) & 0xffff;
  w0 = (s >> 8) & 0xffff;
  decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8_aligned_2bits (exl3_dq.cuh:186-203).
template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset,
                                               FragB& f0, FragB& f1) {
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  b = fshift(b, a, ((~t_offset) & 8) << 1);
  w7 = b & 0xffff;
  w6 = (b >> 2) & 0xffff;
  w5 = (b >> 4) & 0xffff;
  w4 = (b >> 6) & 0xffff;
  w3 = (b >> 8) & 0xffff;
  w2 = (b >> 10) & 0xffff;
  w1 = (b >> 12) & 0xffff;
  w0 = (b >> 14) & 0xffff;
  decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8<3, cb, 4> (exl3_dq.cuh) with the per-lane funnel
// alignment precomputed by the caller.
template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2,
                                               FragB& f0, FragB& f1) {
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  w7 = fshift(b, a, s2);
  w6 = w7 >> 3;
  w5 = w6 >> 3;
  w4 = w5 >> 3;
  w3 = fshift(b, a, s2 + 12);
  w2 = w3 >> 3;
  w1 = w2 >> 3;
  w0 = w1 >> 3;
  decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
              w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

// ─── the emulated fp16-accumulate MMA (hip_compat_hip.cuh:424-468, 487-522) ──
//
// mma.sync.aligned.m16n8k16.row.col.f16 has no AMD form; RDNA3.5's WMMA is
// 16x16x16 and its register layout is different. The emulation gathers each
// lane's two A rows and two B columns with in-warp shuffles (32 lanes, wave32
// — gfx10/11 default), then runs the 16 FMAs per output element as packed
// __hfma2. Accumulating each output element in a half2 (even/odd k in the two
// lanes) and folding once is the SAME operation order CUDA's
// mma.m16n8k16.f16.f16.f16.f16 performs internally, so the 6.0e-3 bound the
// CUDA arm's gate carries is this arm's bound too.

__device__ __forceinline__ half2 shfl_h2(half2 v, int src) {
  uint32_t u = __builtin_bit_cast(uint32_t, v);
  u = __shfl_sync(0xffffffffffffffffULL, u, src);
  return __builtin_bit_cast(half2, u);
}

__device__ __forceinline__ void mma_m16n8k16_gather(const FragA& frag_a, const FragB& frag_b,
                                                    half2 (&a_rows)[2][8],
                                                    half2 (&b_cols)[2][8]) {
  int lane = threadIdx.x & 31;
  int g = lane >> 2, t = lane & 3;

  // A row m lives in lanes 4*(m%8)..4*(m%8)+3. For row g: k pairs 0..7 come from
  // frag_a[0], k pairs 8..15 from frag_a[2]; for row g+8: frag_a[1] / frag_a[3].
  // Lane 4g+i holds k = 2i, 2i+1.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a_rows[0][i] = shfl_h2(frag_a[0], 4 * g + i);
    a_rows[0][4 + i] = shfl_h2(frag_a[2], 4 * g + i);
    a_rows[1][i] = shfl_h2(frag_a[1], 4 * g + i);
    a_rows[1][4 + i] = shfl_h2(frag_a[3], 4 * g + i);
  }
  // B column n lives in lanes 4n..4n+3: lane 4n+t holds k = 2t, 2t+1 in b[0]
  // and k = 2t+8, 2t+9 in b[1]. Columns 2t and 2t+1 are at lane bases 8t, 8t+4.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    b_cols[0][i] = shfl_h2(frag_b[0], 8 * t + i);
    b_cols[0][4 + i] = shfl_h2(frag_b[1], 8 * t + i);
    b_cols[1][i] = shfl_h2(frag_b[0], 8 * t + 4 + i);
    b_cols[1][4 + i] = shfl_h2(frag_b[1], 8 * t + 4 + i);
  }
}

__device__ __forceinline__ void mma_m16n8k16_f16_emu(const FragA& frag_a, const FragB& frag_b,
                                                     FragC_h& frag_c) {
  half2 a_rows[2][8], b_cols[2][8];
  mma_m16n8k16_gather(frag_a, frag_b, a_rows, b_cols);

  half2* c = reinterpret_cast<half2*>(&frag_c);
  half2 acc00 = __float2half2_rn(0.0f), acc01 = acc00, acc10 = acc00, acc11 = acc00;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc00 = __hfma2(a_rows[0][i], b_cols[0][i], acc00);
    acc01 = __hfma2(a_rows[0][i], b_cols[1][i], acc01);
    acc10 = __hfma2(a_rows[1][i], b_cols[0][i], acc10);
    acc11 = __hfma2(a_rows[1][i], b_cols[1][i], acc11);
  }
  c[0] = __hadd2(c[0], __halves2half2(__hadd(__low2half(acc00), __high2half(acc00)),
                                     __hadd(__low2half(acc01), __high2half(acc01))));
  c[1] = __hadd2(c[1], __halves2half2(__hadd(__low2half(acc10), __high2half(acc10)),
                                     __hadd(__low2half(acc11), __high2half(acc11))));
}

// MMODE==0 form of the emulated pair: rows 8..15 of the fragment are provably
// zero (a[1]/a[3] are hzero at the call site), so the second row half of the
// gather — and its 16 hfma2 — is dead work. Keeping the hot decode path off
// them is measured at ~30% of the MMA loop's issue slots.
__device__ __forceinline__ void mma_m16n8k16_f16_emu_m1(const FragA& frag_a,
                                                        const FragB& frag_b,
                                                        FragC_h& frag_c) {
  int lane = threadIdx.x & 31;
  int g = lane >> 2, t = lane & 3;

  // Only row g's eight k pairs: frag_a[0] (k pairs 0..3) and frag_a[2]
  // (4..7) live in lanes 4g+i — the a[1]/a[3] half never loads.
  half2 a_row[8], b_cols[2][8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a_row[i] = shfl_h2(frag_a[0], 4 * g + i);
    a_row[4 + i] = shfl_h2(frag_a[2], 4 * g + i);
    b_cols[0][i] = shfl_h2(frag_b[0], 8 * t + i);
    b_cols[0][4 + i] = shfl_h2(frag_b[1], 8 * t + i);
    b_cols[1][i] = shfl_h2(frag_b[0], 8 * t + 4 + i);
    b_cols[1][4 + i] = shfl_h2(frag_b[1], 8 * t + 4 + i);
  }

  half2* c = reinterpret_cast<half2*>(&frag_c);
  half2 acc00 = __float2half2_rn(0.0f), acc01 = acc00;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc00 = __hfma2(a_row[i], b_cols[0][i], acc00);
    acc01 = __hfma2(a_row[i], b_cols[1][i], acc01);
  }
  // Same fold order as the full form: acc00 folds into c[0], acc01's
  // contribution is what c[0] would carry — and the c[1] accumulate the full
  // form does adds only zeros under m1, so it is skipped outright.
  c[0] = __hadd2(c[0], __halves2half2(__hadd(__low2half(acc00), __high2half(acc00)),
                                     __hadd(__low2half(acc01), __high2half(acc01))));
}

template <bool WMMA>
__device__ __forceinline__ void gemv_mma_16x16_m1(const FragA& a, const FragB& f0,
                                                  const FragB& f1, FragC_h& c0,
                                                  FragC_h& c1) {
  // WMMA has no dead-work to cut: the instruction runs the same 16x16 tile
  // either way, so the WMMA arm keeps the shared path.
  if constexpr (WMMA) {
    gemv_mma_16x16<WMMA>(a, f0, f1, c0, c1);
    return;
  }
  mma_m16n8k16_f16_emu_m1(a, f0, c0);
  mma_m16n8k16_f16_emu_m1(a, f1, c1);
}

// ─── gfx11 WMMA converters (hip_compat_hip.cuh:540-780) ──────────────────────
// EVALUATION ONLY — see the file header and the spec. Instantiated for the
// 4 bpw arms; the default launch never selects it.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define VT_ROCM_EXL3_WMMA_INST 1

using WmmaA16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaB16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaC32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ WmmaA16 ptx_to_wmma_a(const FragA& frag_a) {
  int lane = threadIdx.x & 31;
  int row = lane % 16;

  const half2* a = reinterpret_cast<const half2*>(&frag_a);
  WmmaA16 result;
  half2* r = reinterpret_cast<half2*>(&result);

  half2 va0 = a[0], va1 = a[1], va2 = a[2], va3 = a[3];
  int base = (row % 8) * 4;
  int e_lo = (row >= 8) ? 1 : 0;
  int e_hi = (row >= 8) ? 3 : 2;

  half2 s00 = shfl_h2(va0, base + 0);
  half2 s01 = shfl_h2(va1, base + 0);
  half2 s02 = shfl_h2(va2, base + 0);
  half2 s03 = shfl_h2(va3, base + 0);
  half2 s10 = shfl_h2(va0, base + 1);
  half2 s11 = shfl_h2(va1, base + 1);
  half2 s12 = shfl_h2(va2, base + 1);
  half2 s13 = shfl_h2(va3, base + 1);
  half2 s20 = shfl_h2(va0, base + 2);
  half2 s21 = shfl_h2(va1, base + 2);
  half2 s22 = shfl_h2(va2, base + 2);
  half2 s23 = shfl_h2(va3, base + 2);
  half2 s30 = shfl_h2(va0, base + 3);
  half2 s31 = shfl_h2(va1, base + 3);
  half2 s32 = shfl_h2(va2, base + 3);
  half2 s33 = shfl_h2(va3, base + 3);

  r[0] = (e_lo == 0) ? s00 : s01;
  r[1] = (e_lo == 0) ? s10 : s11;
  r[2] = (e_lo == 0) ? s20 : s21;
  r[3] = (e_lo == 0) ? s30 : s31;
  r[4] = (e_hi == 2) ? s02 : s03;
  r[5] = (e_hi == 2) ? s12 : s13;
  r[6] = (e_hi == 2) ? s22 : s23;
  r[7] = (e_hi == 2) ? s32 : s33;
  return result;
}

__device__ __forceinline__ WmmaB16 ptx_to_wmma_b(const FragB& frag_b0,
                                               const FragB& frag_b1) {
  int lane = threadIdx.x & 31;
  int col = lane % 16;

  const half2* b0 = reinterpret_cast<const half2*>(&frag_b0);
  const half2* b1 = reinterpret_cast<const half2*>(&frag_b1);
  WmmaB16 result;
  half2* r = reinterpret_cast<half2*>(&result);

  half2 vb00 = b0[0], vb01 = b0[1], vb10 = b1[0], vb11 = b1[1];
  int base = (col % 8) * 4;

  half2 shuffles[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    shuffles[s][0] = shfl_h2(vb00, base + s);
    shuffles[s][1] = shfl_h2(vb01, base + s);
    shuffles[s][2] = shfl_h2(vb10, base + s);
    shuffles[s][3] = shfl_h2(vb11, base + s);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int be = j < 4 ? 0 : 1;
    int s = j % 4;
    int elem = be + (col >= 8 ? 2 : 0);
    r[j] = shuffles[s][elem];
  }
  return result;
}

// WMMA D: lane l holds D[2r + l/16][l%16] for r=0..7. PTX C: lane (g,t); c[i]
// holds D[g+8*(i/2)][2t+(i%2)]. D[i][j] is at WMMA lane (i%2)*16 + j, reg i/2.
__device__ __forceinline__ void wmma_to_ptx_c(const WmmaC32& wmma_c, FragC& frag_c0,
                                              FragC& frag_c1) {
  int lane = threadIdx.x & 31;
  int g = lane / 4;
  int t = lane % 4;
  const float* c = reinterpret_cast<const float*>(&wmma_c);
  float* d0 = reinterpret_cast<float*>(&frag_c0);
  float* d1 = reinterpret_cast<float*>(&frag_c1);

  int e0 = g / 2;
  int e1 = g / 2 + 4;
  int lane_base = (g % 2) * 16;

  float cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = c[i];

  int src_lanes[4] = {lane_base + 2 * t, lane_base + 2 * t + 1, lane_base + 2 * t + 8,
                      lane_base + 2 * t + 9};

  float shuffles[4][8];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      shuffles[s][e] = __shfl_sync(0xffffffffffffffffULL, cv[e], src_lanes[s]);
  }

  d0[0] = shuffles[0][e0];
  d0[1] = shuffles[1][e0];
  d0[2] = shuffles[0][e1];
  d0[3] = shuffles[1][e1];
  d1[0] = shuffles[2][e0];
  d1[1] = shuffles[3][e0];
  d1[2] = shuffles[2][e1];
  d1[3] = shuffles[3][e1];
}

// WMMA 16x16x16 fp16→fp32 internally, f16 accumulation — replaces two emulated
// mma_m16n8k16_f16 calls. f32 WMMA + conversion, NOT the f16 WMMA builtin: the
// donor documents that choice as simpler and numerically more robust.
__device__ __forceinline__ void wmma_m16n16k16_f16(const FragA& frag_a, const FragB& frag_b0,
                                                   const FragB& frag_b1, FragC_h& frag_c0,
                                                   FragC_h& frag_c1) {
  WmmaA16 a = ptx_to_wmma_a(frag_a);
  WmmaB16 b = ptx_to_wmma_b(frag_b0, frag_b1);

  WmmaC32 c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);

  FragC add_c0 = {}, add_c1 = {};
  wmma_to_ptx_c(c, add_c0, add_c1);

  float* a0 = reinterpret_cast<float*>(&add_c0);
  float* a1 = reinterpret_cast<float*>(&add_c1);
  half2* d0 = reinterpret_cast<half2*>(&frag_c0);
  half2* d1 = reinterpret_cast<half2*>(&frag_c1);
  d0[0] = __hadd2(d0[0], __floats2half2_rn(a0[0], a0[1]));
  d0[1] = __hadd2(d0[1], __floats2half2_rn(a0[2], a0[3]));
  d1[0] = __hadd2(d1[0], __floats2half2_rn(a1[0], a1[1]));
  d1[1] = __hadd2(d1[1], __floats2half2_rn(a1[2], a1[3]));
}
#else
#define VT_ROCM_EXL3_WMMA_INST 0
#endif

// The paired 16x16 MMA the kernel body calls: one FragA against two adjacent
// FragB tiles into two FragC_h accumulators. WMMA selects the hardware path;
// otherwise the two emulated 16x8 calls upstream's own m16n16 pair expands to.
template <bool WMMA>
__device__ __forceinline__ void gemv_mma_16x16(const FragA& a, const FragB& f0,
                                               const FragB& f1, FragC_h& c0, FragC_h& c1) {
#if VT_ROCM_EXL3_WMMA_INST
  if constexpr (WMMA) {
    wmma_m16n16k16_f16(a, f0, f1, c0, c1);
    return;
  }
#else
  static_assert(!WMMA, "exl3 gemv: WMMA instantiated off gfx11");
#endif
  mma_m16n8k16_f16_emu(a, f0, c0);
  mma_m16n8k16_f16_emu(a, f1, c1);
}

// ─── the kernel body (exl3_gemv_kernel.cuh:138-402) ─────────────────────────
//
// Hadamard stages REMOVED — see adaptation 1 in the header. A is the already-
// transformed a_had; C is the pre-post-scale f32 scratch. Everything between
// the removed stages is verbatim, including the unused-here guards that keep
// the codebook argument exercised identically.
template <int bits, int cb, int MMODE, int CFG, bool SMEM_STAGE, bool WMMA>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256) void Exl3GemvK(
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C,
    int size_m, int size_k, int size_n) {
  static_assert(bits == 2 || bits == 3 || bits == 4,
                "exl3_gemv_kernel supports 2, 3 and 4 bpw");
  constexpr int WK = CFG == 0 ? 16 : 8;   // k-split (warps per block)
  constexpr int WNT = CFG == 0 ? 2 : 4;   // adjacent n-tiles per warp
  constexpr int PF = CFG == 0 ? 4 : 2;    // prefetch ring depth
  constexpr int FOLD = CFG == 0 ? 4 : 2;  // fp16->fp32 fold cadence (divides PF)
  constexpr int THREADS = WK * 32;
  constexpr int ROWS = MMODE == 0 ? 1 : kGemvMaxM;
  constexpr int COLS = WNT * 16;

  constexpr int TWORDS = 8 * bits;                  // uint32 per 16x16 tile
  constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;  // warp loads per k-slice
  constexpr int LSTRIDE = bits == 3 ? 24 : 32;      // uint32 per load
  static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  // Grid-level k-split: blockIdx.y owns slice blockIdx.y of gridDim.y k
  // slices, and the warps split THAT sub-range. The C writeout becomes an
  // atomicAdd whenever more than one block produces a partial per (m,col).
  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const size_t slice_stride = (size_t)ntiles * TWORDS;  // uint32 per k-slice row
  const half2* A2 = reinterpret_cast<const half2*>(A);
  const half2 hzero = __half2half2(__ushort_as_half(0));

  // A fragment row indices for this lane
  const int r0 = lane >> 2;
  const size_t a_row0 = (size_t)r0 * (size_k / 2);
  const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

  // Per-lane extraction constants (dq8_aligned_2bits / dq8<3, cb, 4> in
  // exl3_dq.cuh)
  [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
  if constexpr (bits == 2) {
    int i1 = lane >> 1;
    x_src_b = i1;
    x_src_a = (i1 + 15) & 15;
  }
  if constexpr (bits == 3) {
    int t_offset = lane << 3;
    int b1 = (t_offset + 257) * 3;
    int b2 = b1 + 21;
    int i0 = (b1 - 16) / 32;
    int i2 = (b2 - 1) / 32;
    x_s2 = (i2 + 1) * 32 - b2;
    x_src_a = i0 % 24;
    x_src_b = i2 % 24;
  }

  __shared__ float sh_red[WK][ROWS][COLS];
  [[maybe_unused]] __shared__ uint32_t
      sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;

    // Prefetch ring (indices must be compile-time or pf lands in local memory)
    auto ld_b = [&](int i, int l) -> uint32_t {
      if constexpr (bits == 3)
        return lane < 24 ? __builtin_nontemporal_load(bp + (size_t)i * slice_stride +
                                                      l * LSTRIDE)
                         : 0;
      else
        return __builtin_nontemporal_load(bp + (size_t)i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }

        if constexpr (SMEM_STAGE) {
          __syncwarp();
#pragma unroll
          for (int l = 0; l < LOADS; ++l)
            if (bits != 3 || lane < 24) sh_stage[warp][l * LSTRIDE + lane] = bw[l];
          __syncwarp();
        }

        // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
        const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
        FragB a01, a23;
        a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
        a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
        a01[1] = hzero;
        a23[1] = hzero;

#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          if constexpr (SMEM_STAGE) {
            const uint32_t* tp = &sh_stage[warp][t * TWORDS];
            if constexpr (bits == 4)
              dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
            else if constexpr (bits == 2)
              dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
            else
              dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
          } else if constexpr (bits == 4) {
            uint32_t aw = __shfl_sync(0xffffffffffffffffULL, bw[t], (lane + 31) & 31);
            dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
          } else if constexpr (bits == 2) {
            // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
            const uint32_t w = bw[t >> 1];
            const int base = (t & 1) << 4;
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_b);
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_a);
            dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
          } else {  // bits == 3
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_a);
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }

          FragA a;
          a[0] = a01[0];
          a[1] = a01[1];
          a[2] = a23[0];
          a[3] = a23[1];
          if constexpr (MMODE == 0)
            gemv_mma_16x16_m1<WMMA>(a, f0, f1, ch[t][0], ch[t][1]);
          else
            gemv_mma_16x16<WMMA>(a, f0, f1, ch[t][0], ch[t][1]);
        }

        if ((d + 1) % FOLD == 0 || i + 1 == myn) {
#pragma unroll
          for (int t = 0; t < WNT; ++t)
#pragma unroll
            for (int f = 0; f < 2; ++f) {
              acc0[t][f].x += __low2float(ch[t][f][0]);
              acc0[t][f].y += __high2float(ch[t][f][0]);
              ch[t][f][0] = hzero;
            }
        }
      }
    }

    // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
    // tile*16 + frag*8 + 2*(l%4) (+1)
    {
      const int c0 = 2 * (lane & 3);
      const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
      const int sr0 = MMODE == 0 ? 0 : r0;
      if (store0) {
#pragma unroll
        for (int t = 0; t < WNT; ++t)
#pragma unroll
          for (int f = 0; f < 2; ++f) {
            const int col = t * 16 + f * 8 + c0;
            sh_red[warp][sr0][col + 0] = acc0[t][f].x;
            sh_red[warp][sr0][col + 1] = acc0[t][f].y;
          }
      }
    }
    __syncthreads();

    const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
    for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
      const int r = idx / COLS;
      const int c = idx % COLS;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][r][c];
      const int col = group * COLS + c;
      if (ksplit > 1)
        atomicAdd(C + (size_t)r * size_n + col, sum);
      else
        C[(size_t)r * size_n + col] = sum;
    }
    __syncthreads();
  }
}

// ─── the m=1 direct form (RDNA-specific, spec rocm-exl3-gemv-residual) ──────
//
// dq8's output already lands on the layout a one-token decode wants: lane l
// decodes the four codewords of column l>>2 AND of column l>>2+8, at the k
// quarter t = l&3. Upstream's fragment machinery then gathers every lane's
// data to every other lane because the CUDA MMA's operand layout demands it —
// for m=1 that machinery reduces to a dot product per (column, k quarter),
// which v_dot2_f32_f16 does in one instruction per pair. Per warp-tile this
// arm issues ~60 instructions where the fragment form issues ~350 (measured
// SASS: 296 ds_bpermute + 256 pk_fma + 156 mov_b16 + 124 and per 4-slice),
// and accumulates in f32 instead of the fragment's f16 — strictly nearer the
// CPU reference, not looser.
template <int bits, int cb>
__global__ __launch_bounds__(512) void Exl3GemvM1K(
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C,
    int size_m, int size_k, int size_n) {
  (void)size_m;  // m==1 is a launch-time fact; keeping the same arg list as
                 // Exl3GemvK so the launcher shares one kargs block.
  constexpr int WK = 16;   // k-split (warps per block)
  constexpr int WNT = 2;   // adjacent n-tiles per warp
  constexpr int PF = 4;    // prefetch ring depth
  constexpr int COLS = WNT * 16;

  constexpr int TWORDS = 8 * bits;                  // uint32 per 16x16 tile
  constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;  // warp loads per k-slice
  constexpr int LSTRIDE = bits == 3 ? 24 : 32;      // uint32 per load
  static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");


  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const size_t slice_stride = (size_t)ntiles * TWORDS;
  const half2* A2 = reinterpret_cast<const half2*>(A);

  // Per-lane extraction constants — the same formulas the fragment arm uses
  // (dq8_aligned_2bits / dq8<3, cb, 4> in exl3_dq.cuh).
  [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
  if constexpr (bits == 2) {
    int i1 = lane >> 1;
    x_src_b = i1;
    x_src_a = (i1 + 15) & 15;
  }
  if constexpr (bits == 3) {
    int t_offset = lane << 3;
    int b1 = (t_offset + 257) * 3;
    int b2 = b1 + 21;
    int i0 = (b1 - 16) / 32;
    int i2 = (b2 - 1) / 32;
    x_s2 = (i2 + 1) * 32 - b2;
    x_src_a = i0 % 24;
    x_src_b = i2 % 24;
  }

  __shared__ float sh_red[WK][COLS];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&](int i, int l) -> uint32_t {
      if constexpr (bits == 3)
        return lane < 24 ? __builtin_nontemporal_load(bp + (size_t)i * slice_stride +
                                                          l * LSTRIDE)
                         : 0;
      else
        return __builtin_nontemporal_load(bp + (size_t)i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    // Two accumulators: acc0 for column l>>2, acc1 for column l>>2 + 8.
    float acc0[WNT] = {}, acc1[WNT] = {};

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }

        // A row 0's two half2s at this lane's k quarter — the same pair the
        // fragment arm loads into a01/a23.
        const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
        const half2 a01 = A2[a_col];
        const half2 a23 = A2[a_col + 4];

#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          if constexpr (bits == 4) {
            uint32_t aw = __shfl_sync(0xffffffffffffffffULL, bw[t], (lane + 31) & 31);
            dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
          } else if constexpr (bits == 2) {
            const uint32_t w = bw[t >> 1];
            const int base = (t & 1) << 4;
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_b);
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_a);
            dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
          } else {  // bits == 3
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_a);
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }

          // f0 -> column l>>2 (k pairs 2t/2t+1 and 2t+8/2t+9), f1 -> column
          // l>>2+8, same k quarter. f32 accumulate via v_dot2_f32_f16.


          acc0[t] = amd_mixed_dot(f0[0], a01, acc0[t], false);
          acc0[t] = amd_mixed_dot(f0[1], a23, acc0[t], false);
          acc1[t] = amd_mixed_dot(f1[0], a01, acc1[t], false);
          acc1[t] = amd_mixed_dot(f1[1], a23, acc1[t], false);
        }
      }
    }

    // Fold the four lanes of each column (t bits of the lane id) and drop the
    // pair into the block-level k reduction, which is the same shared-memory
    // shape the fragment arm uses.
    const int c0 = lane >> 2;
#pragma unroll
    for (int t = 0; t < WNT; ++t) {
      float v0 = acc0[t], v1 = acc1[t];
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 1);
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 2);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 1);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 2);
      if ((lane & 3) == 0) {
        sh_red[warp][t * 16 + c0] = v0;
        sh_red[warp][t * 16 + c0 + 8] = v1;
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    __syncthreads();
  }
}

// Wide-load variant of the m=1 arm for 4 bpw. The upstream dq8 lane mapping
// (lane l owns codewords 8l..8l+7) forces one funnel-shift neighbour word per
// lane per tile and a 128-byte strided access pattern. Assigning ownership by
// LOAD POSITION instead — lane l decodes the codewords whose bits sit inside
// its own four words 4l..4l+3 of a four-tile group — removes the shuffle
// entirely: each lane's window set needs only its uint4 plus one boundary
// word (4l-1, the trailing lane's data, an L2 hit after the uint4 stream).
// RowMajorIndex maps t in [32l, 32l+31] to columns (l&7) and (l&7)+8 of tile
// l>>3 across ALL 16 rows, so each lane accumulates whole column partials and
// the epilogue is a block k-fold only — no per-tile xor-shuffles either.
// Per warp-tile-group: one 512-byte contiguous global_load_dwordx4 stream,
// one scalar boundary load, then 32 codeword extracts + 16 v_dot2_f32_f16.
template <int cb>
__global__ __launch_bounds__(512) void Exl3GemvM1K4(
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C,
    int size_m, int size_k, int size_n) {
  (void)size_m;
  constexpr int WK = 16;   // k-split (warps per block)
  constexpr int PF = 4;    // prefetch ring depth
  constexpr int COLS = 64; // one warp covers 4 tiles = 64 columns

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l7 = lane & 7;   // column base within the lane's tile

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  // Tile-word index of this lane's four words inside the 128-word group.
  const int w0 = (lane >> 3) * 32 + (l7 << 2);
  const int wprev = (lane >> 3) * 32 + (((l7 << 2) - 1) & 31);
  const size_t slice_stride = (size_t)ntiles * 32;  // u32 per k-slice at 4bpw
  const half2* A2 = reinterpret_cast<const half2*>(A);

  __shared__ float sh_red[WK][COLS];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + (size_t)group * 128;

    auto ld_b = [&](int i) -> uint4 {
      // Plain uint4 deref lowers to global_load_dwordx4; the stream is
      // single-use so the L1 evict-first policy is all it needs.
      return *reinterpret_cast<const uint4*>(bp + (size_t)i * slice_stride +
                                             w0);
    };

    uint4 pf[PF];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn) pf[d] = ld_b(d);

    float accA = 0.0f, accB = 0.0f;  // columns l7 and l7+8 of tile lane>>3

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        const uint4 v = pf[d];
        if (i + PF < myn) pf[d] = ld_b(i + PF);

        const uint32_t wb =
            bp[(size_t)i * slice_stride + wprev];

        // Window for group-codeword s (tile codeword t = 32*l7 + s): from
        // Exl3TileCodeword, words (i0,i1) map to regs [(4s+1008)>>5, (4s+1023)>>5]
        // relative to word 4*l7-1, and s0 = (28 - 4s) & 31.
        uint32_t w[5] = {wb, v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          FragB fA, fB;
          uint32_t cw[8];
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            const int sc = j * 8 + s;
            // i0 = (4*sc + 1012) >> 5, i1 = (4*sc + 1027) >> 5, both
            // relative to word 4*l7-1 of the tile (index offset 31).
            // fshift(b, a) = (a<<32|b)>>s: the HIGH word is i0's, the LOW
            // word is i1's — argument order is (w[i1], w[i0]).
            cw[s] = fshift(w[((4 * sc + 1027) >> 5) - 31],
                           w[((4 * sc + 1012) >> 5) - 31],
                           (28 - 4 * sc) & 31) & 0xffffu;
          }
          // decode8 emits f=[w0..w3],f1=[w4..w7]; for s in [8j,8j+8) the f
          // half is column l7 (sub<4) and f1 is column l7+8.
          decode8<cb>(cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], cw[7],
                      fA, fB);
          const size_t a_col = (size_t)(ks0 + i) * 8;
          const half2 a01 = A2[a_col + j];
          const half2 a89 = A2[a_col + j + 4];
          accA = amd_mixed_dot(fA[0], a01, accA, false);
          accA = amd_mixed_dot(fA[1], a89, accA, false);
          accB = amd_mixed_dot(fB[0], a01, accB, false);
          accB = amd_mixed_dot(fB[1], a89, accB, false);
        }
      }
    }

    // Whole-column partials already: no intra-warp fold. Warp's tile is
    // (lane>>3), column l7 / l7+8.
    sh_red[warp][(lane >> 3) * 16 + l7] = accA;
    sh_red[warp][(lane >> 3) * 16 + l7 + 8] = accB;
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    __syncthreads();
  }
}
// ─── the fused input-Hadamard m=1 arm ────────────────────────────────────────
//
// For m=1 decode the three-kernel chain launches had_r_128 (A, f16→f16) as a
// separate kernel, then every GEMV block re-streams A_had from global memory
// once per column group. M1K4 v5 (bench3.hip, verified bit-identical against
// HadK<true,true> + the unfused kernel) instead computes a_had in LDS once per
// block: k/128 warp-jobs of the same 5-level xor-partner butterfly HadK runs,
// stores packed half2s, and the dot loop reads shA where the unfused arm read
// A2. One fewer launch and no a_had write+read round-trip per projection.
//
// The butterfly is HadK<true,true> op-for-op: MulF16 pre-scale (the fp16
// product rounded once from f32, same as the host __hmul2), in-lane levels
// 1-2 as (s0+s1, d0+d1, s0-s1, d0-s1), then the five xor-partner levels with
// own-first adds and NegBySignBit flips — the identical instruction sequence,
// LDS in HadK replaced by __shfl_xor here. DF16ToF32/DF32ToF16 come from
// rocm_f16_codec.h; NegBySignBit/MulF16 are transcribed 1:1 from
// rocm_exl3.hip (the two files share no TU).

__device__ __forceinline__ float GemvNegBySignBit(float v) {
  return __int_as_float(static_cast<int>(__float_as_uint(v) ^ 0x80000000u));
}
__device__ __forceinline__ uint16_t GemvMulF16(uint16_t a, uint16_t b) {
  return DF32ToF16(DF16ToF32(a) * DF16ToF32(b));
}
__device__ __forceinline__ void HadBlock128ToH2(const uint16_t* __restrict__ a,
                                              const uint16_t* __restrict__ pre,
                                              int64_t base, half2* out4) {
  const int t = threadIdx.x & 31;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    v[j] = DF16ToF32(GemvMulF16(a[base + 4 * t + j], pre[base + 4 * t + j]));
  const float s0 = v[0] + v[1], d0 = v[0] - v[1];
  const float s1 = v[2] + v[3], d1 = v[2] - v[3];
  float x[4] = {s0 + s1, d0 + d1, s0 - s1, d0 - d1};
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    const bool flip = (t & i) != 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float par = __shfl_xor_sync(0xffffffffffffffffULL, x[j], i);
      const float own = flip ? GemvNegBySignBit(x[j]) : x[j];
      x[j] = own + par;  // own FIRST, matching the host adds
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
    // DF32ToF16, not __float2half: HadK rounds f32(res) then f32->f16 (double
    // rounding). __float2half(x * k) contracts to v_fma_mixlo_f16, a SINGLE
    // rounding, which diffs by 1 ulp on ~1/5K boundary inputs and silently
    // perturbs every GEMV column downstream. Write through uint16_t* —
    // DF32ToF16 returns the bit pattern, not a numeric half.
    reinterpret_cast<uint16_t*>(out4)[j] = DF32ToF16(x[j] * kGemvInvSqrt128);
}

// 4 bpw fused arm — Exl3GemvM1K4 with the input hadamard staged in LDS.
// `shA` is dynamic LDS sized size_k * 2 bytes; every block stages the same
// a_had (only its own group columns differ), which is the trade the launch
// count buys.
template <int cb>
__global__ __launch_bounds__(512) void Exl3GemvM1K4Fused(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ suh,
    const uint16_t* __restrict__ B, float* __restrict__ C, int size_m,
    int size_k, int size_n) {
  (void)size_m;
  constexpr int WK = 16;
  constexpr int PF = 4;
  constexpr int COLS = 64;
  extern __shared__ half2 shA[];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l7 = lane & 7;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  // In-had phase: warp-jobs over k/128 blocks. Every warp iterates its stride;
  // each produces one 128-column block of a_had into LDS as packed half2s.
  const uint16_t* ap = A;
  for (int hb = warp; hb < size_k / 128; hb += WK) {
    half2 o2[2];
    HadBlock128ToH2(ap, suh, hb * 128, o2);
    shA[hb * 64 + lane * 2] = o2[0];
    shA[hb * 64 + lane * 2 + 1] = o2[1];
  }
  __syncthreads();

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const int w0 = (lane >> 3) * 32 + (l7 << 2);
  const size_t slice_stride = (size_t)ntiles * 32;

  __shared__ float sh_red[WK][COLS];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + (size_t)group * 128;
    // The trellis is touched once per decode step — the sub-4 fused arm's
    // nontemporal spelling kept for parity (evict-first .slc on gfx11).
    auto ld_b = [&](int i) -> uint4 {
      const uint32_t* p = bp + (size_t)i * slice_stride + w0;
      return make_uint4(__builtin_nontemporal_load(p),
                        __builtin_nontemporal_load(p + 1),
                        __builtin_nontemporal_load(p + 2),
                        __builtin_nontemporal_load(p + 3));
    };
    uint4 pf[PF];
#pragma unroll
    for (int d = 0; d < PF; ++d) if (d < myn) pf[d] = ld_b(d);

    float accA = 0.0f, accB = 0.0f;
    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;
        const uint4 v = pf[d];
        if (i + PF < myn) pf[d] = ld_b(i + PF);
        const int bsrc = (l7 == 0) ? (lane | 7) : (lane - 1);
        const uint32_t wb = __shfl_sync(0xffffffffffffffffULL, v.w, bsrc);
        uint32_t w[5] = {wb, v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          FragB fA, fB;
          uint32_t cw[8];
#pragma unroll
          for (int s2 = 0; s2 < 8; ++s2) {
            const int sc = j * 8 + s2;
            cw[s2] = fshift(w[((4 * sc + 1027) >> 5) - 31],
                            w[((4 * sc + 1012) >> 5) - 31],
                            (28 - 4 * sc) & 31) & 0xffffu;
          }
          decode8<cb>(cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], cw[7],
                      fA, fB);
          const int a_col = (ks0 + i) * 8;
          const half2 a01 = shA[a_col + j];
          const half2 a89 = shA[a_col + j + 4];
          accA = amd_mixed_dot(fA[0], a01, accA, false);
          accA = amd_mixed_dot(fA[1], a89, accA, false);
          accB = amd_mixed_dot(fB[0], a01, accB, false);
          accB = amd_mixed_dot(fB[1], a89, accB, false);
        }
      }
    }

    sh_red[warp][(lane >> 3) * 16 + l7] = accA;
    sh_red[warp][(lane >> 3) * 16 + l7 + 8] = accB;
    __syncthreads();
    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    __syncthreads();
  }
}

// Sub-4 bpw fused arm — Exl3GemvM1K with the same LDS input-hadamard phase.
// The A read it replaces is `A2[a_col]`/`A2[a_col+4]` (the lane's k quarter),
// which maps 1:1 onto shA since a_had's layout is column-major-contiguous
// half2s.
template <int bits, int cb>
__global__ __launch_bounds__(512) void Exl3GemvM1KFused(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ suh,
    const uint16_t* __restrict__ B, float* __restrict__ C, int size_m,
    int size_k, int size_n) {
  (void)size_m;
  constexpr int WK = 16;
  constexpr int WNT = 2;
  constexpr int PF = 4;
  constexpr int COLS = WNT * 16;
  constexpr int TWORDS = 8 * bits;
  constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
  constexpr int LSTRIDE = bits == 3 ? 24 : 32;
  extern __shared__ half2 shA[];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const uint16_t* ap = A;
  for (int hb = warp; hb < size_k / 128; hb += WK) {
    half2 o2[2];
    HadBlock128ToH2(ap, suh, hb * 128, o2);
    shA[hb * 64 + lane * 2] = o2[0];
    shA[hb * 64 + lane * 2 + 1] = o2[1];
  }
  __syncthreads();

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const size_t slice_stride = (size_t)ntiles * TWORDS;

  [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
  if constexpr (bits == 2) {
    int i1 = lane >> 1;
    x_src_b = i1;
    x_src_a = (i1 + 15) & 15;
  }
  if constexpr (bits == 3) {
    int t_offset = lane << 3;
    int b1 = (t_offset + 257) * 3;
    int b2 = b1 + 21;
    int i0 = (b1 - 16) / 32;
    int i2 = (b2 - 1) / 32;
    x_s2 = (i2 + 1) * 32 - b2;
    x_src_a = i0 % 24;
    x_src_b = i2 % 24;
  }

  __shared__ float sh_red[WK][COLS];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
      if constexpr (bits == 3)
        return lane < 24 ? __builtin_nontemporal_load(bp + (size_t)i * slice_stride +
                                                          l * LSTRIDE)
                         : 0;
      else
        return __builtin_nontemporal_load(bp + (size_t)i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    float acc0[WNT] = {}, acc1[WNT] = {};
    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }

        const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
        const half2 a01 = shA[a_col];
        const half2 a23 = shA[a_col + 4];

#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          if constexpr (bits == 4) {
            uint32_t aw = __shfl_sync(0xffffffffffffffffULL, bw[t], (lane + 31) & 31);
            dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
          } else if constexpr (bits == 2) {
            const uint32_t w = bw[t >> 1];
            const int base = (t & 1) << 4;
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_b);
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_a);
            dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
          } else {
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_a);
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }

          acc0[t] = amd_mixed_dot(f0[0], a01, acc0[t], false);
          acc0[t] = amd_mixed_dot(f0[1], a23, acc0[t], false);
          acc1[t] = amd_mixed_dot(f1[0], a01, acc1[t], false);
          acc1[t] = amd_mixed_dot(f1[1], a23, acc1[t], false);
        }
      }
    }

    const int c0 = lane >> 2;
#pragma unroll
    for (int t = 0; t < WNT; ++t) {
      float v0 = acc0[t], v1 = acc1[t];
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 1);
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 2);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 1);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 2);
      if ((lane & 3) == 0) {
        sh_red[warp][t * 16 + c0] = v0;
        sh_red[warp][t * 16 + c0 + 8] = v1;
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    __syncthreads();
  }
}

// ─── the try-launch (exl3_gemv.cu:29-179) ────────────────────────────────────

constexpr bool GemvArmInstantiated(int bits, int cb) {
  // Upstream's selector table (exl3_gemv.cu SEL_GRID list), read as a
  // predicate: 4 bpw decodes all three codebooks, 2 and 3 bpw instantiate
  // cb 1/2 only (cb 0 at those widths is refused upstream, exl3_gemv.cu:113).
  // ROCm extension: 6 bpw m=1 — this checkpoint's lm_head — is served by the
  // span-decode m1 kernel; upstream has no 6 bpw arm at all.
  return (bits == 4 && (cb == 0 || cb == 1 || cb == 2)) ||
         ((bits == 2 || bits == 3) && (cb == 1 || cb == 2)) ||
         ((bits == 2 || bits == 3) && (cb == 1 || cb == 2));
}

const void* GemvM1KernelFor(int bits, int cb) {
  // The direct m=1 arm's instantiation set is the non-WMMA half of upstream's
  // selector table minus its smem rows (this kernel stages nothing).
#define VT_GEMV_M1(bits_, cb_)                                              \
  if (bits == bits_ && cb == cb_)                                           \
    return reinterpret_cast<const void*>(&Exl3GemvM1K<bits_, cb_>);
#define VT_GEMV_M1W4(bits_, cb_)                                            \
  if (bits == bits_ && cb == cb_)                                           \
    return reinterpret_cast<const void*>(&Exl3GemvM1K4<cb_>);
  // The wide-load ownership arm is instantiated for the 4 bpw rows only —
  // its codeword-span arithmetic is the 4 bpw window math; 2/3 bpw keep the
  // dq8-mapping arm.
  VT_GEMV_M1W4(4, 0) VT_GEMV_M1W4(4, 1) VT_GEMV_M1W4(4, 2)
  VT_GEMV_M1(2, 1) VT_GEMV_M1(2, 2)
  VT_GEMV_M1(3, 1) VT_GEMV_M1(3, 2)
#undef VT_GEMV_M1W4
#undef VT_GEMV_M1
  return nullptr;
}
const void* GemvM1FusedKernelFor(int bits, int cb) {
  // The fused in-had arm's instantiation set mirrors GemvM1KernelFor's: same
  // (bits, cb) coverage, one kernel per row.
#define VT_GEMV_M1F(bits_, cb_)                                             \
  if (bits == bits_ && cb == cb_)                                           \
    return reinterpret_cast<const void*>(&Exl3GemvM1KFused<bits_, cb_>);
#define VT_GEMV_M1F4(bits_, cb_)                                            \
  if (bits == bits_ && cb == cb_)                                           \
    return reinterpret_cast<const void*>(&Exl3GemvM1K4Fused<cb_>);
  VT_GEMV_M1F4(4, 0) VT_GEMV_M1F4(4, 1) VT_GEMV_M1F4(4, 2)
  VT_GEMV_M1F(2, 1) VT_GEMV_M1F(2, 2)
  VT_GEMV_M1F(3, 1) VT_GEMV_M1F(3, 2)
#undef VT_GEMV_M1F4
#undef VT_GEMV_M1F
  return nullptr;
}

const void* GemvKernelFor(int bits, int cb, int mmode, int cfg, bool smem, bool wmma) {
  // The selector spells every instantiation so a typo in the table is a
  // compile error, not a wrong kernel.
#define VT_GEMV_SEL(bits_, cb_, mm_, cfg_, sm_, wm_)                                    \
  if (bits == bits_ && cb == cb_ && mmode == mm_ && cfg == cfg_ && smem == sm_ &&       \
      wmma == wm_)                                                                      \
    return reinterpret_cast<const void*>(&Exl3GemvK<bits_, cb_, mm_, cfg_, sm_, wm_>);
#define VT_GEMV_GRID(bits_, cb_, sm_, wm_)                                              \
  VT_GEMV_SEL(bits_, cb_, 0, 0, sm_, wm_) VT_GEMV_SEL(bits_, cb_, 0, 1, sm_, wm_)       \
      VT_GEMV_SEL(bits_, cb_, 1, 0, sm_, wm_) VT_GEMV_SEL(bits_, cb_, 1, 1, sm_, wm_)

  if (!wmma) {
    // Upstream's table minus its c_fp32=false row: C here is always the f32
    // scratch, so only the fp32-write arm exists on this TU. smem staging is
    // instantiated where upstream instantiates it (2 and 3 bpw only).
    VT_GEMV_GRID(4, 0, false, false) VT_GEMV_GRID(4, 1, false, false)
        VT_GEMV_GRID(4, 2, false, false)
        VT_GEMV_GRID(2, 1, false, false) VT_GEMV_GRID(2, 2, false, false)
        VT_GEMV_GRID(2, 1, true, false) VT_GEMV_GRID(2, 2, true, false)
        VT_GEMV_GRID(3, 1, false, false) VT_GEMV_GRID(3, 2, false, false)
        VT_GEMV_GRID(3, 1, true, false) VT_GEMV_GRID(3, 2, true, false)
  } else {
#if VT_ROCM_EXL3_WMMA_INST
    // WMMA is evaluation-only (header, adaptation 2): the 4 bpw arms, which is
    // the decode shape this row serves.
    VT_GEMV_GRID(4, 0, false, true) VT_GEMV_GRID(4, 1, false, true)
        VT_GEMV_GRID(4, 2, false, true)
#endif
  }
#undef VT_GEMV_GRID
#undef VT_GEMV_SEL
  return nullptr;
}

int GemvOccupancyRocm(const void* kernel, int block_dim) {
  // exl3_gemv.cu:120-134 + exllamav3-rocm's RDNA correction: the occupancy
  // query overestimates co-residency on RDNA (reported 3, real 2, on gfx1100),
  // so the reported count is decremented once. Without grid.sync the number no
  // longer decides correctness — only which cfg the heuristic picks — and the
  // correction is kept so the heuristic stays conservative, not adventurous.
  static std::mutex mtx;
  static std::map<const void*, int> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(kernel);
  if (it != cache.end()) return it->second;
  int blocks_per_sm = 0;
  CheckG(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0),
         "hipOccupancyMaxActiveBlocksPerMultiprocessor exl3_gemv");
  if (blocks_per_sm > 1) blocks_per_sm -= 1;
  cache[kernel] = blocks_per_sm;
  return blocks_per_sm;
}

bool GemvWmmaMode() {
  // VT_ROCM_EXL3_WMMA — documented in docs/ENVIRONMENT.md. Evaluation only:
  // the default (unset/0) is the emulated path the gates measure.
  static const int mode = [] {
    const char* env = std::getenv("VT_ROCM_EXL3_WMMA");
    return env ? std::atoi(env) : 0;
  }();
  return mode == 1;
}

}  // namespace

// Returns true when the GEMV arm launched. The same decline contract the CUDA
// arm documents: a false here is a fall-through to Exl3GemmK, never a failure.
bool Exl3GemvTryLaunchRocm(hipStream_t st, const Tensor& a_had, const Tensor& trellis,
                           float* raw, int size_m, int size_k, int size_n, int device,
                           const Exl3GemmArgs& args) {
  // exl3_gemv.cu:108-116: the free integer tests first, then the env read.
  if (args.force_gemv == 0) return false;
  if (!GemvArmInstantiated(args.bits, args.codebook)) return false;
  // The m1 arm is the only fast arm for bits > 4: Exl3GemvHardEligible's
  // bits<=4 test mirrors upstream's fragment kernel, not the span decoder.
  if (!Exl3GemvHardEligible(size_m, size_k, size_n, args.bits, args.codebook,
                            /*has_su_sv=*/true) &&
      !(size_m == 1 && args.bits >= 5 && args.bits <= 8 &&
        size_k % 128 == 0 && size_n % 128 == 0))
    return false;
  const int mode = args.force_gemv > 0 ? 2 : Exl3GemvMode();
  if (mode == 0) return false;
  const int mmode = size_m == 1 ? 0 : 1;
  const bool smem = Exl3GemvSmemMode() == 1;
  const bool wmma = GemvWmmaMode();

  // wave32 is structural: the A/B fragment gathers and the bits-3 24-lane
  // window are written for 32-lane warps. A wave64 build takes the scalar arm
  // rather than decode wrong.
  int warp_size = 0;
  CheckG(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device),
         "warp size query");
  if (warp_size != 32) return false;

  int num_sms = 0;
  CheckG(hipDeviceGetAttribute(&num_sms, hipDeviceAttributeMultiprocessorCount, device),
         "multiprocessor count");
  if (num_sms < 1) num_sms = 1;

  const void* narrow = GemvKernelFor(args.bits, args.codebook, mmode, 0, smem, wmma);
  // bits>4 has no fragment arm at all — narrow is null there, the cfg
  // heuristic is meaningless for it, and only the m=1 span arm can serve it
  // (the m1 branch below); other ms fall to the dot/transcription arms in
  // the caller.
  if (narrow == nullptr && !(mmode == 0 && args.bits >= 5 && args.bits <= 8))
    return false;

  // VT_EXL3_GEMV_CFG forces the upstream cfg choice — a tuning probe, not a
  // contract: 0 is the 512-thread narrow arm, 1 the 256-thread WNT=4 arm.
  static const int kCfgEnv = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_CFG");
    return e ? std::atoi(e) : -1;
  }();

  int cfg = 0;
  if (narrow != nullptr) {
    const int narrow_coresident = GemvOccupancyRocm(narrow, 512) * num_sms;

    // exllamav3-rocm passes CC_RDNA3 into this heuristic, where every
    // RDNA3-reachable arm lands on the same decision as CC_ADA (K==3 gate)
    // or the fallthrough. Exl3Cc has no gfx bucket and gains none: passing
    // kAda is the same function evaluated, not an approximation of it.
    const int cfg0 =
        Exl3GemvSelectConfig(Exl3Cc::kAda, size_m, size_k, size_n, args.bits,
                             args.codebook, mode, narrow_coresident);
    // ROCm extension (spec rocm-exl3-gemv-residual §R1): a -1 here is
    // upstream's Ampere-tuned heuristic having no opinion. On this backend
    // the alternative is the byte-exact scalar transcription (~30 GB/s
    // effective), never another fast arm, so a hard-eligible call falls
    // back to upstream's own forced rule (exl3_gemv.cu:66-67).
    cfg = kCfgEnv >= 0 ? kCfgEnv
                       : (cfg0 >= 0 ? cfg0 : (size_n <= 8192 ? 0 : 1));
  }

  // m=1 decode takes the direct arm: same launch geometry as cfg 0 but the
  // fragment gather + fp16-fragment MMA replaced by four v_dot2_f32_f16 per
  // lane-pair — ~6x fewer issued instructions per tile (SASS-measured), and
  // f32 accumulation where the fragment arm used f16 pairs. The WMMA env
  // still selects the WMMA evaluation arm for A/B completeness.
  const void* kernel;
  int block_dim, cols;
  if (mmode == 0 && !wmma && kCfgEnv < 0) {
    kernel = GemvM1KernelFor(args.bits, args.codebook);
    block_dim = 512;
    // M1K4's warp owns a four-tile group (64 columns); the dq8 arm's warp
    // covers two tiles (32).
    cols = (args.bits == 4) ? 64 : 32;
  } else {
    kernel = cfg == 0 ? narrow
                      : GemvKernelFor(args.bits, args.codebook, mmode, cfg, smem,
                                      wmma);
    block_dim = cfg == 0 ? 512 : 256;
    cols = cfg == 0 ? 32 : 64;
  }

  // Lab diagnostic: VT_EXL3_GEMV_DUMP=1 prints one stderr line per launch with
  // the resolved shape. Off by default; launch sites only.
  static const bool kDump = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_DUMP");
    return e != nullptr && e[0] == '1';
  }();
  if (kDump)
    std::fprintf(stderr,
                 "[exl3_gemv] m=%d k=%d n=%d bits=%d cb=%d cols=%d block=%d\n",
                 size_m, size_k, size_n, args.bits, args.codebook, cols,
                 block_dim);
  if (kernel == nullptr) return false;

  const int max_blocks = GemvOccupancyRocm(kernel, block_dim) * num_sms;
  const int grid = size_n / cols < max_blocks ? size_n / cols : max_blocks;
  if (grid < 1) return false;

  // Grid-level k-split (the kernel halves its per-warp serial k-chain per
  // extra blockIdx.y) is a bench lever first: VT_EXL3_GEMV_KSPLIT=N raises
  // grid.y to N and pays one f32 atomicAdd per (m, stripe col) per slice.
  // It stays OFF by default until serving measurements pick a value.
  static const int kKsplitEnv = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_KSPLIT");
    return e ? std::atoi(e) : 1;
  }();
  const int ksplit = std::max(
      1, std::min(kKsplitEnv, (size_k / 16) / 16));
  if (ksplit > 1)
    CheckG(hipMemsetAsync(raw, 0, (size_t)size_m * size_n * sizeof(float), st),
           "exl3_gemv ksplit zero");
  const half* a_ptr = a_had.Ptr<half>();
  const uint16_t* b_ptr = trellis.Ptr<uint16_t>();
  // Normal launch, not hipLaunchCooperativeKernel — adaptation 1. The kernel
  // pointer table types it as void*, so dispatch by cfg size.
  void* kargs[] = {static_cast<void*>(&a_ptr),  static_cast<void*>(&b_ptr),
                   static_cast<void*>(&raw),    static_cast<void*>(&size_m),
                   static_cast<void*>(&size_k), static_cast<void*>(&size_n)};
  CheckG(hipLaunchKernel(kernel,
                         dim3(static_cast<unsigned>(grid),
                              static_cast<unsigned>(ksplit)),
                         dim3(static_cast<unsigned>(block_dim)), kargs, 0, st),
         "exl3_gemv launch");
  CheckG(hipGetLastError(), "exl3_gemv launch");
  return true;
}
// Spec gfx1100-exl3-decode-60tps: the fused in-had m=1 arm. Runs the
// HadBlock128ToH2 staging phase (bit-identical to HadK<true,true>) inside the
// GEMV kernel, so the caller must NOT launch the separate input hadamard when
// this returns true. Declines on the same terms Exl3GemvTryLaunchRocm does,
// plus m != 1, an uninstantiated fused (bits, cb), a null suh (the kernel
// dereferences it), or size_k not a multiple of 128 (the had block size).
bool Exl3GemvM1FusedTryLaunchRocm(hipStream_t st, const void* a_raw,
                                  const uint16_t* suh_ptr,
                                  const Tensor& trellis, float* raw,
                                  int size_k, int size_n, int device,
                                  const Exl3GemmArgs& args) {
  if (args.force_gemv == 0) return false;
  // Same hard gates as the unfused arm — n%128 included. Without it a
  // shape like n%64==0 && n%128!=0 was accepted and the group loop left
  // trailing columns of C stale (reviewer finding on 1d04d72dc).
  if (!Exl3GemvHardEligible(1, size_k, size_n, args.bits, args.codebook,
                            /*has_su_sv=*/true))
    return false;
  if (suh_ptr == nullptr || a_raw == nullptr) return false;
  const int mode = args.force_gemv > 0 ? 2 : Exl3GemvMode();
  if (mode == 0) return false;
  if (!GemvArmInstantiated(args.bits, args.codebook)) return false;
  static const int kFuseHadEnv = [] {
    const char* e = std::getenv("VT_EXL3_FUSED_HAD");
    return e ? std::atoi(e) : 1;
  }();
  if (kFuseHadEnv != 1) return false;
  if (GemvWmmaMode()) return false;  // fused arm is the dot decode, not WMMA

  int warp_size = 0;
  CheckG(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device),
         "warp size query");
  if (warp_size != 32) return false;
  int num_sms = 0;
  CheckG(hipDeviceGetAttribute(&num_sms, hipDeviceAttributeMultiprocessorCount, device),
         "multiprocessor count");
  if (num_sms < 1) num_sms = 1;

  const void* kernel = GemvM1FusedKernelFor(args.bits, args.codebook);
  if (kernel == nullptr) return false;
  const int cols = (args.bits == 4) ? 64 : 32;

  // Dynamic LDS is the whole a_had row: size_k halves. Above the 48 KiB
  // hipLaunchKernel limit the arm declines (bigger-k models keep the
  // separate-had path); hipFuncSetAttributeOpt-in is a later row.
  const size_t shmem = static_cast<size_t>(size_k) * sizeof(uint16_t);
  if (shmem > 48 * 1024) return false;

  int max_blocks = GemvOccupancyRocm(kernel, 512);
  if (max_blocks < 1) max_blocks = 1;
  const int grid = size_n / cols < max_blocks * num_sms ? size_n / cols
                                                        : max_blocks * num_sms;
  if (grid < 1) return false;

  static const int kKsplitEnv = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_KSPLIT");
    return e ? std::atoi(e) : 1;
  }();
  const int ksplit = std::max(1, std::min(kKsplitEnv, (size_k / 16) / 16));
  if (ksplit > 1)
    CheckG(hipMemsetAsync(raw, 0, (size_t)size_n * sizeof(float), st),
           "exl3_gemv fused ksplit zero");


  static const bool kDumpF = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_DUMP");
    return e != nullptr && e[0] == '1';
  }();
  if (kDumpF)
    std::fprintf(stderr,
                 "[exl3_gemv] fused m=1 k=%d n=%d bits=%d cb=%d grid=%d lds=%zu\n",
                 size_k, size_n, args.bits, args.codebook, grid, shmem);
  const uint16_t* a_ptr = static_cast<const uint16_t*>(a_raw);
  const uint16_t* b_ptr = trellis.Ptr<uint16_t>();
  int size_m = 1;
  void* kargs[] = {static_cast<void*>(&a_ptr),  static_cast<void*>(&suh_ptr),
                   static_cast<void*>(&b_ptr),  static_cast<void*>(&raw),
                   static_cast<void*>(&size_m), static_cast<void*>(&size_k),
                   static_cast<void*>(&size_n)};
  CheckG(hipLaunchKernel(kernel,
                         dim3(static_cast<unsigned>(grid),
                              static_cast<unsigned>(ksplit)),
                         dim3(512), kargs, shmem, st),
         "exl3_gemv fused-had launch");
  CheckG(hipGetLastError(), "exl3_gemv fused-had launch");
  return true;
}

// ─── spec rocm-exl3-gemv-residual: the residual dispatch surface ─────────────
//
// Exl3GemmKernelRocm needs two more facts from this TU: whether the shape
// itself can take the GEMV arm regardless of m (the chunked launcher feeds it
// 8-row slices), and a chunked variant for 8 < m <= 32. Both stay inside
// Exl3GemvMode() != 0 — VT_EXL3_GEMV=0 selects the transcription for EVERY
// shape on this backend, which is what the byte-exact gate relies on.

bool Exl3GemvShapeEligibleRocm(int size_k, int size_n, int bits, int cb) {
  // The m term of Exl3GemvHardEligible is evaluated per chunk (always <= 8),
  // so shape eligibility is the predicate with m=1.
  return GemvArmInstantiated(bits, cb) &&
         Exl3GemvHardEligible(1, size_k, size_n, bits, cb, /*has_su_sv=*/true);
}

bool Exl3GemvChunkedTryLaunchRocm(hipStream_t st, const Tensor& a_had,
                                  const Tensor& trellis, float* raw, int size_m,
                                  int size_k, int size_n, int device,
                                  const Exl3GemmArgs& args) {
  // Every chunk runs upstream's own m<=8 arm (mmode 1) on an 8-row slice of
  // a_had and raw. B is re-read per chunk — upstream's own trade-off for
  // mmode 1 — so the caller caps this path at m <= 32, above which the
  // reconstruct arm's one-pass decode + hipBLAS GEMM is the right shape.
  for (int base = 0; base < size_m; base += kGemvMaxM) {
    const int rows = std::min(kGemvMaxM, size_m - base);
    Tensor a_slice = a_had;
    a_slice.data = static_cast<char*>(a_had.data) +
                   static_cast<size_t>(base) * static_cast<size_t>(size_k) * sizeof(half);
    float* raw_slice = raw + static_cast<size_t>(base) * static_cast<size_t>(size_n);
    if (!Exl3GemvTryLaunchRocm(st, a_slice, trellis, raw_slice, rows, size_k, size_n,
                               device, args))
      return false;
  }
  return true;
}


}  // namespace vt::rocm
