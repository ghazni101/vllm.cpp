// Shared EXL3 trellis decode — the scalar codec three ROCm EXL3 TUs use.
//
// Source of every function: the portable CPU reference's device transcription
// in rocm_exl3.hip (itself a 1:1 port of exllamav3 @
// 2398c05635fbbad01a0a51dce63c85c6c8a8450e — exl3_dq.cuh:15-31, codebook.cuh,
// quantize.py:22-42). Extracted so rocm_exl3.hip (the fused GEMM), its GEMV
// sibling and rocm_exl3_recon.hip (the reconstruct+GEMM arm) decode identically
// rather than three copies drifting.
//
// __device__ inline — header-only by construction, same shape as
// rocm_f16_codec.h which it depends on.
#pragma once

#include <cstdint>

#include "vt/rocm/rocm_f16_codec.h"

namespace vt::rocm {

// cpu_exl3_dequant.cpp RoundHalf.
__device__ inline float Exl3RoundHalf(float v) { return DF16ToF32(DF32ToF16(v)); }

// The tile's uint32 view (exl3_dq.cuh:25-26), assembled by hand so the trellis
// may sit at any alignment inside the loaded buffer.
__device__ inline uint32_t Exl3TileWord32(const uint16_t* tile, int index) {
  return static_cast<uint32_t>(tile[2 * index]) |
         (static_cast<uint32_t>(tile[2 * index + 1]) << 16);
}

// Exl3TileCodeword, verbatim. `+ 256*bits` keeps the tail-biting wrap
// non-negative; the `% words32` on the word index is the wrap itself.
__device__ inline uint16_t Exl3TileCodeword(const uint16_t* tile, int bits, int t) {
  const int words32 = bits * 256 / 32;
  const int b0 = t * bits + bits - 16 + 256 * bits;
  const int b1 = b0 + 16;
  const int i0 = b0 / 32;
  const int i1 = (b1 - 1) / 32;
  const int s0 = (i1 + 1) * 32 - b1;
  const uint32_t a = Exl3TileWord32(tile, i0 % words32);
  const uint32_t b = Exl3TileWord32(tile, i1 % words32);
  const uint64_t merged = (static_cast<uint64_t>(a) << 32) | b;
  return static_cast<uint16_t>((merged >> s0) & 0xffffu);
}

// Exl3DecodeCodeword (codebook.cuh:56-90). All THREE codebooks, because an AMD
// box has no reason to see fewer artifacts than an NVIDIA one:
//   cb 0  3INST, the DEFAULT — a checkpoint that ships neither an `mcg` nor a
//         `mul1` marker lands here, which is every stock turboderp/*-exl3
//   cb 1  MCG, which the SparkInfer DeepSeek-V4 artifact marks
//   cb 2  mul1, a DIFFERENT SHAPE and not a third multiplier: the product's
//         four UNSIGNED BYTES are summed into a fixed accumulator, the sum is
//         REINTERPRETED as an fp16 bit pattern, and an fp16 affine map turns
//         it into the codebook value
// `codebook` is validated on the host before launch, so there is no fall-off
// arm here; a device VT_CHECK does not exist and a silent wrong decode is
// exactly the failure this family documents (right distribution, no
// correlation).
__device__ inline float Exl3DecodeCodeword(uint16_t codeword, int codebook) {
  uint32_t x = static_cast<uint32_t>(codeword);
  if (codebook == 2) {
    x *= 0x83DCD12Du;
    // `__dp4a(x, 0x01010101u, acc)` == acc + the four UNSIGNED bytes of x.
    // Scalar here for the same reason rocm_grouped_gemm.hip's Dp4a is scalar:
    // integer arithmetic is exact either way, and v_dot4 is a perf lever, not
    // a correctness requirement.
    const uint32_t byte_sum = (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) +
                              ((x >> 24) & 0xffu);
    // 0x6400 is chosen so the reinterpretation is EXACT: the fp16 binade
    // [1024, 2048) has an ULP of exactly 1.0 and the byte sum is at most 1020,
    // so the pattern never leaves that binade.
    const uint32_t sum = 0x6400u + byte_sum;
    const float h = DF16ToF32(static_cast<uint16_t>(sum));
    // BIT PATTERNS, not the rounded decimals upstream's comments carry:
    // 0x1eee is 887/131072 and 0xc931 is -1329/128.
    const float k_inv = DF16ToF32(static_cast<uint16_t>(0x1eeeu));
    const float k_bias = DF16ToF32(static_cast<uint16_t>(0xc931u));
    // Upstream ends in __hfma, ONE rounding. Evaluating the product and the
    // sum separately in f32 reproduces it exactly (the CPU arm proves why over
    // all 1021 reachable sums), and -ffp-contract=off keeps the compiler from
    // fusing them into a different single rounding.
    return Exl3RoundHalf(h * k_inv + k_bias);
  }
  if (codebook == 0) {
    x *= 89226354u;
    x += 64248484u;
  } else {
    x *= 0xCBAC1FEDu;
  }
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  // The halves of x are the two f16 codebook contributions; their f16 sum
  // is the weight. f32-add of two f16-exact values is exact in f32, so RN
  // back to f16 equals __hadd — one V_ADD_F16 replaces both DF16ToF32
  // converts, the f32 add, and Exl3RoundHalf.
  const half2 pair = __builtin_bit_cast(half2, x);
  return __hadd(__low2half(pair), __high2half(pair));
}

// Exl3TileRowMajorIndex (quantize.py:28-42). `t / 8` is the tensor-core lane,
// `t % 8` its eight fragment slots. The host's kRowOffset[8] = {0,1,8,9,0,1,8,9}
// is written as arithmetic here so no constant array lands in scratch.
__device__ inline int Exl3TileRowMajorIndex(int t) {
  const int lane = t >> 3;
  const int sub = t & 7;
  const int q = sub & 3;
  const int row_offset = q < 2 ? q : q + 6;  // 0,1,8,9
  const int r = (lane % 4) * 2 + row_offset;
  const int c = lane / 4 + (sub < 4 ? 0 : 8);
  return r * 16 + c;
}

// The INVERSE of Exl3TileRowMajorIndex: the codeword index `t` whose decoded
// weight lands at row-major position (r, c) inside the 16x16 tile. The
// reconstruct kernel enumerates output elements, not codewords, so each
// codeword is decoded exactly once and writes stay coalesced. Derived from the
// map's own arithmetic: row_offset is {0,1} for r < 8 and {8,9} for r >= 8,
// lane%4 is r/2 mod 4 in both halves.
__device__ inline int Exl3TileRowMajorIndexInv(int r, int c) {
  const int l4 = (r < 8) ? (r >> 1) : ((r - 8) >> 1);
  const int ro = r - 2 * l4;                 // 0,1 or 8,9
  const int q = (ro < 2) ? ro : (ro - 6);    // undo the q -> q+6 shift
  const int lane = (c & 7) * 4 + l4;
  const int sub = q + ((c & 8) ? 4 : 0);
  return lane * 8 + sub;
}

}  // namespace vt::rocm
