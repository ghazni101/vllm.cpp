// ROCm GDN recurrence core (BACKEND-ROCM-GDN-KERNELS family 4; issue #41).
// kGdnPrefill + kGdnDecode over the PORTABLE scan — hand-translation of
// cuda_gdn.cu GdnScanKernel (:1856) + its launchers, readable side by side
// against the donor. CPU oracle: src/vt/cpu/cpu_ops.cpp:1331/1368
// (GdnHeadTokenStep); formula reference gdn-semantics.md §7/§8. One kernel
// serves both ops (qsl != null => varlen prefill; qsl == null => single-step
// decode with optional state_idx + NULL-block zero-out).
//
// Deliberately NOT ported (docs/ROCM.md §6): the WMMA chunked prefill
// (`#if __CUDA_ARCH__ >= 800`), the fused/packed/Triton-AOT decode variants —
// NVIDIA-only machinery; the scan is the correctness baseline and the perf
// variants are M5 levers (upstream's ROCm answer, Triton/CK, is a later row).
//
// Gate: the recurrence case in test_backend_cross_device.cpp — NMSE <= 5e-4
// vs the CPU oracle on out AND in-place state; decode NULL-slot zero-out
// EXACT.

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm gdn: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}
inline unsigned GridFor(int64_t n) {
  if (n <= 0) return 1;
  const int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(g > 65535 ? 65535 : g);
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

// ── Recurrence scan (donor cuda_gdn.cu:1856; serves prefill AND decode) ─────
template <typename Tin, typename Tout, typename TState>
__global__ void GdnScanK(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                         const float* g, const float* beta, TState* state,
                         const int32_t* qsl, const int32_t* state_idx,
                         int64_t state_slots, int64_t hk_n, int64_t dk,
                         int64_t hv_n, int64_t dv, float scale) {
  const int64_t s = blockIdx.y;
  const int64_t hv = blockIdx.x;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t state_slot = state_idx != nullptr ? state_idx[s] : s;
  if (state_slot < 0 || state_slot >= state_slots) {
    const int64_t begin = qsl != nullptr ? qsl[s] : s;
    const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
    for (int64_t t = begin; t < end; ++t)
      for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x)
        St(out, (t * hv_n + hv) * dv + vi, 0.0f);
    return;
  }
  extern __shared__ float smem[];  // [dk] q' then [dk] k
  float* q_sh = smem;
  float* k_sh = smem + dk;
  TState* s_head = state + (state_slot * hv_n + hv) * dv * dk;
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Ld(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Ld(k, (t * hk_n + hk) * dk + i);
    }
    __syncthreads();
    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
      TState* s_row = s_head + vi * dk;
      float dot = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float decayed = Ld(s_row, ki) * decay;
        dot += decayed * k_sh[ki];
      }
      const float vp = (Ld(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float updated = Ld(s_row, ki) * decay + vp * k_sh[ki];
        St(s_row, ki, updated);
        o += updated * q_sh[ki];
      }
      St(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();
  }
}

template <typename Tin, typename Tout>
void LaunchGdnScanState(hipStream_t s, Tensor& out, const Tensor& q_in,
                        const Tensor& k, const Tensor& v, const Tensor& g,
                        const Tensor& beta, Tensor& state, const int32_t* qsl,
                        const int32_t* state_idx, int64_t n,
                        const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const dim3 grid(static_cast<unsigned>(hv_n), static_cast<unsigned>(n));
  const size_t shmem = 2 * static_cast<size_t>(dk) * sizeof(float);
  if (state.dtype == DType::kF16) {
    GdnScanK<Tin, Tout, __half><<<grid, kBlock, shmem, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
        g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__half>(), qsl, state_idx,
        state.shape[0], hk_n, dk, hv_n, dv, args.scale);
  } else if (state.dtype == DType::kBF16) {
    GdnScanK<Tin, Tout, __hip_bfloat16><<<grid, kBlock, shmem, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
        g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__hip_bfloat16>(), qsl,
        state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale);
  } else {
    GdnScanK<Tin, Tout, float><<<grid, kBlock, shmem, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
        g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<float>(), qsl, state_idx,
        state.shape[0], hk_n, dk, hv_n, dv, args.scale);
  }
  Check(hipGetLastError(), "gdn scan launch");
}

template <typename Tin>
void LaunchGdnScanIn(hipStream_t s, Tensor& out, const Tensor& q_in,
                     const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, Tensor& state, const int32_t* qsl,
                     const int32_t* state_idx, int64_t n, const GdnArgs& args) {
  if (out.dtype == DType::kBF16)
    LaunchGdnScanState<Tin, __hip_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl,
                                            state_idx, n, args);
  else
    LaunchGdnScanState<Tin, float>(s, out, q_in, k, v, g, beta, state, qsl,
                                   state_idx, n, args);
}

// ===========================================================================
// Chunk-parallel GDN prefill scan (port of cuda_gdn.cu:3237 chunked path).
// Splits the sequence into kChunk-token chunks: intra-chunk work (steps A/B/D)
// is parallel across chunks; only the cross-chunk state recurrence (step C) is
// sequential. DeltaH step C uses WMMA for the state-update GEMM.
// ===========================================================================
constexpr int kChunk = 64;
constexpr int kChunkMaxDim = 128;
constexpr int kBV = 64;  // vi-tiling for DeltaH (dv/kBV blocks per (seq,v-head))
constexpr int WM = 16, WN = 16, WK = 16;  // WMMA tile

bool ChunkedPrefillEnabled() {
  const char* e = std::getenv("VT_GDN_CHUNKED");
  return e == nullptr || e[0] != '0';
}

// Step A — chunk_local_cumsum: inclusive prefix sum of g within each chunk.
__global__ void GdnChunkCumsumKR(float* gcum, const float* g, const int32_t* tok0a,
                                 const int32_t* lena, int64_t hv_n) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  if (threadIdx.x != 0) return;
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  float acc = 0.0f;
  for (int64_t i = 0; i < len; ++i) {
    acc += g[(tok0 + i) * hv_n + hv];
    gcum[(tok0 + i) * hv_n + hv] = acc;
  }
}

// Step B — WY decomposition (VEC path): KK^T via WMMA, triangular inverse,
// then WMMA apply: u = T @ (beta*v), w = T @ (beta*eg*k).
// Replaces per-column forward substitution (73% of scalar WU time) with
// a single triangular inverse + two WMMA matrix multiplies.
// Shared (40.75KB): R0[16KB dyn] (Ks→Tf→Bop) + R1[24KB stat] (Am→Tb+Osh) + gs/bs/eg
template <typename Tin>
__global__ void GdnChunkWUKR(float* u, float* w, const Tin* k, const Tin* v,
                             const float* beta, const float* gcum,
                             const int32_t* tok0a, const int32_t* lena,
                             int64_t hk_n, int64_t dk, int64_t hv_n, int64_t dv) {
  using namespace rocwmma;
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int kOBlk = 64;  // WMMA apply output column block
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int tid = threadIdx.x;
  const int wf = tid / 32, n_wfs = blockDim.x / 32;

  // Dynamic shared: R0 = max(Ks[BT,dk] half, Tf[BT,BT] float, Bop[BT,dcol] half)
  //  = max(16KB, 16KB, 16KB) = 16KB
  extern __shared__ __half R0_dyn[];
  __half* Ks = R0_dyn;
  // Static shared: R1 = Am[BT,BT] float (16KB) → Tb[BT,BT] half (8KB) + Osh[BT,kOBlk] float (16KB) = 24KB
  __shared__ float R1[kChunk * kChunk + kChunk * kOBlk];  // 16KB + 16KB = 32KB
  float* Am = R1;
  __half* Tb = reinterpret_cast<__half*>(R1);  // aliases Am after inverse
  float* Osh = R1 + kChunk * kChunk;             // Osh after Tb (16KB)
  __shared__ float gs[kChunk], bs[kChunk], eg[kChunk];

  // 1. Stage K [kChunk, dk] as half (zero-padded past len)
  for (int64_t idx = tid; idx < kChunk * dk; idx += blockDim.x) {
    int64_t i = idx / dk, d = idx % dk;
    Ks[idx] = (i < len)
      ? __float2half(__bfloat162float(Ld(k, (tok0 + i) * hk_n * dk + hk * dk + d)))
      : __float2half(0.0f);
  }
  for (int64_t i = tid; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // 2. KK^T via WMMA: Am[i,j] = sum_d Ks[i,d] * Ks[j,d]
  {
    const int M_t = kChunk / WM, N_t = kChunk / WN, K_t = dk / WK;
    for (int t = wf; t < M_t * N_t; t += n_wfs) {
      const int mt = t / N_t, nt2 = t % N_t;
      fragment<accumulator, WM, WN, WK, float> acc;
      fill_fragment(acc, 0.0f);
      for (int kt = 0; kt < K_t; ++kt) {
        fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
        fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag;
        load_matrix_sync(a_frag, Ks + mt * WM * dk + kt * WK, (uint32_t)dk);
        load_matrix_sync(b_frag, Ks + nt2 * WN * dk + kt * WK, (uint32_t)dk);
        mma_sync(acc, a_frag, b_frag, acc);
      }
      store_matrix_sync(Am + mt * WM * kChunk + nt2 * WN, acc,
                        (uint32_t)kChunk, mem_row_major);
    }
  }
  __syncthreads();

  // 3. Fold beta*exp(G_i-G_j) into Am: Am[i,j] = bs[i]*exp(gs[i]-gs[j])*Am[i,j] for j<i, else 0
  for (int64_t e = tid; e < kChunk * kChunk; e += blockDim.x) {
    int64_t i = e / kChunk, j = e % kChunk;
    Am[e] = (i < len && j < i) ? bs[i] * expf(gs[i] - gs[j]) * Am[e] : 0.0f;
  }
  __syncthreads();  // Ks (R0) free → Tf aliases it

  // 4. Triangular inverse: T = (I + Am)^-1, unit lower-triangular
  // BLOCKED path (port of CUDA VT_GDN_WY_BLOCKED): 4 diagonal 16×16 blocks
  // solved in parallel by 4 warps, then 6 off-diagonal blocks filled by
  // scalar Schur merges. Cuts critical path from ~2016 to ~216 dependent FMAs.
  float* Tf = reinterpret_cast<float*>(R0_dyn);  // reuse R0 (Ks no longer needed)
  for (int64_t e = tid; e < kChunk * kChunk; e += blockDim.x) Tf[e] = 0.0f;
  __syncthreads();
  constexpr int kBlk = 16;  // block size for blocked inverse
  constexpr int kNBlk = kChunk / kBlk;  // 4
  // Diagonal blocks: warp w solves block (w,w)
  if (wf < kNBlk) {
    const int blk = wf;
    const int base = blk * kBlk;
    for (int c = tid % 32; c < kBlk; c += 32) {
      const int gc = base + c;  // global column
      Tf[gc * kChunk + gc] = 1.0f;
      if (gc < len) {
        for (int i = c + 1; i < kBlk && base + i < len; ++i) {
          const int gi = base + i;
          float sfs = 0.0f;
          for (int m = c; m < i; ++m)
            sfs += Am[gi * kChunk + base + m] * Tf[(base + m) * kChunk + gc];
          Tf[gi * kChunk + gc] = -sfs;
        }
      }
    }
  }
  __syncthreads();
  // Off-diagonal Schur merges: T(bi,bj) = -T(bi,bi) @ Σ_{bj≤k<bi} A(bi,k) @ T(k,bj)
  // Phased by distance so inputs are ready. Each merge done by 1 warp (scalar 16×16 MM).
  auto schurMerge = [&](int bi, int bj, int warp_id) {
    if (wf != warp_id) return;
    // S[16,16] = Σ_{k=bj}^{bi-1} A(bi,k) @ T(k,bj)
    float S[kBlk * kBlk] = {0.0f};
    for (int k = bj; k < bi; ++k) {
      for (int idx = tid % 32; idx < kBlk * kBlk; idx += 32) {
        int i = idx / kBlk, j = idx % kBlk;
        float s = 0.0f;
        for (int m = 0; m < kBlk; ++m)
          s += Am[(bi * kBlk + i) * kChunk + (k * kBlk + m)] *
               Tf[(k * kBlk + m) * kChunk + (bj * kBlk + j)];
        S[i * kBlk + j] += s;
      }
    }
    // T(bi,bj) = -T(bi,bi) @ S
    for (int idx = tid % 32; idx < kBlk * kBlk; idx += 32) {
      int i = idx / kBlk, j = idx % kBlk;
      float s = 0.0f;
      for (int m = 0; m < kBlk; ++m)
        s += Tf[(bi * kBlk + i) * kChunk + (bi * kBlk + m)] * S[m * kBlk + j];
      Tf[(bi * kBlk + i) * kChunk + (bj * kBlk + j)] = -s;
    }
  };
  // Distance 1: (1,0), (2,1), (3,2)
  schurMerge(1, 0, 0);
  schurMerge(2, 1, 1);
  schurMerge(3, 2, 2);
  __syncthreads();
  // Distance 2: (2,0), (3,1)
  schurMerge(2, 0, 0);
  schurMerge(3, 1, 1);
  __syncthreads();
  // Distance 3: (3,0)
  schurMerge(3, 0, 0);
  __syncthreads();

  // 5. Convert T to half: Tb (8KB, reuses Am space in R1)
  for (int64_t e = tid; e < kChunk * kChunk; e += blockDim.x)
    Tb[e] = __float2half(Tf[e]);
  __syncthreads();  // Tf (R0) free → Bop aliases it

  // 6. Apply: u = T @ (beta*v), w = T @ (beta*eg*k) via WMMA
  for (int p = 0; p < 2; ++p) {
    float* dst = (p == 0) ? u : w;
    const int64_t dcol = (p == 0) ? dv : dk;
    const int64_t stride = (p == 0) ? hv_n * dv : hv_n * dk;
    const int64_t hoff = (p == 0) ? hv * dv : hv * dk;
    const int64_t hk_off = (p == 0) ? 0 : hk * dk;  // w uses hk indexing

    // Build Bop [kChunk, dcol] as half in R0
    __half* Bop = R0_dyn;
    for (int64_t e = tid; e < kChunk * dcol; e += blockDim.x) {
      int64_t i = e / dcol, c = e % dcol;
      float val = 0.0f;
      if (i < len) {
        if (p == 0)
          val = bs[i] * Ld(v, (tok0 + i) * hv_n * dv + hv * dv + c);
        else
          val = bs[i] * eg[i] * Ld(k, (tok0 + i) * hk_n * dk + hk_off + c);
      }
      Bop[e] = __float2half(val);
    }
    __syncthreads();

    // WMMA apply: Osh[kChunk, bw] = Tb @ Bop[:, cb:cb+bw]
    for (int64_t cb = 0; cb < dcol; cb += kOBlk) {
      const int bw = static_cast<int>(dcol - cb < kOBlk ? dcol - cb : kOBlk);
      const int ntiles = (kChunk / WM) * (bw / WM);
      for (int t = wf; t < ntiles; t += n_wfs) {
        const int it = t % (kChunk / WM), ct = t / (kChunk / WM);
        const int i0 = it * WM, c0 = ct * WM;
        fragment<accumulator, WM, WN, WK, float> acc;
        fill_fragment(acc, 0.0f);
        for (int64_t jj = 0; jj < kChunk; jj += WK) {
          fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, __half, row_major> b_frag;
          load_matrix_sync(a_frag, Tb + i0 * kChunk + jj, (uint32_t)kChunk);
          load_matrix_sync(b_frag, Bop + jj * dcol + cb + c0, (uint32_t)dcol);
          mma_sync(acc, a_frag, b_frag, acc);
        }
        store_matrix_sync(Osh + i0 * bw + c0, acc, (uint32_t)bw, mem_row_major);
      }
      __syncthreads();
      // Write to global
      for (int64_t e = tid; e < kChunk * bw; e += blockDim.x) {
        int64_t i = e / bw, c = e % bw;
        if (i < len)
          dst[(tok0 + i) * stride + hoff + cb + c] = Osh[e];
      }
      __syncthreads();
    }
  }
}

// Step C — cross-chunk state recurrence. SEQUENTIAL over chunks.
// Uses WMMA for BOTH v_new (u - w @ hsh^T) and state_update (hsh += vd^T @ k).
// kBVD=32 vi-tiling to fit both GEMMs in 64KB LDS.
// Shared: hsh(16KB dyn) + k_cache(16KB) + w_buf(16KB reused) + hsh_T(8KB) + decay(256B) = 56.3KB
template <typename Tin>
__global__ void GdnChunkDeltaHKR(float* state, float* hstate, float* v_new,
                                 const Tin* k, const float* u, const float* w,
                                 const float* gcum, const int32_t* qsl,
                                 const int32_t* boh, int64_t hk_n, int64_t dk,
                                 int64_t hv_n, int64_t dv) {
  using namespace rocwmma;
  constexpr int kBVD = 32;
  const int64_t n = blockIdx.x, hv = blockIdx.y;
  const int64_t vbase = static_cast<int64_t>(blockIdx.z) * kBVD;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = threadIdx.x;
  const int wf = tid / 32;
  const int n_wfs = blockDim.x / 32;

  // Dynamic shared: hsh [kBVD, dk] float = 16KB for kBVD=32, dk=128
  extern __shared__ float hsh_dyn[];
  float* hsh = hsh_dyn;
  __shared__ float decay[kChunk];
  // w_buf: reused — phase 1 stores w_cache[kChunk,dk] half (16KB),
  //         phase 2 stores vd_T[kBVD,kChunk] half (4KB within 16KB)
  __shared__ __half w_buf[kChunk * kChunkMaxDim];
  // k_cache [kChunk, dk] half = 16KB
  __shared__ __half k_cache[kChunk * kChunkMaxDim];
  // hsh_T [dk, kBVD] half = 8KB — hsh transposed for v_new WMMA B matrix
  __shared__ __half hsh_T[kChunkMaxDim * kBVD];

  float* s_head = state + (n * hv_n + hv) * sd;
  // Load initial state for this vi_tile
  for (int64_t e = tid; e < kBVD * dk; e += blockDim.x) {
    int64_t vi = e / dk, ki = e % dk;
    int64_t gvi = vbase + vi;
    hsh[e] = (gvi < dv) ? s_head[gvi * dk + ki] : 0.0f;
  }
  __syncthreads();

  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * kChunk;
    const int64_t rem = seqlen - it * kChunk;
    const int64_t len = rem < kChunk ? rem : kChunk;
    float* hstart = hstate + (gc * hv_n + hv) * sd;
    // Snapshot start state for step D
    for (int64_t e = tid; e < kBVD * dk; e += blockDim.x) {
      int64_t vi = e / dk, ki = e % dk;
      int64_t gvi = vbase + vi;
      if (gvi < dv) hstart[gvi * dk + ki] = hsh[e];
    }
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += blockDim.x)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);
    __syncthreads();

    // --- Phase 1: v_new = u - w @ hsh^T  (WMMA) ---
    // Stage w_buf [len, dk] as half (w_cache)
    for (int64_t idx = tid; idx < len * dk; idx += blockDim.x) {
      int64_t i = idx / dk, ki = idx % dk;
      w_buf[i * dk + ki] = __float2half(w[(tok0 + i) * hv_n * dk + hv * dk + ki]);
    }
    // Stage hsh_T [dk, kBVD] as half col_major: hsh_T[ki, vi] = hsh[vi, ki]
    for (int64_t idx = tid; idx < dk * kBVD; idx += blockDim.x) {
      int64_t ki = idx / kBVD, vi = idx % kBVD;
      hsh_T[ki * kBVD + vi] = __float2half(hsh[vi * dk + ki]);
    }
    __syncthreads();
    // WMMA: v_new[len, kBVD] = w[len, dk] @ hsh_T[dk, kBVD]
    // A = w_buf [len, dk] row_major, B = hsh_T [dk, kBVD] col_major
    // Actually hsh_T is stored as hsh_T[ki * kBVD + vi] = row_major [dk, kBVD]
    // For col_major B: B[ki, vi] = ptr[ki + vi * ldm]. We want B[ki, vi] = hsh_T[ki * kBVD + vi].
    // With row_major: B[ki, vi] = ptr[ki * ldm + vi] = hsh_T[ki * kBVD + vi] with ldm=kBVD. ✓
    {
      const int M_t = (len + WM - 1) / WM;  // 4 (len=64)
      const int N_t = kBVD / WN;             // 2
      const int K_t = dk / WK;               // 8
      for (int t = wf; t < M_t * N_t; t += n_wfs) {
        const int mt = t / N_t, nt2 = t % N_t;
        fragment<accumulator, WM, WN, WK, float> c_frag;
        fill_fragment(c_frag, 0.0f);
        for (int kt = 0; kt < K_t; ++kt) {
          fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, __half, row_major> b_frag;
          load_matrix_sync(a_frag, w_buf + mt * WM * dk + kt * WK, (uint32_t)dk);
          load_matrix_sync(b_frag, hsh_T + kt * WK * kBVD + nt2 * WN, (uint32_t)kBVD);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        // Store w@hsh^T to hsh_T (reused as float, 8KB fits [len, kBVD] subset)
        // Actually store to a temp area in w_buf (which is done with w)
        // We need [len, kBVD] float = 64*32*4 = 8KB. hsh_T is 8KB. Reuse it.
        float* vnew_tmp = reinterpret_cast<float*>(hsh_T);
        store_matrix_sync(vnew_tmp + mt * WM * kBVD + nt2 * WN, c_frag,
                          (uint32_t)kBVD, mem_row_major);
      }
    }
    __syncthreads();
    // v_new[i,vi] = u[i,vi] - w@hsh^T[i,vi], write to global
    {
      float* vnew_tmp = reinterpret_cast<float*>(hsh_T);
      for (int64_t idx = tid; idx < len * kBVD; idx += blockDim.x) {
        int64_t i = idx / kBVD, vi = idx % kBVD;
        int64_t gvi = vbase + vi;
        if (gvi < dv)
          v_new[(tok0 + i) * hv_n * dv + hv * dv + gvi] =
              u[(tok0 + i) * hv_n * dv + hv * dv + gvi] - vnew_tmp[idx];
      }
    }
    __syncthreads();

    // --- Phase 2: hsh = hsh*eglast + vd^T @ k  (WMMA) ---
    // Stage k_cache [len, dk] as half
    for (int64_t idx = tid; idx < len * dk; idx += blockDim.x) {
      int64_t i = idx / dk, ki = idx % dk;
      k_cache[i * dk + ki] = __float2half(Ld(k, (tok0 + i) * hk_n * dk + hk * dk + ki));
    }
    // Stage w_buf (reused as vd_T [kBVD, len] col_major): vd_T[vi, i] = v_new[i,vi]*decay[i]
    for (int64_t idx = tid; idx < len * kBVD; idx += blockDim.x) {
      int64_t i = idx / kBVD, vi = idx % kBVD;
      int64_t gvi = vbase + vi;
      float val = (gvi < dv)
        ? v_new[(tok0 + i) * hv_n * dv + hv * dv + gvi] * decay[i]
        : 0.0f;
      w_buf[vi * kChunk + i] = __float2half(val);  // col_major [kBVD, len]
    }
    __syncthreads();
    // WMMA: hsh[kBVD, dk] += vd_T[kBVD, len] @ k[len, dk]
    // A = w_buf [kBVD, len] col_major (stored as w_buf[vi*kChunk + i])
    // B = k_cache [len, dk] row_major
    // C = hsh [kBVD, dk] row_major
    {
      const int M_t = kBVD / WM;    // 2
      const int N_t = dk / WN;      // 8
      const int K_t = (len + WK - 1) / WK;  // 4
      for (int t = wf; t < M_t * N_t; t += n_wfs) {
        const int mt = t / N_t, nt2 = t % N_t;
        fragment<accumulator, WM, WN, WK, float> c_frag;
        load_matrix_sync(c_frag, hsh + mt * WM * dk + nt2 * WN, (uint32_t)dk, mem_row_major);
        for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] *= eglast;
        for (int kt = 0; kt < K_t; ++kt) {
          fragment<matrix_a, WM, WN, WK, __half, col_major> a_frag;
          fragment<matrix_b, WM, WN, WK, __half, row_major> b_frag;
          load_matrix_sync(a_frag, w_buf + mt * WM + kt * WK * kChunk, (uint32_t)kChunk);
          load_matrix_sync(b_frag, k_cache + kt * WK * dk + nt2 * WN, (uint32_t)dk);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(hsh + mt * WM * dk + nt2 * WN, c_frag, (uint32_t)dk, mem_row_major);
      }
    }
    __syncthreads();
  }
  // Write final state
  for (int64_t e = tid; e < kBVD * dk; e += blockDim.x) {
    int64_t vi = e / dk, ki = e % dk;
    int64_t gvi = vbase + vi;
    if (gvi < dv) s_head[gvi * dk + ki] = hsh[e];
  }
}

// Step D — chunk output: cross-chunk + intra-chunk causal term.
// WMMA version: uses tensor cores for QK^T, cross-chunk, and intra-chunk GEMMs.
// Shared memory (48.5KB): Qs[64,128] half (16KB dyn) + A[64,64] float (16KB)
//   + buf[16KB] (Ks→A_half→Hstart/V_new tiles) + gs/eg (0.5KB)
template <typename Tin, typename Tout>
__global__ void GdnChunkOKR(Tout* out, const Tin* q, const Tin* k, const float* v_new,
                            const float* hstate, const float* gcum,
                            const int32_t* tok0a, const int32_t* lena,
                            int64_t hk_n, int64_t dk, int64_t hv_n, int64_t dv,
                            float scale) {
  using namespace rocwmma;
  constexpr int WM = 16, WN = 16, WK = 16;
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int64_t sd = dv * dk;
  const float* hstart = hstate + (gc * hv_n + hv) * sd;
  const int tid = threadIdx.x;
  const int wf = tid / 32, n_wfs = blockDim.x / 32;

  // Dynamic shared: Qs [kChunk, dk] half = 16KB
  extern __shared__ __half Qs_dyn[];
  __half* Qs = Qs_dyn;
  // Static shared: A [kChunk, kChunk] float = 16KB (QK^T result, then outc)
  __shared__ float A[kChunk * kChunk];
  // buf: reused — Ks [kChunk, dk] half (16KB), then A_half [kChunk, kChunk] half (8KB),
  //       then Hstart tile [16, dk] half (4KB) + V_new tile [kChunk, 16] half (2KB)
  __shared__ __half buf[kChunk * kChunkMaxDim];
  __shared__ float gs[kChunk], eg[kChunk];

  // 1. Load Qs [len, dk] as half
  for (int64_t idx = tid; idx < len * dk; idx += blockDim.x) {
    int64_t i = idx / dk, d = idx % dk;
    Qs[idx] = __float2half(__bfloat162float(
        Ld(q, (tok0 + i) * hk_n * dk + hk * dk + d)));
  }
  // Load Ks [len, dk] as half into buf
  for (int64_t idx = tid; idx < len * dk; idx += blockDim.x) {
    int64_t i = idx / dk, d = idx % dk;
    buf[idx] = __float2half(__bfloat162float(
        Ld(k, (tok0 + i) * hk_n * dk + hk * dk + d)));
  }
  // Load gs, eg
  for (int64_t i = tid; i < len; i += blockDim.x) {
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // 2. QK^T via WMMA: A[i,j] = sum_d Qs[i,d] * Ks[j,d]
  // A_mat = Qs [kChunk, dk] row_major, B_mat = Ks^T [dk, kChunk] col_major
  {
    const int M_t = (kChunk + WM - 1) / WM;  // 4
    const int N_t = (kChunk + WN - 1) / WN;  // 4
    const int K_t = dk / WK;                  // 8
    for (int t = wf; t < M_t * N_t; t += n_wfs) {
      const int mt = t / N_t, nt2 = t % N_t;
      fragment<accumulator, WM, WN, WK, float> acc;
      fill_fragment(acc, 0.0f);
      for (int kt = 0; kt < K_t; ++kt) {
        fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
        fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag;
        load_matrix_sync(a_frag, Qs + mt * WM * dk + kt * WK, (uint32_t)dk);
        load_matrix_sync(b_frag, buf + nt2 * WN * WK + kt * WK, (uint32_t)WK);
        mma_sync(acc, a_frag, b_frag, acc);
      }
      store_matrix_sync(A + mt * WM * kChunk + nt2 * WN, acc,
                        (uint32_t)kChunk, mem_row_major);
    }
  }
  __syncthreads();

  // 3. A[i,j] = exp(gs[i]-gs[j]) * A[i,j] for j<=i, else 0 (causal)
  for (int64_t idx = tid; idx < kChunk * kChunk; idx += blockDim.x) {
    int64_t i = idx / kChunk, j = idx % kChunk;
    float val = 0.0f;
    if (i < len && j < len && j <= i)
      val = expf(gs[i] - gs[j]) * A[idx];
    A[idx] = val;
  }
  __syncthreads();

  // 4. Convert A to half into buf (A_half, 8KB)
  for (int64_t idx = tid; idx < kChunk * kChunk; idx += blockDim.x)
    buf[idx] = __float2half(A[idx]);
  __syncthreads();

  // 5. For each dv tile of 16: cross-chunk + intra-chunk
  const int n_dv_tiles = (dv + WN - 1) / WN;  // 8
  for (int vt = 0; vt < n_dv_tiles; ++vt) {
    const int vi0 = vt * WN;
    // Load Hstart tile [WN, dk] as half into buf+8KB (reuse after A_half)
    __half* hstart_tile = buf + kChunk * kChunk;  // 8KB offset, 4KB size
    for (int64_t idx = tid; idx < WN * dk; idx += blockDim.x) {
      int64_t vi = idx / dk, d = idx % dk;
      int64_t gvi = vi0 + vi;
      hstart_tile[idx] = (gvi < dv)
        ? __float2half(hstart[gvi * dk + d])
        : __float2half(0.0f);
    }
    __syncthreads();

    // Cross-chunk WMMA: outc[kChunk, WN] = Qs @ Hstart_tile^T
    // Store into A with ldm=kChunk (reuse first kChunk*WN floats of A)
    {
      const int M_t = (kChunk + WM - 1) / WM;  // 4
      const int N_t = 1;                         // WN=16, one tile
      const int K_t = dk / WK;                   // 8
      for (int t = wf; t < M_t; t += n_wfs) {
        fragment<accumulator, WM, WN, WK, float> acc;
        fill_fragment(acc, 0.0f);
        for (int kt = 0; kt < K_t; ++kt) {
          fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag;
          load_matrix_sync(a_frag, Qs + t * WM * dk + kt * WK, (uint32_t)dk);
          load_matrix_sync(b_frag, hstart_tile + kt * WK, (uint32_t)WN);
          mma_sync(acc, a_frag, b_frag, acc);
        }
        store_matrix_sync(A + t * WM * kChunk, acc,
                          (uint32_t)kChunk, mem_row_major);
      }
    }
    __syncthreads();

    // Scale cross by eg[i]
    for (int64_t i = tid; i < kChunk; i += blockDim.x) {
      for (int vi = 0; vi < WN; ++vi)
        A[i * kChunk + vi] *= (i < len) ? eg[i] : 0.0f;
    }
    __syncthreads();

    // Load V_new tile [len, WN] as half into buf+8KB+4KB (after hstart_tile)
    __half* vnew_tile = hstart_tile + WN * dk;  // 12KB offset, 2KB size
    for (int64_t idx = tid; idx < len * WN; idx += blockDim.x) {
      int64_t j = idx / WN, vi = idx % WN;
      int64_t gvi = vi0 + vi;
      vnew_tile[idx] = (gvi < dv)
        ? __float2half(v_new[(tok0 + j) * hv_n * dv + hv * dv + gvi])
        : __float2half(0.0f);
    }
    __syncthreads();

    // Intra-chunk WMMA: outc += A_half @ V_new_tile
    // A_half [kChunk, kChunk] row_major, V_new_tile [kChunk, WN] row_major
    // Accumulate into A (ldm=kChunk)
    {
      const int M_t = (kChunk + WM - 1) / WM;  // 4
      const int N_t = 1;                         // WN=16
      const int K_t = (kChunk + WK - 1) / WK;   // 4
      for (int t = wf; t < M_t; t += n_wfs) {
        fragment<accumulator, WM, WN, WK, float> acc;
        load_matrix_sync(acc, A + t * WM * kChunk, (uint32_t)kChunk, mem_row_major);
        for (int kt = 0; kt < K_t; ++kt) {
          fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, __half, row_major> b_frag;
          load_matrix_sync(a_frag, buf + t * WM * kChunk + kt * WK, (uint32_t)kChunk);
          load_matrix_sync(b_frag, vnew_tile + kt * WK * WN, (uint32_t)WN);
          mma_sync(acc, a_frag, b_frag, acc);
        }
        store_matrix_sync(A + t * WM * kChunk, acc,
                          (uint32_t)kChunk, mem_row_major);
      }
    }
    __syncthreads();

    // Write output: out[i, vi] = scale * A[i, vi]
    for (int64_t idx = tid; idx < len * WN; idx += blockDim.x) {
      int64_t i = idx / WN, vi = idx % WN;
      int64_t gvi = vi0 + vi;
      if (gvi < dv)
        St(out, (tok0 + i) * hv_n * dv + hv * dv + gvi,
           static_cast<Tout>(scale * A[i * kChunk + vi]));
    }
    __syncthreads();
  }
}

// Build chunk metadata from query_start_loc on device.
__global__ void GdnBuildChunkMetaKR(int32_t* tok0, int32_t* lenv, int32_t* boh,
                                    const int32_t* qsl, int64_t n_seq) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int32_t nt = 0;
  for (int64_t n = 0; n < n_seq; ++n) {
    boh[n] = nt;
    const int32_t begin = qsl[n];
    const int32_t len = qsl[n + 1] - begin;
    const int32_t ntn = (len + kChunk - 1) / kChunk;
    for (int32_t it = 0; it < ntn; ++it) {
      const int32_t rem = len - it * kChunk;
      tok0[nt] = begin + it * kChunk;
      lenv[nt] = rem < kChunk ? rem : kChunk;
      ++nt;
    }
  }
}

// Launcher for the chunked parallel prefill.
template <typename Tin, typename Tout>
void LaunchChunkedPrefillRocm(hipStream_t s, Tensor& out, const Tensor& q_in,
                              const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, Tensor& state,
                              const int32_t* qsl_dev, const int32_t* qsl_host,
                              int64_t n_seq, int64_t nt_tot,
                              const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const int64_t t_tot = qsl_host ? qsl_host[n_seq] : 0;

  float* gcum = nullptr;
  float* u = nullptr;
  float* w = nullptr;
  float* v_new = nullptr;
  float* hstate = nullptr;
  int32_t *d_tok0 = nullptr, *d_len = nullptr, *d_boh = nullptr;

  Check(hipMallocAsync(&gcum, static_cast<size_t>(t_tot * hv_n) * sizeof(float), s),
        "gdn chunked gcum alloc");
  Check(hipMallocAsync(&u, static_cast<size_t>(t_tot * hv_n * dv) * sizeof(float), s),
        "gdn chunked u alloc");
  Check(hipMallocAsync(&w, static_cast<size_t>(t_tot * hv_n * dk) * sizeof(float), s),
        "gdn chunked w alloc");
  Check(hipMallocAsync(&v_new, static_cast<size_t>(t_tot * hv_n * dv) * sizeof(float), s),
        "gdn chunked v_new alloc");
  Check(hipMallocAsync(&hstate, static_cast<size_t>(nt_tot * hv_n * dv * dk) * sizeof(float), s),
        "gdn chunked hstate alloc");
  Check(hipMallocAsync(&d_tok0, static_cast<size_t>(nt_tot) * sizeof(int32_t), s),
        "gdn chunked tok0 alloc");
  Check(hipMallocAsync(&d_len, static_cast<size_t>(nt_tot) * sizeof(int32_t), s),
        "gdn chunked len alloc");
  Check(hipMallocAsync(&d_boh, static_cast<size_t>(n_seq) * sizeof(int32_t), s),
        "gdn chunked boh alloc");

  GdnBuildChunkMetaKR<<<1, 1, 0, s>>>(d_tok0, d_len, d_boh, qsl_dev, n_seq);
  Check(hipGetLastError(), "gdn chunked build-meta launch");

  const dim3 grid_chunk(static_cast<unsigned>(nt_tot), static_cast<unsigned>(hv_n));
  constexpr int kBVD = 32;
  const int64_t nv_d = (dv + kBVD - 1) / kBVD;
  const dim3 grid_seq(static_cast<unsigned>(n_seq), static_cast<unsigned>(hv_n),
                      static_cast<unsigned>(nv_d));
  const size_t delta_shmem = static_cast<size_t>(kBVD * dk) * sizeof(float);

  // Step A: cumsum
  GdnChunkCumsumKR<<<grid_chunk, 1, 0, s>>>(gcum, g.Ptr<float>(), d_tok0, d_len, hv_n);
  Check(hipGetLastError(), "gdn chunked cumsum launch");

  // Step B: WY decomposition (WMMA KK^T — needs dynamic shared for Ks [kChunk, dk] half)
  const size_t wu_shmem = static_cast<size_t>(kChunk * dk) * sizeof(__half);
  GdnChunkWUKR<Tin><<<grid_chunk, 128, wu_shmem, s>>>(
      u, w, k.Ptr<Tin>(), v.Ptr<Tin>(), beta.Ptr<float>(), gcum, d_tok0, d_len,
      hk_n, dk, hv_n, dv);
  Check(hipGetLastError(), "gdn chunked wu launch");

  // Step C: cross-chunk state recurrence (WMMA state update)
  GdnChunkDeltaHKR<Tin><<<grid_seq, kBlock, delta_shmem, s>>>(
      state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl_dev, d_boh,
      hk_n, dk, hv_n, dv);
  Check(hipGetLastError(), "gdn chunked delta_h launch");

  // Step D: output (WMMA — needs dynamic shared for Qs [kChunk, dk] half)
  const size_t o_shmem = static_cast<size_t>(kChunk * dk) * sizeof(__half);
  GdnChunkOKR<Tin, Tout><<<grid_chunk, kBlock, o_shmem, s>>>(
      out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v_new, hstate, gcum, d_tok0,
      d_len, hk_n, dk, hv_n, dv, args.scale);
  Check(hipGetLastError(), "gdn chunked o launch");

  Check(hipFreeAsync(gcum, s), "gdn chunked gcum free");
  Check(hipFreeAsync(u, s), "gdn chunked u free");
  Check(hipFreeAsync(w, s), "gdn chunked w free");
  Check(hipFreeAsync(v_new, s), "gdn chunked v_new free");
  Check(hipFreeAsync(hstate, s), "gdn chunked hstate free");
  Check(hipFreeAsync(d_tok0, s), "gdn chunked tok0 free");
  Check(hipFreeAsync(d_len, s), "gdn chunked len free");
  Check(hipFreeAsync(d_boh, s), "gdn chunked boh free");
}

// Chunked prefill entry point.
template <typename Tin, typename Tout>
void GdnPrefillChunkedRocm(hipStream_t s, Tensor& out, const Tensor& q_in,
                           const Tensor& k, const Tensor& v, const Tensor& g,
                           const Tensor& beta, Tensor& state, const Tensor& qsl,
                           const GdnArgs& args) {
  const int64_t n_seq = state.shape[0];
  const int32_t* qsl_dev = qsl.Ptr<int32_t>();
  const int32_t* qsl_host = args.query_start_loc_host;

  int32_t nt_tot = 0;
  if (qsl_host != nullptr) {
    for (int64_t n = 0; n < n_seq; ++n) {
      const int32_t len = qsl_host[n + 1] - qsl_host[n];
      nt_tot += (len + kChunk - 1) / kChunk;
    }
  } else {
    std::vector<int32_t> qslh(static_cast<size_t>(n_seq) + 1);
    Check(hipMemcpyAsync(qslh.data(), qsl_dev,
                         static_cast<size_t>(n_seq + 1) * sizeof(int32_t),
                         hipMemcpyDeviceToHost, s),
          "gdn chunked qsl download");
    Check(hipStreamSynchronize(s), "gdn chunked qsl sync");
    for (int64_t n = 0; n < n_seq; ++n) {
      const int32_t len = qslh[static_cast<size_t>(n + 1)] - qslh[static_cast<size_t>(n)];
      nt_tot += (len + kChunk - 1) / kChunk;
    }
  }
  if (nt_tot == 0) return;

  LaunchChunkedPrefillRocm<Tin, Tout>(s, out, q_in, k, v, g, beta, state, qsl_dev,
                                      qsl_host, n_seq, nt_tot, args);
}

}  // namespace

void GdnPrefillKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta,
                          Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  const int64_t n = state.shape[0];
  if (n == 0) return;
  hipStream_t s = AsStream(q);
  // Chunked parallel prefill (default ON; VT_GDN_CHUNKED=0 falls back to scan)
  if (ChunkedPrefillEnabled() && qsl.shape[0] > 0) {
    if (q_in.dtype == DType::kBF16) {
      if (out.dtype == DType::kBF16)
        GdnPrefillChunkedRocm<__hip_bfloat16, __hip_bfloat16>(s, out, q_in, k, v, g, beta,
                                                               state, qsl, args);
      else
        GdnPrefillChunkedRocm<__hip_bfloat16, float>(s, out, q_in, k, v, g, beta, state,
                                                      qsl, args);
    } else {
      if (out.dtype == DType::kBF16)
        GdnPrefillChunkedRocm<float, __hip_bfloat16>(s, out, q_in, k, v, g, beta, state,
                                                      qsl, args);
      else
        GdnPrefillChunkedRocm<float, float>(s, out, q_in, k, v, g, beta, state, qsl, args);
    }
    return;
  }
  if (q_in.dtype == DType::kBF16) {
    LaunchGdnScanIn<__hip_bfloat16>(s, out, q_in, k, v, g, beta, state,
                                    qsl.Ptr<int32_t>(), nullptr, n, args);
  } else {
    LaunchGdnScanIn<float>(s, out, q_in, k, v, g, beta, state, qsl.Ptr<int32_t>(),
                           nullptr, n, args);
  }
}

void GdnDecodeKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         Tensor& state, const Tensor* state_idx, const GdnArgs& args) {
  const int64_t n = q_in.shape[0];
  if (n == 0) return;
  const int32_t* si = state_idx != nullptr ? state_idx->Ptr<int32_t>() : nullptr;
  hipStream_t s = AsStream(q);
  if (q_in.dtype == DType::kBF16) {
    LaunchGdnScanIn<__hip_bfloat16>(s, out, q_in, k, v, g, beta, state, nullptr, si, n,
                                    args);
  } else {
    LaunchGdnScanIn<float>(s, out, q_in, k, v, g, beta, state, nullptr, si, n, args);
  }
}

}  // namespace vt::rocm
