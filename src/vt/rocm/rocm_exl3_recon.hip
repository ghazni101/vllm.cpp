// EXL3 reconstruct + GEMM — the ROCm arm of OpId::kExl3ReconstructGemm, the
// op dense_attn_block.h::Exl3MatmulD dispatches at M > 144 (upstream's
// AUTO_RECONSTRUCT_THRESHOLD, exl3.py:10,135). Spec
// .agents/specs/rocm-exl3-gemv.md (Slice B), row BACKEND-ROCM.
//
// THE SHAPE IS THE CUDA ARM'S UNFUSED PATH, NOT ITS FUSED ONE.
// cuda_exl3.cu's Exl3ReconstructGemmKernelCuda picks `reconstruct_had_slice`
// — one kernel that decodes the trellis AND applies both Hadamard transforms
// to the reconstructed weight — when M >= 1024, and the plain
// `reconstruct_slice` + cuBLAS + separate had_r_128 calls otherwise. The fused
// kernel's geometry (had_fh_r_128_xxx_dual, 128x64 LDS tiles, xor-shuffled
// fragments) has no small port to LDS; the unfused chain is already fully
// served by this backend's pieces:
//
//   1. had_r_128(A, suh)             via Exl3LaunchHadRocm (rocm_exl3.hip —
//                                    the same kernel the fused chain runs)
//   2. W = reconstruct(trellis)      ReconstructDecodeK below, ONE thread per
//                                    output element over the inverse fragment
//                                    map (rocm_exl3_decode.h), so writes are
//                                    coalesced and every codeword is decoded
//                                    exactly once
//   3. C = A_had @ W                 hipblasGemmEx, (f16,f16)->f16|f32
//   4. had_r_128(C, svh), in place   Exl3LaunchHadRocm again
//
// M >= 1024 therefore takes the UNFUSED path on this backend: correct at every
// M the op serves, where the CUDA arm's M>=1024 fused kernel is owed
// separately. The N > 32768 slice loop mirrors upstream's own structure.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_exl3_decode.h"
#include "vt/rocm/rocm_getblas_product.h"

namespace vt::rocm {
namespace {

inline void CheckR(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm exl3 recon: ") + what + ": " +
                             hipGetErrorString(err));
}
inline void CheckBlas(hipblasStatus_t st, const char* what) {
  if (st != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("vt rocm exl3 recon: ") + what +
                             " failed with status " + std::to_string(static_cast<int>(st)));
}
inline hipStream_t AsStreamR(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

constexpr float kInvSqrt128R = 0.088388347648f;
// Upstream's slice bound (reconstruct.cu / cuda_exl3.cu:2774): the fp16
// scratch never holds more than this many reconstructed columns at once.
constexpr int64_t kReconSliceN = 32768;

// ─── step 2: trellis -> dense fp16 weight ────────────────────────────────────
//
// One thread per (row, col) output element. The fragment layout means codeword
// t's decoded weight lands at (r, c) = Exl3TileRowMajorIndex(t), so the thread
// OWNING (r, c) inverts the map (Exl3TileRowMajorIndexInv, rocm_exl3_decode.h)
// and decodes exactly its own codeword — no LDS, no barriers, every codeword
// decoded once, and the 16-wide x-tile keeps writes coalesced. `n_slice` is
// w's row stride and `n_packed` the packed trellis's own n: the two differ
// under the 32768-column slicing, and conflating them is the reconstruct-class
// defect this family files.
__global__ void ReconstructDecodeSliceK(half* __restrict__ w, const uint16_t* __restrict__ trellis,
                                        int64_t k, int64_t n_slice, int64_t n_packed,
                                        int bits, int codebook, int64_t packed_n_off) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= k || c >= n_slice) return;
  const int64_t tiles_n = n_packed / 16;
  const int64_t tile_words = 16 * static_cast<int64_t>(bits);
  const uint16_t* tp =
      trellis + (((r >> 4) * tiles_n) + packed_n_off + (c >> 4)) * tile_words;
  const int t = Exl3TileRowMajorIndexInv(static_cast<int>(r & 15), static_cast<int>(c & 15));
  w[r * n_slice + c] =
      __float2half_rn(Exl3DecodeCodeword(Exl3TileCodeword(tp, bits, t), codebook));
}

// ─── step 3: f16 @ f16 -> f16|f32 GEMM through hipblasGemmEx ─────────────────
//
// Row-major C[M,N] = A[M,K] @ W[K,N] is BLAS's column-major
// C_cm(N,M) = W_cm(N,K) @ A_cm(K,M): both inputs are already column-major in
// memory (a row-major [X,Y] buffer IS a column-major [Y,X]), so OP_N/OP_N.
// `ldc` honours a column slice's parent stride exactly as the CUDA arm's
// CublasLtHgemm honours c.stride[0] (cuda_matmul.cu:1336-1382).
void HgemmRowMajor(int device, hipStream_t s, void* C, const void* A, const void* W,
                   int64_t m, int64_t n, int64_t k, int64_t ldc, bool c_f32) {
  hipblasHandle_t h = ProductGetBlasHandle(device, s);
  if (h == nullptr) throw std::runtime_error("vt rocm exl3 recon: no hipBLAS handle");
  const float alpha = 1.0f, beta = 0.0f;
  const hipDataType c_type = c_f32 ? HIP_R_32F : HIP_R_16F;
  CheckBlas(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(n),
                          static_cast<int>(m), static_cast<int>(k), &alpha, W, HIP_R_16F,
                          static_cast<int>(n), A, HIP_R_16F, static_cast<int>(k), &beta,
                          C, c_type, static_cast<int>(ldc), HIPBLAS_COMPUTE_32F,
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx hgemm");
}

// ─── the persistent reconstruct scratch (QUANT-EXL3 W7's discipline) ─────────
//
// Grow-only, one buffer per stream, hipMallocAsync, never freed — the same
// contract rocm_exl3.hip's EnsureRawScratch carries and for the same reason:
// hipMalloc/hipFree/hipStreamSynchronize per call are illegal under hipGraph
// capture. Growth while the stream is capturing is refused BY NAME, mirroring
// the CUDA arm's Exl3PersistentReconScratch (cuda_exl3.cu:2824).
struct ReconScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};

ReconScratch& ReconScratchFor(hipStream_t s) {
  static std::mutex mu;
  static std::unordered_map<hipStream_t, ReconScratch> pools;
  std::lock_guard<std::mutex> lk(mu);
  return pools[s];
}

bool StreamCapturing(hipStream_t s) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(s, &status) != hipSuccess) return false;
  return status == hipStreamCaptureStatusActive;
}

half* EnsureReconScratch(size_t bytes, hipStream_t s) {
  ReconScratch& sc = ReconScratchFor(s);
  if (bytes > sc.bytes) {
    if (StreamCapturing(s)) {
      throw std::runtime_error(
          "vt rocm exl3 recon: persistent weight scratch must grow to " +
          std::to_string(bytes) +
          " bytes but the queue's stream is capturing; growth during capture would "
          "bake a pointer the pool then re-points. Run one eager step at this weight's "
          "shape first — the buffer's size depends on the weight, not on m.");
    }
    CheckR(hipMallocAsync(&sc.buf, bytes, s), "recon scratch grow");
    sc.bytes = bytes;
  }
  return static_cast<half*>(sc.buf);
}

}  // namespace

// The had_r_128 stages shared with rocm_exl3.hip.
void Exl3LaunchHadRocm(hipStream_t st, void* out, const void* in, const uint16_t* pre,
                       const uint16_t* post, float r_scale, int64_t rows, int64_t cols,
                       bool half_in, bool half_out);

void Exl3ReconstructGemmKernelRocm(Queue& q, Tensor& c, const Tensor& a,
                                   const Tensor& trellis, const Tensor& suh,
                                   const Tensor& svh, Tensor& a_had,
                                   Tensor& w_scratch, const Exl3GemmArgs& args) {
  EnsureQueueDevice(q);
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "vt rocm exl3 recon: bits must be in [1, 8]; got " + std::to_string(args.bits));
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "vt rocm exl3 recon: codebook " + std::to_string(args.codebook) +
               " is not implemented (0 == 3INST, 1 == MCG, 2 == mul1). Upstream defines "
               "no other value: `decode_3inst<cb>` (codebook.cuh:56-90) has arms for 0, "
               "1 and 2 and falls off the end for anything else.");
  // This backend's scalar decoder decodes EVERY width upstream defines — the
  // CUDA arm's {(3,0),(3,1),(6,0),(3,2),(4,2),(5,2),(6,2)} table is its own
  // instantiation limit, not the format's, so there is nothing to refuse here.
  const int64_t size_m = a.shape[0];
  const int64_t size_k = a.shape[1];
  const int64_t size_n = c.shape[1];
  if (size_m == 0 || size_k == 0 || size_n == 0) return;

  hipStream_t st = AsStreamR(q);

  // exl3.py:186-190. Input Hadamard — this arm is ALWAYS the unfused path (the
  // file header names why), so the transform always runs, into the caller's
  // a_had scratch.
  Exl3LaunchHadRocm(st, a_had.data, a.data, suh.Ptr<uint16_t>(), nullptr, kInvSqrt128R,
                    size_m, size_k, /*half_in=*/true, /*half_out=*/true);

  // QUANT-EXL3 W7: an EMPTY w_scratch selects this stream's persistent buffer,
  // resolved before any launch so a refused growth enqueues nothing.
  half* scratch_w = nullptr;
  int64_t slice_cols = size_n < kReconSliceN ? size_n : kReconSliceN;
  Tensor persistent_scratch;
  const Tensor* scratch = &w_scratch;
  if (w_scratch.data == nullptr) {
    persistent_scratch = vt::Tensor::Contiguous(
        EnsureReconScratch(static_cast<size_t>(size_k * slice_cols) * sizeof(half), st),
        DType::kF16, q.device, {size_k, slice_cols});
    scratch = &persistent_scratch;
  }
  scratch_w = scratch->Ptr<half>();

  const int64_t tiles_n_packed = trellis.shape[1];  // n/16
  const int64_t n_packed = tiles_n_packed * 16;

  // exl3.py:192-211. Reconstruct + GEMM, in 32768-column slices when the
  // weight is wider than the scratch bound.
  for (int64_t n_start = 0; n_start < size_n; n_start += kReconSliceN) {
    const int64_t n_slice =
        (size_n - n_start) < kReconSliceN ? (size_n - n_start) : kReconSliceN;

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((n_slice + 15) / 16),
              static_cast<unsigned>((size_k + 15) / 16));
    ReconstructDecodeSliceK<<<grid, block, 0, st>>>(
        scratch_w, trellis.Ptr<uint16_t>(), size_k, n_slice, n_packed, args.bits,
        args.codebook, n_start / 16);
    CheckR(hipGetLastError(), "reconstruct decode");

    // GEMM into the matching column slice of c. A slice views the parent's row
    // stride, so ldc is c.stride[0], not n_slice.
    void* c_slice = static_cast<char*>(c.data) +
                    static_cast<size_t>(n_start) * (c.dtype == DType::kF32 ? 4 : 2);
    HgemmRowMajor(q.device.index, st, c_slice, a_had.data, scratch_w, size_m, n_slice,
                  size_k, c.stride[0], c.dtype == DType::kF32);
  }

  // exl3.py:213-214. Output Hadamard, in place on c — safe because each
  // 128-block is independent (HadK documents the aliasing).
  Exl3LaunchHadRocm(st, c.data, c.data, nullptr, svh.Ptr<uint16_t>(), kInvSqrt128R,
                    size_m, size_n, /*half_in=*/c.dtype != DType::kF32,
                    /*half_out=*/c.dtype != DType::kF32);
}

}  // namespace vt::rocm
