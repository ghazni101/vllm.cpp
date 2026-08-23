// ROCm keep-quant GEMM gate (GFX1100-TG200). The campaign spec names
// `tests/vt/test_rocm_quant_dot.cpp` as the quant-path lever gate; until T4a
// that file DID NOT EXIST — the GPU-parity cases lived in
// tests/vt/test_cuda_quant_dot.cpp behind HasCuda() and so SKIPPED on this
// ROCm-only box (the exact T3a blind spot: op-level green while the engine
// produced garbage). This file is the fix: a focused gate for the ROCm
// kMatmulBTQuant provider (src/vt/rocm/rocm_grouped_gemm.hip) guarded on ROCM
// availability, never on CUDA.
//
// The T4a lever is an MMVQ-style decode GEMV arm behind VT_GEMV_MMVQ=1
// (default OFF; the default path must stay byte-unchanged). The new arm keeps
// the CPU integer core exactly and reproduces the CPU oracle's FLOAT
// association too (per-super-block positional sums[] chains + sequential dmin
// chain, cpu_quant_dot.cpp VecDot{Q4,Q5,Q6}_KQ8_K), so it gates at
// BIT-EXACTNESS vs vt::MatmulBTQuant on host tensors — STRICTLY tighter than
// the 1e-6 NMSE band the warp-reduction baseline can only claim (its
// __shfl_down tree reassociates the float sum).
//
// RED-first contract: before the dispatch arm exists VT_GEMV_MMVQ=1 is inert,
// the baseline kernel runs, and its reassociated float sum fails the
// bit-exact compare below.
//
// Skips cleanly (returns) when the build has HIP but the box has no AMD GPU,
// so the CPU CI leg stays green.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/rocm/rocm_runtime.h"
#include "vt/tensor.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace vt::rocm {
void MmvqQuantScratchForTesting(Queue& q, void* dst, const Tensor& a,
                                bool fused_semantics);
}  // namespace vt::rocm

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device GpuDev() { return Device{DeviceType::kROCM, 0}; }

// test-backend-ops.cpp:4277 via test_cuda_quant_dot.cpp:78 — the NMSE band the
// DEFAULT (warp-reduction) arm is held to vs the CPU oracle. Only the
// VT_GEMV_MMVQ=1 arm claims bit-exactness.


constexpr double kMaxNmseVsCpu = 1e-6;

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
};

// Same table discipline as test_cuda_quant_dot.cpp:113 (offsets restated from
// ggml-common.h): the three K-quants the ROCm provider serves natively.
const WeightCase kKQuantCases[] = {
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
};

std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
  }
  return bytes;
}

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

double Nmse(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return num / den;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = GpuDev();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct EnvGuard {
  explicit EnvGuard(bool on) { ::setenv("VT_GEMV_MMVQ", on ? "1" : "0", 1); }
  ~EnvGuard() { ::unsetenv("VT_GEMV_MMVQ"); }
};

}  // namespace

TEST_CASE("ROCm K-quant decode arm (VT_GEMV_MMVQ=1) is BIT-EXACT vs the CPU oracle") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // m=1 (the decode shape the arm serves), Q4_K/Q5_K/Q6_K, nsb edges
  // (nsb=1 -> one partial pass; nsb=3 -> ragged tail pass) and odd-but-valid
  // N (warp-guard edge).
  for (const WeightCase& c : kKQuantCases) {
    for (int64_t nsb : {int64_t{1}, int64_t{3}, int64_t{10}}) {
      const int64_t k = nsb * c.block_elems;
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{129}}) {
        for (uint32_t seed : {0x5EEDU, 0xA11CEU}) {
          CAPTURE(c.name);
          CAPTURE(k);
          CAPTURE(n);
          CAPTURE(seed);

          std::vector<uint8_t> wq = RandomBlocks(c, n * nsb, seed);
          // Engine-realistic dtypes too: the model runs these projections with
          // bf16 activations and bf16 outputs; f32-only tests were the blind
          // spot that let the first fused build pass ops while the engine
          // degraded. Activation storage is generated in `adt`.
          for (DType adt : {DType::kF32, DType::kBF16, DType::kF16}) {
          for (DType odt : {DType::kF32, DType::kBF16}) {
          CAPTURE(adt);
          CAPTURE(odt);
          std::vector<float> af(static_cast<size_t>(k));
          GenerateData(static_cast<float>(seed) + 0.5F * static_cast<float>(int(adt)),
                       af.size(), af.data());
          std::vector<uint8_t> abuf(af.size() *
                                    (adt == DType::kF32 ? 4 : 2));
          for (size_t i2 = 0; i2 < af.size(); ++i2) {
            if (adt == DType::kF32)
              std::memcpy(abuf.data() + 4 * i2, &af[i2], 4);
            else if (adt == DType::kBF16) {
              const uint16_t h = vt::F32ToBF16(af[i2]);
              std::memcpy(abuf.data() + 2 * i2, &h, 2);
            } else {
              const uint16_t h = vt::F32ToF16(af[i2]);
              std::memcpy(abuf.data() + 2 * i2, &h, 2);
            }
          }

          // --- CPU oracle (host tensors, generic nrc==1 tier at m==1) -------
          std::vector<float> cpu_out(static_cast<size_t>(n), 0.0F);
          {
            Tensor at = Tensor::Contiguous(abuf.data(), adt, Cpu(), {1, k});
            Tensor bt =
                Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
            bt.dtype = c.dtype;
            Tensor ot =
                Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {1, n});
            vt::MatmulBTQuant(cq, ot, at, bt);
          }

          // --- ROCm path with the MMVQ decode arm forced ON -----------------
          const size_t oesz = odt == DType::kF32 ? 4 : 2;
          void* d_a = gpu.Alloc(abuf.size());
          void* d_w = gpu.Alloc(wq.size());
          void* d_o = gpu.Alloc(oesz * static_cast<size_t>(n));
          gpu.Copy(gq, d_a, abuf.data(), abuf.size());
          gpu.Copy(gq, d_w, wq.data(), wq.size());
          std::vector<float> rocm_out(static_cast<size_t>(n), 0.0F);
          {
            EnvGuard on(true);
            Tensor at = DevTensor(d_a, adt, {1, k});
            Tensor bt = DevTensor(d_w, c.dtype, {n, k});
            Tensor ot = DevTensor(d_o, odt, {1, n});
            vt::MatmulBTQuant(gq, ot, at, bt);
            // read back through the SAME dtype the kernel wrote
            std::vector<unsigned char> obuf(oesz * static_cast<size_t>(n));
            gpu.Copy(gq, obuf.data(), d_o, obuf.size());
            for (size_t i2 = 0; i2 < rocm_out.size(); ++i2)
              rocm_out[i2] = odt == DType::kF32
                                 ? reinterpret_cast<float*>(obuf.data())[i2]
                                 : vt::BF16ToF32(
                                       reinterpret_cast<uint16_t*>(obuf.data())[i2]);
            gpu.Synchronize(gq);
          }
          gpu.Free(d_a);
          gpu.Free(d_w);
          gpu.Free(d_o);

          // CPU side mirrors the output dtype conversion exactly
          std::vector<float> cpu_ref(cpu_out.size());
          for (size_t i2 = 0; i2 < cpu_out.size(); ++i2)
            cpu_ref[i2] = odt == DType::kF32
                              ? cpu_out[i2]
                              : vt::BF16ToF32(vt::F32ToBF16(cpu_out[i2]));
          CHECK(std::memcmp(rocm_out.data(), cpu_ref.data(),
                            cpu_ref.size() * sizeof(float)) == 0);
          }  // odt
          }  // adt
        }
      }
    }
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("ROCm K-quant DEFAULT arm (env unset) stays within 1e-6 NMSE vs CPU") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // Default-OFF inertness probe: with no VT_GEMV_MMVQ in the environment the
  // baseline warp-reduction kernel must be untouched by the T4a change. The
  // baseline's shfl tree reassociates the float sum, so this holds it to the
  // SAME 1e-6 NMSE-vs-CPU band as the CUDA sibling gate — not bit-exactness.
  const WeightCase& c = kKQuantCases[0];  // q4_K
  const int64_t nsb = 10, k = nsb * c.block_elems, n = 7;
  std::vector<uint8_t> wq = RandomBlocks(c, n * nsb, 0x5EEDU);
  std::vector<float> a(static_cast<size_t>(k));
  GenerateData(1.0F, a.size(), a.data());

  std::vector<float> cpu_out(static_cast<size_t>(n), 0.0F);
  {
    Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {1, k});
    Tensor bt = Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
    bt.dtype = c.dtype;
    Tensor ot = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {1, n});
    vt::MatmulBTQuant(cq, ot, at, bt);
  }

  void* d_a = gpu.Alloc(a.size() * sizeof(float));
  void* d_w = gpu.Alloc(wq.size());
  void* d_o = gpu.Alloc(sizeof(float) * static_cast<size_t>(n));
  gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
  gpu.Copy(gq, d_w, wq.data(), wq.size());
  std::vector<float> rocm_out(static_cast<size_t>(n), 0.0F);
  {
    EnvGuard off(false);  // explicitly "0": the arm must NOT engage
    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    Tensor bt = DevTensor(d_w, c.dtype, {n, k});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, n});
    vt::MatmulBTQuant(gq, ot, at, bt);
    gpu.Copy(gq, rocm_out.data(), d_o, rocm_out.size() * sizeof(float));
    gpu.Synchronize(gq);
  }
  gpu.Free(d_a);
  gpu.Free(d_w);
  gpu.Free(d_o);

  const double nmse = Nmse(rocm_out, cpu_out);
  CAPTURE(nmse);
  CHECK(nmse <= kMaxNmseVsCpu);
  gpu.DestroyQueue(gq);
}

TEST_CASE("Fused-prologue Q8_K quantization is BYTE-IDENTICAL to the standalone quantizer") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  // nsb=10 covers this model's decode K; inputs: pseudo-random rows plus an
  // ADVERSARIAL tied-amax row (+max first, equal-magnitude negative later, so
  // the amax FIRST-occurrence tie-break is what decides mx's sign) and an
  // all-zero row.
  const int64_t k = 10 * 256;
  std::mt19937 rng(0xB00B5U);
  std::vector<std::vector<float>> rows;
  for (int r = 0; r < 4; ++r) {
    std::vector<float> a(static_cast<size_t>(k));
    for (float& v : a) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 500.0F;
    rows.push_back(std::move(a));
  }
  {
    std::vector<float> a(static_cast<size_t>(k), 0.0F);
    a[0] = 3.5F;
    a[17] = -3.5F;  // exact fabs tie; FIRST occurrence (index 0) must win
    a[291] = -3.5F; // another tie, still after index 0
    rows.push_back(std::move(a));
  }
  rows.push_back(std::vector<float>(static_cast<size_t>(k), 0.0F));

  for (size_t r = 0; r < rows.size(); ++r) {
    CAPTURE(r);
    const std::vector<float>& a = rows[r];
    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_sa = gpu.Alloc(10 * 292);   // sizeof(BlockQ8_K), pinned by static_assert
    void* d_sb = gpu.Alloc(10 * 292);
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    vt::rocm::MmvqQuantScratchForTesting(gq, d_sa, at, false);
    vt::rocm::MmvqQuantScratchForTesting(gq, d_sb, at, true);
    std::vector<unsigned char> sa(10 * 292), sb(10 * 292);
    gpu.Copy(gq, sa.data(), d_sa, sa.size());
    gpu.Copy(gq, sb.data(), d_sb, sb.size());
    gpu.Synchronize(gq);
    gpu.Free(d_a); gpu.Free(d_sa); gpu.Free(d_sb);
    CHECK(std::memcmp(sa.data(), sb.data(), sa.size()) == 0);
  }
  gpu.DestroyQueue(gq);
}
