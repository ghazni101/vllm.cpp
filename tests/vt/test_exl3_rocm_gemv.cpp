// The EXL3 m<=8 GEMV arm on ROCm — spec .agents/specs/rocm-exl3-gemv.md,
// row BACKEND-ROCM, issue .agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3QJSFATGWXMMQKTVDG6BG5G.md.
//
// THE BOUND IS TIER 3c, NOT THE BYTE GATE. test_exl3_rocm.cpp pins
// Exl3GemmK BYTE-equal to the CPU arm because that kernel reproduces the
// host's accumulation ORDER. This arm accumulates in fp16 fragments and folds
// to f32 every FOLD k-tiles (exl3_gemv_kernel.cuh:37-52) — the same numeric
// arm as the CUDA GEMV, which spec quant-exl3-perf bounds at 6.0e-3 relative
// RMS. The emulated mma_m16n8k16_f16 reproduces that accumulation order
// exactly (its __hfma2 lane-pair fold IS CUDA's mma.f16.f16.f16.f16), so the
// bound transfers unchanged. `VT_ROCM_EXL3_WMMA=1` swaps in the gfx11 WMMA
// path, which carries its OWN numerics and is evaluation-only; this suite is
// written to the default arm.
//
// DISPATCH IS GATED IN BOTH DIRECTIONS, because this backend's two arms are
// distinguishable BYTE-FOR-BYTE. force_gemv = 0 selects Exl3GemmK, whose
// output is byte-identical to the CPU arm — so a decline result must be
// byte-equal to the CPU reference, and a GEMV result must not be. That is a
// sharper reach check than the CUDA suite's, whose two f32/f16 arms already
// differed incidentally.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;
using exl3_test::UlpF16;

bool HasRocmExl3() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kROCM);
    return vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM);
  } catch (const std::runtime_error&) {
    return false;
  }
}

std::vector<uint16_t> CpuArm(const Exl3Fixture& f, vt::Queue& hq,
                             const std::vector<uint16_t>& a, int64_t m, int codebook) {
  const int64_t k = f.k, n = f.n;
  std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
  std::vector<uint16_t> a_had_h(static_cast<size_t>(m * k), 0);
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  vt::Tensor ta = vt::Tensor::Contiguous(const_cast<uint16_t*>(a.data()), vt::DType::kF16,
                                       hq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had_h.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tc = vt::Tensor::Contiguous(out.data(), vt::DType::kF16, hq.device, {m, n});
  vt::Tensor tb =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8,
                             hq.device, {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()),
                                           vt::DType::kF16, hq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()),
                                           vt::DType::kF16, hq.device, {n});
  vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, args);
  return out;
}

std::vector<uint16_t> RocmArm(vt::Backend& be, const Exl3Fixture& f,
                              const std::vector<uint16_t>& a, int64_t m, int codebook,
                              int force_gemv) {
  vt::Queue dq = be.CreateQueue();
  const int64_t k = f.k, n = f.n;
  const size_t ab = a.size() * sizeof(uint16_t);
  const size_t bb = f.trellis.size() * sizeof(uint16_t);
  std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);

  void* d_a = be.Alloc(ab);
  void* d_ah = be.Alloc(ab);
  void* d_b = be.Alloc(bb);
  void* d_suh = be.Alloc(f.suh.size() * 2);
  void* d_svh = be.Alloc(f.svh.size() * 2);
  void* d_c = be.Alloc(out.size() * 2);
  be.Copy(dq, d_a, a.data(), ab);
  be.Copy(dq, d_b, f.trellis.data(), bb);
  be.Copy(dq, d_suh, f.suh.data(), f.suh.size() * 2);
  be.Copy(dq, d_svh, f.svh.data(), f.svh.size() * 2);

  vt::Tensor ta = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tb =
      vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(d_c, vt::DType::kF16, dq.device, {m, n});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  args.force_gemv = force_gemv;
  vt::Exl3Gemm(dq, tc, ta, tb, tsuh, tsvh, tah, args);
  be.Synchronize(dq);
  be.Copy(dq, out.data(), d_c, out.size() * 2);
  be.Synchronize(dq);
  be.Free(d_a);
  be.Free(d_ah);
  be.Free(d_c);
  be.Free(d_b);
  be.Free(d_suh);
  be.Free(d_svh);
  be.DestroyQueue(dq);
  return out;
}

}  // namespace

// Every instantiated arm meets tier 3c against the CPU arm, and every arm is
// FAR from the same bits decoded under a different codebook — the
// discrimination check that keeps a mis-threaded `cb` from reporting green.
// Upstream's selector table is instantiated: 4 bpw all three codebooks, 3 bpw
// cb 1/2; cb 0 at 3 bpw is refused upstream (exl3_gemv.cu:113) and here.
TEST_CASE("exl3 rocm gemv: every instantiated arm meets tier 3c") {
  if (!HasRocmExl3()) {
    MESSAGE(
        "SKIPPED, no ROCm device: rocm-exl3-gemv's tier-3c bound is PENDING. Reproduce with: "
        "ctest --test-dir build-hip -R test_exl3_rocm_gemv -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  struct Arm {
    int bits, cb;
    int64_t k, n, m;
    std::string what;
  };
  // Shapes: k=2048 n=4096 resolves the envelope to cfg 0 on every cc bucket
  // (the `size_k <= 2048 && size_n <= 8192` branch), n=8320 resolves mode 2 to
  // cfg 1 (wide). m=1 exercises mmode 0, m=8 mmode 1 — different compiled
  // kernels (ROWS 1 vs 8), so both get a run.
  const Arm kArms[] = {
      {4, 0, 2048, 4096, 1, "(4,0) narrow m=1"},  // the stock turboderp arm
      {4, 1, 2048, 4096, 1, "(4,1) narrow m=1"},
      {4, 2, 2048, 4096, 1, "(4,2) narrow m=1"},
      {3, 1, 2048, 4096, 1, "(3,1) narrow m=1"},
      {3, 2, 2048, 4096, 1, "(3,2) narrow m=1"},
      {4, 2, 2048, 8320, 1, "(4,2) wide m=1"},
      {4, 0, 2048, 4096, 8, "(4,0) narrow m=8"},
      {3, 1, 2048, 4096, 2, "(3,1) narrow m=2"},
  };

  for (const Arm& arm : kArms) {
    CAPTURE(arm.what);
    CAPTURE(arm.bits);
    CAPTURE(arm.cb);
    const int64_t k = arm.k, n = arm.n, m = arm.m;
    Exl3Fixture f = MakeFixture(k, n, arm.bits, 0x5EEDu);
    std::vector<uint16_t> a(static_cast<size_t>(m * k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    const int sibling_cb = arm.cb == 2 ? 1 : 2;
    const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, arm.cb);
    const std::vector<uint16_t> sib = CpuArm(f, hq, a, m, sibling_cb);

    // FORCED — force_gemv=1, upstream's own direct-entry lever, so a device
    // whose heuristic declined cannot report a transcription result as this
    // arm's.
    const std::vector<uint16_t> got = RocmArm(be, f, a, m, arm.cb, /*force_gemv=*/1);

    double sq = 0.0, rq = 0.0, worst = 0.0, sq_sib = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double r = vt::F16ToF32(ref[i]);
      const double s = vt::F16ToF32(sib[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      sq_sib += (g - s) * (g - s);
      rq += r * r;
      worst = std::max(worst, std::fabs(g - r));
    }
    const double rms_ref = std::sqrt(rq / static_cast<double>(got.size()));
    const double rel = std::sqrt(sq / static_cast<double>(got.size())) / rms_ref;
    const double rel_sib = std::sqrt(sq_sib / static_cast<double>(got.size())) / rms_ref;
    MESSAGE(arm.what, " tier 3c: relative RMS ", rel, ", worst elementwise ", worst,
            ", relative RMS against codebook ", sibling_cb, " ", rel_sib);
    CHECK(rel <= 6.0e-3);
    CHECK(worst <= 64.0 * UlpF16(rms_ref));
    CHECK(rms_ref > 0.0);
    CHECK(rel_sib > 100.0 * 6.0e-3);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

// Dispatch pinned in BOTH directions — possible because this backend's two
// arms are byte-distinguishable. force_gemv=0 is Exl3GemmK and MUST reproduce
// the CPU arm byte-for-byte; force_gemv=1 is the GEMV and MUST differ from it
// (the fp16 fold order is not the host's f32 order, so a byte-equal GEMV
// result means the arm never ran).
TEST_CASE("exl3 rocm gemv: force_gemv selects arms byte-detectably") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  Exl3Fixture f = MakeFixture(2048, 4096, 4, 0xC0DEu);
  const int64_t m = 1;
  std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/0);
  const std::vector<uint16_t> off = RocmArm(be, f, a, m, 0, /*force_gemv=*/0);
  const std::vector<uint16_t> on = RocmArm(be, f, a, m, 0, /*force_gemv=*/1);

  size_t off_equal = 0, on_diff = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (off[i] == ref[i]) ++off_equal;
    if (on[i] != ref[i]) ++on_diff;
  }
  MESSAGE("force_gemv=0 byte-equal to CPU arm: ", off_equal, " of ", ref.size());
  MESSAGE("force_gemv=1 differs from CPU arm at ", on_diff, " of ", ref.size(), " outputs");
  CHECK(off_equal == ref.size());      // the transcription is byte-exact
  // The forced arm provably ran if ANY output differs: the m1 arm accumulates
  // in f32 (v_dot2_f32_f16) and lands within fp16 rounding of the CPU
  // reference on ~99% of elements, so the old `> ref.size()/2` threshold —
  // calibrated to the fp16-fragment arm's error profile — fails on a
  // strictly-more-accurate arm. >0 is the right discriminator.
  CHECK(on_diff > 0);

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

// NOTHING LANDS DEAD. The production call leaves force_gemv at -1, so the
// envelope decides. At k=2048 n=4096 the `size_k <= 2048 && size_n <= 8192`
// branch resolves to cfg 0 on every bucket — including Exl3Cc::kAda, which is
// what Exl3GemvTryLaunchRocm passes for RDNA3 — with NO occupancy input, so a
// default-mode launch on this device must produce the GEMV arm's bytes.
TEST_CASE("exl3 rocm gemv: the default dispatch reaches the GEMV at mode 1") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);

  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 2048, 4096, 4, 0,
                                 /*mode=*/1, /*narrow_coresident=*/0) == 0);

  Exl3Fixture f = MakeFixture(2048, 4096, 4, 0xC0DEu);
  const int64_t m = 1;
  std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  const std::vector<uint16_t> unforced = RocmArm(be, f, a, m, 0, /*force_gemv=*/-1);
  const std::vector<uint16_t> forced = RocmArm(be, f, a, m, 0, /*force_gemv=*/1);
  size_t same = 0;
  for (size_t i = 0; i < unforced.size(); ++i)
    if (unforced[i] == forced[i]) ++same;
  MESSAGE("(4,0) reached UNFORCED at mode 1: ", same, " of ", unforced.size(),
          " outputs byte-equal to the forced launch");
  CHECK(same == unforced.size());
}

// ─── spec rocm-exl3-gemv-residual: the three residual shapes ────────────────
//
// The rocprofv3 trace that filed ISSUE-LOCAL-01M3RM2TM98FY569AZ6CEASBD3 found
// Exl3GemmK still owned 91% of decode kernel time on this checkpoint because
// (a) the upstream selector declines k=4096/n=4096/cb1 at 4 bpw, (b) the 6 bpw
// lm_head is outside the GEMV envelope, and (c) m in (8, 144] had no arm at
// all. These cases pin each fix against a DETECTABLE wrong arm: on this
// backend the transcription is byte-equal to the CPU arm, so a GEMV or recon
// result that differs from it proves the fast path ran, and equality to the
// forced launch proves WHICH fast path ran.

TEST_CASE("exl3 rocm gemv: upstream-declined (4,1) shape takes the forced cfg") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // The exact decline the trace recorded: k=4096 n=4096 (4,1) falls through
  // every branch of Exl3GemvSelectConfig at mode 1 (size_n/32=128 >
  // narrow_coresident, size_n < 8192) — upstream returns -1 and the CUDA arm
  // would run its tensor-core GEMM. Assert the selector STILL returns -1 (the
  // function is kept upstream-verbatim) and that the unforced launch now
  // lands the GEMV arm anyway, byte-equal to the forced one.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 4096, 4, 1,
                                 /*mode=*/1, /*narrow_coresident=*/0) == -1);

  Exl3Fixture f = MakeFixture(4096, 4096, 4, 0x9EEDEDu);
  const int64_t m = 1;
  std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/1);
  const std::vector<uint16_t> scalar = RocmArm(be, f, a, m, 1, /*force_gemv=*/0);
  const std::vector<uint16_t> unforced = RocmArm(be, f, a, m, 1, /*force_gemv=*/-1);
  const std::vector<uint16_t> forced = RocmArm(be, f, a, m, 1, /*force_gemv=*/1);

  size_t scalar_eq = 0, unf_eq_forced = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (scalar[i] == ref[i]) ++scalar_eq;
    if (unforced[i] == forced[i]) ++unf_eq_forced;
  }
  MESSAGE("(4,1) k=4096 n=4096: scalar byte-equal to CPU ", scalar_eq, " of ", ref.size(),
          "; unforced == forced at ", unf_eq_forced);
  CHECK(scalar_eq == ref.size());      // the transcription is still byte-exact
  CHECK(unf_eq_forced == ref.size());  // the fallback cfg reached the GEMV arm

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

TEST_CASE("exl3 rocm gemv: mid-m chunked GEMV meets tier 3c") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // m in (8, 32] exercises the 8-row chunk loop — 9 is the smallest mid-m,
  // 21 the recorded prefill shape, 32 the cap's edge. m=33 routes to the
  // reconstruct arm and is asserted NOT byte-equal to the transcription the
  // same way (recon's own suite holds its 1.0e-3 bound).
  for (const int64_t m : {9, 21, 32, 33}) {
    CAPTURE(m);
    Exl3Fixture f = MakeFixture(2048, 4096, 4, 0x51CE00u + static_cast<uint32_t>(m));
    std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/1);
    const std::vector<uint16_t> scalar = RocmArm(be, f, a, m, 1, /*force_gemv=*/0);
    const std::vector<uint16_t> got = RocmArm(be, f, a, m, 1, /*force_gemv=*/-1);

    size_t scalar_eq = 0, fast_diff = 0;
    double sq = 0.0, rq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (scalar[i] == ref[i]) ++scalar_eq;
      if (got[i] != scalar[i]) ++fast_diff;
      const double r = vt::F16ToF32(ref[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      rq += r * r;
    }
    const double rel =
        std::sqrt(sq / static_cast<double>(ref.size())) /
        std::sqrt(rq / static_cast<double>(ref.size()));
    MESSAGE("m=", m, " (4,1): rel RMS ", rel,
            ", fast arm differs from the transcription at ", fast_diff, " of ",
            ref.size());
    CHECK(scalar_eq == ref.size());               // force_gemv=0 is still the byte arm
    CHECK(rel <= (m <= 32 ? 6.0e-3 : 1.0e-2));    // chunked dot / recon bound
    // The dot arm accumulates in f32, so it is byte-NEARER the transcription
    // than the fp16-fragment GEMV — a handful of differing outputs, not half
    // of them, is what proves a fast arm ran.
    CHECK(fast_diff > 0);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

TEST_CASE("exl3 rocm gemv: 6 bpw takes the reconstruct arm at every m") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // The checkpoint's lm_head is 6 bpw cb 1 — outside upstream's GEMV envelope
  // entirely. The residual spec routes it to the reconstruct+hipBLAS arm
  // inside Exl3GemmKernelRocm, whose bound is 1.0e-3 rel RMS; m=1 is decode,
  // m=8 the GEMV boundary, m=21 the recorded prefill shape.
  for (const int64_t m : {1, 8, 21}) {
    CAPTURE(m);
    Exl3Fixture f = MakeFixture(256, 4096, 6, 0x6EAD0u + static_cast<uint32_t>(m));
    std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/1);
    const std::vector<uint16_t> scalar = RocmArm(be, f, a, m, 1, /*force_gemv=*/0);
    const std::vector<uint16_t> got = RocmArm(be, f, a, m, 1, /*force_gemv=*/-1);

    size_t scalar_eq = 0, fast_diff = 0;
    double sq = 0.0, rq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (scalar[i] == ref[i]) ++scalar_eq;
      if (got[i] != scalar[i]) ++fast_diff;
      const double r = vt::F16ToF32(ref[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      rq += r * r;
    }
    const double rel =
        std::sqrt(sq / static_cast<double>(ref.size())) /
        std::sqrt(rq / static_cast<double>(ref.size()));
    MESSAGE("bits=6 cb=1 m=", m, ": rel RMS ", rel,
            ", dot arm differs from the transcription at ", fast_diff, " of ",
            ref.size());
    CHECK(scalar_eq == ref.size());  // bits 6 is still byte-exact off the fast path
    CHECK(rel <= 1.0e-2);            // f32 accumulation, well inside tier 3
    CHECK(fast_diff > 0);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}
