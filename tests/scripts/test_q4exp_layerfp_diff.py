#!/usr/bin/env python3
"""The layer axis survives the comparator, and the legacy loader is why it did not.

#2877. `scripts/q4exp-layerfp-diff.py` replaces the differ inlined in
`docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904/run2-job.sh`, which
collapsed 1311 tap lines to 42 because it built its key by splitting each token on
`'='` while the tap prints the layer as `L%+03lld` -- `L+00`, with no `=`.

`LegacyLoaderReproduction` runs the committed loader VERBATIM on lines built from
the instrument's exact `printf`, and asserts it collapses. That is the red. It is
the contrast that makes the counted property below mean something: without it, "48
rows" is a number with nothing to fail against.

`RepairedLoader` asserts the counted property -- 48 synthetic per-layer rows load
as 48, with 48 distinct layers -- and that the tool REFUSES rather than
deduplicates when a key really does repeat, and REFUSES a fingerprint whose
`taps=N END` disagrees with the rows parsed.

`MetricHonesty` pins the thing the numbers are worth: `rel_sumabs` is a difference
of NORMS and reads ZERO on two tensors that differ in every element, while
`head_dmax` -- an exact elementwise difference over the emitted values -- does not.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "q4exp-layerfp-diff.py"
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location("q4exp_layerfp_diff", TOOL)
diff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diff)


def tap_line(step, il, tag, sumabs, v=(0.0, 0.0, 0.0, 0.0)):
    """The instrument's EXACT format, src/vllm/model_executor/models/qwen4_exp_forward.cpp."""
    return ("q4fp step=%d L%+03d tag=%-10s dtype=%-4s dev=%d n=%d "
            "nonfinite=%d maxabs=%.9g sumabs=%.9g v=%.9g,%.9g,%.9g,%.9g\n"
            % (step, il, tag, "bf16", 0, 12800, 0, 1.0, sumabs, v[0], v[1], v[2], v[3]))


def fingerprint(path, layers=48, tags=("blk",), sumabs=lambda il, tag: 100.0, v=None,
                steps=1):
    """Write `steps` fingerprinted forwards, exactly as the instrument prints them.

    `taps=` is CUMULATIVE. `LayerFp` does `++s.taps` on a counter that
    `LayerFpEndStep` never resets (`qwen4_exp_forward.cpp:153`), so a real
    three-step run of a 437-tap forward closes its steps with `taps=437`,
    `taps=874`, `taps=1311` -- which is what the committed
    `run2-results.txt` records. `steps=1` is the degenerate case where cumulative
    and per-step are the SAME number, and it is the only case the first version of
    this file could express.
    """
    lines, taps = [], 0
    for step in range(steps):
        for il in range(layers):
            for tag in tags:
                vv = v(il, tag) if v else (0.0, 0.0, 0.0, 0.0)
                lines.append(tap_line(step, il, tag, sumabs(il, tag), vv))
                taps += 1
        lines.append("q4fp step=%d taps=%d END\n" % (step, taps))
    path.write_text("".join(lines), encoding="utf-8")
    return taps


# The loader as committed in run2-job.sh, copied VERBATIM. Do not repair it here:
# its whole job in this file is to fail. Its unclosed `open()` is part of what was
# committed, so the ResourceWarning is filtered rather than the line changed.
warnings.filterwarnings("ignore", category=ResourceWarning)


def legacy_load(p):
    rows, order = {}, []
    for ln in open(p):
        if not ln.startswith('q4fp ') or ' taps=' in ln:
            continue
        f = {}
        for tok in ln.split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                f[k] = v
        key = (f.get('step'), f.get('L'), f.get('tag'))
        if key in rows:
            continue
        rows[key] = f
        order.append(key)
    return rows, order


class LegacyLoaderReproduction(unittest.TestCase):
    """THE RED: the committed loader collapses 48 layers to 1, silently."""

    def test_legacy_collapses_the_layer_axis(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            taps = fingerprint(p, layers=48, sumabs=lambda il, tag: 100.0 + il)
            self.assertEqual(taps, 48)
            rows, order = legacy_load(p)
            self.assertEqual(len(rows), 1, "the #2877 defect: 48 taps must collapse to 1")
            self.assertEqual(order[0][1], None, "the layer field must never parse")
            self.assertEqual(float(rows[order[0]]["sumabs"]), 100.0,
                             "first-wins dedup keeps layer 0 and drops 1..47")

    def test_legacy_probe_form_can_report_48(self):
        """POSITIVE CONTROL: the same loader on an `L=` spelling gives 48 rows,
        so the 1 above is the defect and not a dead probe."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text("".join(
                "q4fp step=0 L=%+03d tag=blk dtype=bf16 dev=0 n=1 nonfinite=0 "
                "maxabs=1 sumabs=%.9g v=0,0,0,0\n" % (il, 100.0 + il)
                for il in range(48)), encoding="utf-8")
            rows, _ = legacy_load(p)
            self.assertEqual(len(rows), 48)


class RepairedLoader(unittest.TestCase):
    """THE GREEN, as a COUNTED PROPERTY: 48 rows and 48 distinct layers."""

    def test_counted_property_48_rows_48_layers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            fingerprint(p, layers=48, sumabs=lambda il, tag: 100.0 + il)
            rows, order, declared = diff.parse(p)
            self.assertEqual(len(rows), 48)
            self.assertEqual(len({k[1] for k in rows}), 48)
            self.assertEqual(declared["0"], 48)
            checks = diff.check_counted_property(str(p), rows, declared)
            self.assertEqual(checks, [(0, 48, 48, True)])

    def test_refuses_a_duplicate_key_instead_of_deduplicating(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text(tap_line(0, 0, "blk", 1.0) + tap_line(0, 0, "blk", 2.0),
                         encoding="utf-8")
            with self.assertRaises(diff.ParseError) as cm:
                diff.parse(p)
            self.assertIn("duplicate key", str(cm.exception))

    def test_refuses_a_line_with_no_layer_field(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text("q4fp step=0 tag=blk dtype=bf16 dev=0 n=1 nonfinite=0 "
                         "maxabs=1 sumabs=1 v=0,0,0,0\n", encoding="utf-8")
            with self.assertRaises(diff.ParseError) as cm:
                diff.parse(p)
            self.assertIn("no L<layer> field", str(cm.exception))

    def test_refuses_an_empty_parse(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text("nothing here\n", encoding="utf-8")
            with self.assertRaises(diff.ParseError) as cm:
                diff.parse(p)
            self.assertIn("ZERO tap rows", str(cm.exception))

    def test_cli_compares_every_layer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / "a.txt"
            b = pathlib.Path(td) / "b.txt"
            fingerprint(a, layers=48, sumabs=lambda il, tag: 100.0 + il)
            fingerprint(b, layers=48, sumabs=lambda il, tag: 100.0 + il + 0.001)
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b), "--top", "0"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("TAPS COMPARED                 : 48", r.stdout)
            self.assertIn("distinct layers: base=48 other=48", r.stdout)
            self.assertIn("DIFFERENCE OF NORMS", r.stdout)

    def test_cli_refuses_a_counted_property_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / "a.txt"
            b = pathlib.Path(td) / "b.txt"
            fingerprint(a, layers=48, sumabs=lambda il, tag: 1.0)
            # 48 rows but the instrument declares 60: the comparator must refuse.
            b.write_text("".join(tap_line(0, il, "blk", 1.0) for il in range(48))
                         + "q4fp step=0 taps=60 END\n", encoding="utf-8")
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("did not parse every tap", r.stderr)


class CumulativeTapCounter(unittest.TestCase):
    """`taps=N END` is a RUNNING TOTAL, and reading it per-step refuses real data.

    THE RED THIS CLASS EXISTS FOR. Every case above fingerprints ONE step, where
    the cumulative count and the per-step count are the same number, so the first
    version of the tool -- which subtracted nothing -- passed all eleven of them
    and still exited 2 on every genuine multi-step run. The committed evidence
    reads `q4fp step=0 taps=437 END q4fp step=1 taps=874 END q4fp step=2
    taps=1311 END` (`run2-results.txt:12`), and the tool refused it.

    The instrument is the authority for the shape, not this file:
    `src/vllm/model_executor/models/qwen4_exp_forward.cpp:153` increments
    `s.taps` per tap and `LayerFpEndStep` prints it without resetting.
    """

    def _three_steps(self, td, name, bump=0.0):
        p = pathlib.Path(td) / name
        fingerprint(p, layers=3, steps=3, sumabs=lambda il, tag: 100.0 + il + bump)
        return p

    def test_the_evidence_shape_is_cumulative_not_per_step(self):
        """The fixture reproduces 437/874/1311's ARITHMETIC at 3 layers x 3 steps."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = self._three_steps(td, "fp.txt")
            declared = [ln.split("taps=")[1].split()[0]
                        for ln in p.read_text(encoding="utf-8").splitlines()
                        if " taps=" in ln]
            self.assertEqual(declared, ["3", "6", "9"],
                             "the instrument declares a running total, not 3,3,3")

    def test_counted_property_accepts_a_real_multi_step_fingerprint(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = self._three_steps(td, "fp.txt")
            rows, _, declared = diff.parse(p)
            self.assertEqual(len(rows), 9)
            checks = diff.check_counted_property(str(p), rows, declared)
            self.assertEqual(checks, [(0, 3, 3, True), (1, 6, 6, True), (2, 9, 9, True)],
                             "step 1 declares 6 taps SINCE STEP 0, and 6 have been parsed")

    def test_cli_does_not_refuse_a_real_multi_step_fingerprint(self):
        """THE WHOLE FINDING: the tool exited 2 on every genuine run."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = self._three_steps(td, "a.txt")
            b = self._three_steps(td, "b.txt", bump=0.001)
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b), "--top", "0"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("did not parse every tap", r.stderr)
            self.assertIn("TAPS COMPARED                 : 9", r.stdout)
            self.assertIn("(cumulative; +3 this step)", r.stdout)

    def test_a_short_step_is_still_refused_under_the_cumulative_reading(self):
        """The repair must not become 'accept anything'. A missing tap still reds."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = self._three_steps(td, "a.txt")
            b = pathlib.Path(td) / "b.txt"
            # Steps 0 and 2 complete; step 1 prints two of its three taps.
            lines = []
            for step, present in ((0, 3), (1, 2), (2, 3)):
                for il in range(present):
                    lines.append(tap_line(step, il, "blk", 100.0 + il))
                lines.append("q4fp step=%d taps=%d END\n" % (step, 3 * (step + 1)))
            b.write_text("".join(lines), encoding="utf-8")
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("did not parse every tap", r.stderr)
            self.assertIn("RUNNING TOTAL", r.stderr)

    def test_taps_past_the_last_END_are_refused_as_a_truncated_capture(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text(tap_line(0, 0, "blk", 1.0)
                         + "q4fp step=0 taps=1 END\n"
                         + tap_line(1, 0, "blk", 1.0), encoding="utf-8")
            rows, _, declared = diff.parse(p)
            self.assertEqual(diff.unclosed_steps(rows, declared), [1])
            r = subprocess.run([sys.executable, str(TOOL), str(p), str(p)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("truncated", r.stderr)


class MetricSpread(unittest.TestCase):
    """WHAT A RATIO BETWEEN TWO `rel(sumabs)` NUMBERS IS WORTH.

    The first version of this file's tool quoted ONE seed draw -- "122.7x at
    sigma 1e-3 and 229.8x at sigma 1e-4" -- to four significant figures. It is a
    distribution, its median is not those numbers, and its spread is the part
    that decides whether any ratio in the #2877 reading means anything.

    Hermetic: standard library only, fixed seeds, no artifact and no GPU. It
    models the perturbation as i.i.d. zero-mean against a Gaussian signal at the
    committed `L00 moe` scale (`n = 12800`, `sum|x| ~ 390`), which is the premise
    under which `rel(sumabs)` is being read. It bounds the METRIC's resolution.
    It is not a significance test on the real tensors.
    """

    N = 12800
    SIGMA_A = 0.0382  # sum|a| ~ 390 over n = 12800

    def _draw(self, seed, sigma, aligned=False):
        import math
        import random
        r = random.Random(seed)
        a = [r.gauss(0.0, self.SIGMA_A) for _ in range(self.N)]
        if aligned:
            e = [math.copysign(abs(r.gauss(0.0, sigma)), x) for x in a]
        else:
            e = [r.gauss(0.0, sigma) for _ in range(self.N)]
        sa = sum(abs(x) for x in a)
        sb = sum(abs(x + y) for x, y in zip(a, e))
        m = max(sa, sb)
        # (the committed metric, the honest one)
        return abs(sa - sb) / m, sum(abs(y) for y in e) / m

    @staticmethod
    def _pct(values, q):
        v = sorted(values)
        return v[min(len(v) - 1, int(q / 100.0 * len(v)))]

    def test_positive_control_a_sign_aligned_perturbation_reads_1x(self):
        """Where the two measures MUST agree they do, so the gap below is real."""
        for seed in range(8):
            rs, rl = self._draw(seed, 1e-3, aligned=True)
            self.assertAlmostEqual(rl / rs, 1.0, places=6)

    def test_the_under_report_is_a_distribution_and_not_122x(self):
        draws = [self._draw(s, 1e-3) for s in range(64)]
        ratios = sorted(rl / rs for rs, rl in draws)
        median = ratios[len(ratios) // 2]
        self.assertGreater(median, 20.0, "a zero-mean perturbation is heavily under-read")
        self.assertLess(median, 400.0)
        # The point of the case: p05..p95 spans an order of magnitude, so no
        # single figure -- 122.7x, 229.8x or this median -- is a constant.
        self.assertGreater(self._pct(ratios, 95) / self._pct(ratios, 5), 10.0)

    def test_at_a_FIXED_true_divergence_the_metric_still_spans_an_order_of_magnitude(self):
        draws = [self._draw(s, 1e-3) for s in range(64)]
        rel_sumabs = [rs for rs, _ in draws]
        true_l1 = [rl for _, rl in draws]
        self.assertLess(max(true_l1) / min(true_l1), 1.1,
                        "the TRUE divergence is held fixed across these seeds")
        self.assertGreater(max(rel_sumabs) / min(rel_sumabs), 20.0,
                           "the committed metric is not, on the same divergence")

    def test_a_ratio_of_two_readings_below_about_18x_ranks_nothing(self):
        """The consequence for #2877: 1.80x, 2.02x and 3.15x are NO CHANGE.

        Pairs drawn from readings of the SAME true divergence. If the moves the
        reading argues over are ordinary values of this ratio, the reading cannot
        order them -- in either direction.
        """
        rel_sumabs = [self._draw(s, 1e-3)[0] for s in range(64)]
        pairs = []
        for i in range(len(rel_sumabs)):
            for j in range(i + 1, len(rel_sumabs)):
                x, y = rel_sumabs[i], rel_sumabs[j]
                pairs.append(max(x, y) / min(x, y))
        pairs.sort()
        for move in (1.80, 2.02, 3.15):
            share = sum(1 for r in pairs if r >= move) / len(pairs)
            self.assertGreater(share, 0.20,
                               "%.2fx is an ordinary reading of an UNCHANGED "
                               "divergence (%.0f%% of pairs reach it)"
                               % (move, 100 * share))
        self.assertGreater(self._pct(pairs, 95), 8.0,
                           "two readings of one divergence differ by ~an order of "
                           "magnitude at p95")

    def test_the_tool_reports_the_spread_and_not_a_single_constant(self):
        """Every run's stdout must carry what the cases above measured.

        The first version printed `~122x under-report measured at this tap's
        n=12800` -- one seed draw, presented as the property of the tap.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / "a.txt"
            b = pathlib.Path(td) / "b.txt"
            fingerprint(a, layers=2, steps=2)
            fingerprint(b, layers=2, steps=2)
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("DISTRIBUTION, not a constant", r.stdout)
            self.assertIn("p05..p95", r.stdout)
            self.assertNotIn("~122x", r.stdout,
                             "one seed draw must not be printed as the tap's property")


class MetricHonesty(unittest.TestCase):
    """rel_sumabs is a difference of NORMS. head_dmax is a difference."""

    def test_rel_sumabs_reads_zero_on_two_wholly_different_tensors(self):
        # Equal L1 norms, opposite signs: every element differs, rel_sumabs == 0.
        self.assertEqual(diff.rel_sumabs("390.0", "390.0"), 0.0)

    def test_head_dmax_sees_what_rel_sumabs_cannot(self):
        a = {"v": "1.0,2.0,3.0,4.0", "sumabs": "10"}
        b = {"v": "-1.0,-2.0,-3.0,-4.0", "sumabs": "10"}
        self.assertEqual(diff.rel_sumabs(a["sumabs"], b["sumabs"]), 0.0,
                         "the norms are equal, so the committed metric reads agreement")
        self.assertEqual(diff.head_dmax(a, b), 8.0,
                         "an elementwise difference cannot cancel")

    def test_head_dmax_positive_control_is_zero_on_identical_values(self):
        a = {"v": "1.0,2.0,3.0,4.0"}
        self.assertEqual(diff.head_dmax(a, dict(a)), 0.0)


if __name__ == "__main__":
    unittest.main()
