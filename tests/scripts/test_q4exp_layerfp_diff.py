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


def fingerprint(path, layers=48, tags=("blk",), sumabs=lambda il, tag: 100.0, v=None):
    lines, taps = [], 0
    for il in range(layers):
        for tag in tags:
            vv = v(il, tag) if v else (0.0, 0.0, 0.0, 0.0)
            lines.append(tap_line(0, il, tag, sumabs(il, tag), vv))
            taps += 1
    lines.append("q4fp step=0 taps=%d END\n" % taps)
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
