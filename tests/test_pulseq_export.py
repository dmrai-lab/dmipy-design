"""Pulseq export — a NOW design becomes a scanner-runnable spin echo, checked offline.

Requires the ``[pulseq]`` extra (dmipy-sim + pypulseq); skipped otherwise.
"""
import pytest

pytest.importorskip("dmipy_sim")
pytest.importorskip("pypulseq")

from dmipy_design.optimizers import design_waveform_now
from dmipy_design.pulseq_export import design_to_pulseq, pulseq_delivery_report


def test_now_design_exports_to_runnable_prisma_spin_echo(tmp_path):
    d = design_waveform_now(1.0, TE=0.08, n_t=100, n_restarts=6, seed=0)
    assert d.feasible
    seq = design_to_pulseq(d, scanner="siemens_prisma",
                           filename=str(tmp_path / "design.seq"))
    rep = pulseq_delivery_report(d, seq, scanner="siemens_prisma")
    assert rep["timing_ok"], rep["timing_error"]
    assert rep["grad_ok"] and rep["slew_ok"]
    assert rep["b_rel_err"] < 0.15          # assembled .seq encodes ~ the designed b
