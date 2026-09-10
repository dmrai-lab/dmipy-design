"""Pulseq export — a NOW design becomes a scanner-runnable spin echo through dmipy-sim's exporter, checked
offline. Requires the ``[pulseq]`` extra (pypulseq); skipped otherwise.
"""
import numpy as np
import pytest

pytest.importorskip("pypulseq")

from dmipy_design import ScannerLimits
from dmipy_design.optimizers import design_waveform_now
from dmipy_design.pulseq_export import design_to_pulseq, pulseq_delivery_report


def test_now_design_exports_to_runnable_prisma_spin_echo(tmp_path):
    lim = ScannerLimits.of("siemens_prisma")
    d = design_waveform_now(1.0, limits=lim, TE=0.08, n_t=81, n_restarts=6, seed=0)   # dt = 1 ms: on the 10 us raster
    assert d.feasible
    seq = design_to_pulseq(d, scanner="siemens_prisma", filename=str(tmp_path / "design.seq"))
    rep = pulseq_delivery_report(d, seq, scanner="siemens_prisma")
    assert rep["timing_ok"], rep["timing_error"]
    assert rep["grad_ok"] and rep["slew_ok"]
    assert rep["b_rel_err"] < 0.05          # ramps between the design's samples: the delivered b within a few %


def test_a_design_off_the_raster_is_refused_with_the_grids_that_fit():
    d = design_waveform_now(1.0, limits=ScannerLimits.of("siemens_prisma"), TE=0.08, n_t=100, n_restarts=2, seed=0)
    with pytest.raises(ValueError, match=r"not a whole number of the 10 us gradient raster.*n_t = \[81, 101\]"):
        design_to_pulseq(d, scanner="siemens_prisma")
