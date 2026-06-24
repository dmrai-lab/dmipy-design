"""Tier-2 deliverability: a designed waveform exports to a runnable Pulseq spin echo
that passes the vendor-agnostic acceptance checks we can run without a scanner -- timing
(raster/dead-time/contiguity), the Gmax/slew box, and a b-tensor round-trip."""
import numpy as np
import pytest

pytest.importorskip("pypulseq")

from dmipy_design.optimizers import design_waveform, SequenceTiming
from dmipy_design.pulseq_export import design_to_pulseq, pulseq_delivery_report


def test_design_exports_to_runnable_prisma_spin_echo(tmp_path):
    """An LTE design on a real Prisma timing budget exports to a .seq that (a) writes and
    passes check_timing, (b) is within the Prisma Gmax/slew limits, and (c) re-encodes the
    designed b to a few percent (the export drops the held <=2% gradient at the 180 and
    adds the slew-limited ramps the design omits)."""
    d = design_waveform(1.0, G_max=0.08, slew_rate_max=200.0, TE=0.060,
                        timing=SequenceTiming(t_excite=3e-3, t_refocus=6e-3,
                                              t_readout_pre_echo=14e-3),
                        n_t=200, n_restarts=16, n_outer=12,
                        null_M1=False, null_M2=False, maxwell=False, seed=0)
    assert d.feasible, d.report

    seq = design_to_pulseq(d, scanner='siemens_prisma')
    seq.write(str(tmp_path / 'lte_prisma.seq'))               # raster-aligned -> writes

    rep = pulseq_delivery_report(d, seq, scanner='siemens_prisma')
    assert rep['timing_ok'], rep['timing_error']              # Pulseq raster/limit checks
    assert rep['grad_ok'], (rep['max_grad_mT'], rep['limit_grad_mT'])
    assert rep['slew_ok'], (rep['max_slew'], rep['limit_slew'])
    assert rep['b_rel_err'] < 0.05, rep                       # b round-trips: ask == encode
