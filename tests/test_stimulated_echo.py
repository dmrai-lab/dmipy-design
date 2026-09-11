"""PGSTE -- stimulated-echo design via the NOW core, in dmipy-sim's stimulated-echo layout."""
import numpy as np
import pytest

from dmipy_design import ScannerLimits
from dmipy_design.optimizers import SequenceTiming, design_stimulated_echo, NowDesign

LIM = ScannerLimits.of("siemens_prisma")
TIMING = SequenceTiming(t_excite=2e-3, t_refocus=4e-3, t_readout_pre_echo=2e-3)


def test_design_stimulated_echo_is_feasible_and_is_a_stimulated_echo():
    d = design_stimulated_echo(1.0, limits=LIM, TM=50e-3, TE=0.12, timing=TIMING, n_t=80, n_restarts=3, seed=0)
    assert isinstance(d, NowDesign) and d.feasible and d.b_value > 0
    assert d.refocus_residual < 1e-2
    assert d.store_idx is not None and d.recall_idx == d.echo_idx       # the sign flips at the recall
    seq = d.to_sequence()
    assert seq.stimulated_echo and seq.TM == pytest.approx(50e-3, abs=2 * seq.dt)
    assert [e.label for e in seq.rf] == ["Mz→Mxy", "store", "recall"] and seq.timing is TIMING
    assert seq.refocusing_residual < 1e-2
    np.testing.assert_allclose(seq.b(), [d.b_value], rtol=1e-6)


def test_a_te_too_short_for_the_mixing_time_is_refused():
    with pytest.raises(ValueError, match="no room"):
        design_stimulated_echo(1.0, limits=LIM, TM=50e-3, TE=0.055, timing=TIMING, n_t=80, n_restarts=1)
