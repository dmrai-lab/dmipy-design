"""PGSTE — stimulated-echo timing (matched periods) and design via the NOW core."""
import numpy as np
import pytest

from dmipy_design.optimizers import (
    StimulatedEchoTiming, design_stimulated_echo, NowDesign)


def test_matched_encoding_periods_and_masks():
    st = StimulatedEchoTiming(t_excite=2e-3, TM=50e-3)
    TE = st.min_TE() + 40e-3
    assert st.tau(TE) > 0
    on, flip = st.masks(TE, 200)
    on = on[:, 0]
    assert on.sum() > 0
    assert 0 < flip < 200                             # sign flip sits in the dead middle
    # two encoding windows (τ1, τ3), and the effective diffusion time is dominated by TM
    assert st.effective_diffusion_time(TE) > st.TM
    # τ1 and τ3 windows are equal-length by construction (up to grid rounding)
    dt = TE / 199
    n_on = int(round(st.tau(TE) / dt))
    assert abs(on.sum() - 2 * n_on) <= 2


def test_min_te_guard():
    st = StimulatedEchoTiming(t_excite=2e-3, TM=50e-3)
    with pytest.raises(ValueError):
        st.masks(st.min_TE() - 1e-3, 200)


def test_design_stimulated_echo_is_feasible():
    d = design_stimulated_echo(1.0, TM=50e-3, TE=0.12, n_t=80, n_restarts=3, seed=0)
    assert isinstance(d, NowDesign)
    assert d.feasible
    assert d.b_value > 0
    assert d.refocus_residual < 1e-2
