"""min_te_for_b — the SNR-optimal inverse of max-b-at-fixed-TE."""
import pytest

from dmipy_design.optimizers import min_te_for_b, design_waveform_now, SequenceTiming

TIMING = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)


def test_min_te_reaches_target_b_and_is_tight():
    b_target = 4e8                                  # s/m² (400 s/mm²)
    d, te = min_te_for_b(b_target, b_delta=1.0, timing=TIMING,
                         n_t=64, n_restarts=3, tol_te=2e-3)
    assert d.feasible
    assert d.b_value >= b_target                    # reaches the target
    assert te >= TIMING.min_TE()                    # respects the timing floor
    # tightness: just below the returned TE the target is no longer reached
    lo = te - 4e-3
    if lo > TIMING.min_TE():
        d_lo = design_waveform_now(1.0, TE=lo, timing=TIMING, n_t=64, n_restarts=3, seed=0)
        assert d_lo.b_value < b_target


def test_shorter_target_gives_shorter_te():
    """A smaller required b is reachable at a shorter TE (monotonic b–TE relation)."""
    _, te_small = min_te_for_b(2e8, b_delta=1.0, timing=TIMING, n_t=64, n_restarts=3, tol_te=2e-3)
    _, te_big = min_te_for_b(6e8, b_delta=1.0, timing=TIMING, n_t=64, n_restarts=3, tol_te=2e-3)
    assert te_small < te_big


def test_unreachable_target_raises():
    with pytest.raises(ValueError):
        min_te_for_b(1e12, b_delta=1.0, timing=TIMING, n_t=48, n_restarts=2,
                     te_max=0.06, tol_te=3e-3)
