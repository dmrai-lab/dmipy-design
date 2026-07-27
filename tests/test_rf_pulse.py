"""Deliverable, B1-robust 180° refocusing-RF design (Bloch forward)."""
import numpy as np
import pytest

from dmipy_design.optimizers import design_refocusing_rf, RfPulseDesign

GAMMA = 2.675e8


def test_design_is_deliverable_and_well_formed():
    d = design_refocusing_rf(rf_duration=6e-3, dt=2e-4, B1_max=20e-6,
                             n_b1=5, n_off_resonance=5, n_restarts=3, seed=0)
    assert isinstance(d, RfPulseDesign)
    n = int(round(6e-3 / 2e-4))
    assert d.B1.shape == (n,) and d.flip_angles.shape == (n,)
    assert np.all(np.isfinite(d.B1))
    # nominal on-resonance pulse is a 180° (|flips| sum to π)
    assert np.sum(np.abs(d.flip_angles)) == pytest.approx(np.pi, rel=1e-6)
    # refocused fraction is a valid coherence in [0, 1]
    assert 0.0 <= d.refocused_fraction <= 1.0
    # deliverability box respected (within the 2 % feasibility tolerance)
    assert d.feasible
    assert d.peak_B1 <= d.B1_max * 1.02
    assert d.sar_proxy <= d.sar_budget * 1.02
    # B1 envelope is consistent with the reported flip angles
    np.testing.assert_allclose(d.B1, d.flip_angles / (GAMMA * d.dt), rtol=1e-6)


def test_not_worse_than_hard_180_baseline():
    """Optimising over the transmit/off-resonance ensemble should not lose to a hard 180°."""
    d = design_refocusing_rf(rf_duration=6e-3, dt=2e-4, B1_max=20e-6,
                             n_b1=5, n_off_resonance=5, n_restarts=4, seed=0)
    assert d.refocused_fraction >= d.refocused_fraction_hard - 1e-3


def test_tighter_peak_b1_lowers_peak():
    """A tighter peak-B1 limit must pull the delivered peak down."""
    loose = design_refocusing_rf(rf_duration=6e-3, dt=2e-4, B1_max=40e-6,
                                 n_b1=5, n_off_resonance=5, n_restarts=3, seed=0)
    tight = design_refocusing_rf(rf_duration=6e-3, dt=2e-4, B1_max=8e-6,
                                 n_b1=5, n_off_resonance=5, n_restarts=3, seed=0)
    assert tight.peak_B1 <= loose.peak_B1 * 1.02
    assert tight.peak_B1 <= tight.B1_max * 1.02


def test_deterministic_for_fixed_seed():
    kw = dict(rf_duration=6e-3, dt=2e-4, B1_max=20e-6, n_b1=5,
              n_off_resonance=5, n_restarts=3, seed=1)
    a = design_refocusing_rf(**kw)
    b = design_refocusing_rf(**kw)
    np.testing.assert_allclose(a.B1, b.B1, rtol=1e-9, atol=0)
    assert a.refocused_fraction == pytest.approx(b.refocused_fraction, rel=1e-9)


def test_times_axis_is_centred():
    d = design_refocusing_rf(rf_duration=4e-3, dt=2e-4, n_b1=3, n_off_resonance=3,
                             n_restarts=1, seed=0)
    t = d.times()
    assert t.shape == d.B1.shape
    assert t.mean() == pytest.approx(0.0, abs=1e-12)
