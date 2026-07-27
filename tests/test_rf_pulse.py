"""Deliverable, B1-robust 180° refocusing-RF design (Bloch refocusing-fidelity objective)."""
import numpy as np
import pytest

from dmipy_design.optimizers import design_refocusing_rf, RfPulseDesign
from dmipy_design.optimizers.rf_pulse import _inversion_mz

GAMMA = 2.675e8


def _design(**kw):
    base = dict(rf_duration=6e-3, dt=2e-4, B1_max=20e-6, n_b1=5, n_off_resonance=5,
                n_basis=8, n_restarts=4, seed=0)
    base.update(kw)
    return design_refocusing_rf(**base)


def test_design_is_well_formed_and_deliverable():
    d = _design()
    assert isinstance(d, RfPulseDesign)
    n = int(round(6e-3 / 2e-4))
    assert d.B1.shape == (n,) and np.iscomplexobj(d.B1)
    assert np.all(np.isfinite(d.B1))
    assert 0.0 <= d.refocusing_efficiency <= 1.0
    assert d.peak_B1 <= d.B1_max * 1.02        # hard peak-B1 limit respected
    assert d.feasible


def test_beats_hard_180_refocusing():
    """A genuinely robust 180° must refocus the ensemble better than a plain hard 180°."""
    d = _design(n_restarts=6)
    assert d.refocusing_efficiency > d.refocusing_efficiency_hard + 0.1


def test_peak_limited_design_inverts_across_b1_range():
    """Peak-limited (no SAR cap): the pulse should invert (M_z→−1) across the whole B1⁺ range,
    which a hard 180° cannot — the core robustness claim, checked per-spin (no coherence games)."""
    d = _design(B1_max=19e-6, n_b1=7, n_off_resonance=7, n_basis=10, n_restarts=8)
    b1_probe = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    zero = np.zeros_like(b1_probe)
    mz_des = _inversion_mz(d.B1, b1_probe, zero, d.dt)
    A0 = np.pi / (GAMMA * d.B1.shape[0] * d.dt)
    mz_hard = _inversion_mz(np.full(d.B1.shape[0], A0, complex), b1_probe, zero, d.dt)
    assert np.all(mz_des < -0.8)                       # every probed spin genuinely inverts
    assert mz_des.mean() < mz_hard.mean()              # and better than hard everywhere on average
    assert d.refocusing_efficiency > 0.9               # near-ideal when peak-limited


def test_tighter_peak_b1_lowers_peak():
    loose = _design(B1_max=40e-6)
    tight = _design(B1_max=8e-6)
    assert tight.peak_B1 <= loose.peak_B1 * 1.02
    assert tight.peak_B1 <= tight.B1_max * 1.02


def test_sar_budget_is_respected_when_set():
    d = _design(sar_headroom=3.0)
    assert np.isfinite(d.sar_budget)
    assert d.sar_proxy <= d.sar_budget * 1.02
    assert d.sar_ratio <= 3.0 * 1.02


def test_deterministic_for_fixed_seed():
    a = _design(seed=1)
    b = _design(seed=1)
    np.testing.assert_allclose(a.B1, b.B1, rtol=1e-9, atol=0)
    assert a.refocusing_efficiency == pytest.approx(b.refocusing_efficiency, rel=1e-9)


def test_times_axis_is_centred():
    d = _design(rf_duration=4e-3)
    t = d.times()
    assert t.shape == d.B1.shape
    assert t.mean() == pytest.approx(0.0, abs=1e-12)


# ── dmipy-sim bridge (needs the [sim] extra with B1Pulse) ─────────────────────
_HAS_B1PULSE = False
try:
    from dmipy_sim.rf import B1Pulse, bloch_simulate  # noqa: F401
    _HAS_B1PULSE = True
except Exception:
    pass


@pytest.mark.skipif(not _HAS_B1PULSE, reason="dmipy-sim with B1Pulse not installed")
def test_to_b1pulse_round_trips_into_sim():
    d = _design(B1_max=19e-6, n_b1=7, n_off_resonance=7, n_basis=10, n_restarts=6)
    p = d.to_b1pulse()
    np.testing.assert_allclose(p.b1, d.B1, rtol=0, atol=0)      # exact complex envelope
    assert p.dt == pytest.approx(d.dt, rel=1e-12)
    # the designed pulse inverts +z in the sim Bloch forward, across the B1⁺ range
    for b1 in (0.7, 1.0, 1.3):
        _, Mz = bloch_simulate(p, df_hz=0.0, b1_scale=b1)
        assert float(Mz[0]) < -0.8
