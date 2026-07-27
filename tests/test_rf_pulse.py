"""B1-robust adiabatic (hyperbolic-secant) refocusing-RF design."""
import numpy as np
import pytest

from dmipy_design.optimizers import design_refocusing_rf, RfPulseDesign
from dmipy_design.optimizers.rf_pulse import _inversion_mz, GAMMA


def _design(**kw):
    base = dict(rf_duration=6e-3, dt=1e-4, B1_max=19e-6, n_b1=5, n_off_resonance=5, n_mu=10,
                refine=False)               # HS-only for speed; refinement has its own test
    base.update(kw)
    return design_refocusing_rf(**base)


def test_design_is_well_formed_and_deliverable():
    d = _design()
    assert isinstance(d, RfPulseDesign)
    assert d.B1.shape == (int(round(6e-3 / 1e-4)),) and np.iscomplexobj(d.B1)
    assert np.all(np.isfinite(d.B1))
    assert d.peak_B1 <= d.B1_max * 1.02          # hard peak-B1 limit
    assert d.bandwidth_hz > 2 * 250.0            # sweep covers the ±250 Hz off-res band
    assert d.mu > 0 and d.beta > 0
    assert d.feasible


def test_adiabatic_pulse_inverts_across_b1_range():
    """The HS passage should invert (M_z→−1) across the whole B1⁺ range — the robustness claim,
    checked per-spin — where a hard 180° only inverts at the nominal point."""
    d = _design(n_b1=7, n_off_resonance=7, n_mu=14)
    b1_probe = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    zero = np.zeros_like(b1_probe)
    mz_des = _inversion_mz(d.B1, b1_probe, zero, d.dt)
    A0h = np.pi / (GAMMA * d.B1.shape[0] * d.dt)
    mz_hard = _inversion_mz(np.full(d.B1.shape[0], A0h, complex), b1_probe, zero, d.dt)
    assert np.all(mz_des < -0.8)                 # every probed spin genuinely inverts
    assert mz_des.mean() < mz_hard.mean()
    assert d.refocusing_efficiency > 0.9
    assert d.refocusing_efficiency > d.refocusing_efficiency_hard + 0.2


def test_smooth_monotone_passage():
    """Adiabatic following ⇒ a smooth spiral: M_z descends near-monotonically +1→−1, unlike a
    flailing pulse. Check the nominal spin's M_z(t) makes few upward excursions."""
    d = _design(n_mu=12)
    n = d.B1.shape[0]
    # step-by-step M_z for the nominal spin
    mz = [1.0]
    Mx = My = 0.0; Mz = 1.0
    for k in range(n):
        nx = GAMMA * d.B1[k].real * d.dt; ny = GAMMA * d.B1[k].imag * d.dt
        th = np.hypot(nx, ny)
        if th > 1e-30:
            kx, ky = nx / th, ny / th
            c, s = np.cos(th), np.sin(th)
            kdotM = kx * Mx + ky * My
            Mx, My, Mz = (Mx * c + (ky * Mz) * s + kx * kdotM * (1 - c),
                          My * c + (-kx * Mz) * s + ky * kdotM * (1 - c),
                          Mz * c + (kx * My - ky * Mx) * s)
        mz.append(Mz)
    mz = np.array(mz)
    ups = np.sum(np.diff(mz) > 1e-3)             # upward steps
    assert mz[-1] < -0.8                          # ends inverted
    assert ups < 0.25 * n                         # mostly descending — not flailing


def test_grape_refinement_improves_on_hs():
    """The optimal-control refinement, warm-started from HS, should not lose to the template
    and should keep the pulse deliverable (peak within limit)."""
    kw = dict(rf_duration=6e-3, dt=1e-4, B1_max=19e-6, n_b1=7, n_off_resonance=7, n_mu=14)
    hs = design_refocusing_rf(refine=False, **kw)
    ref = design_refocusing_rf(refine=True, n_refine_basis=8, **kw)
    assert ref.refined                                    # refinement was kept
    assert ref.refocusing_efficiency >= hs.refocusing_efficiency - 1e-6
    assert ref.peak_B1 <= ref.B1_max * 1.02               # still deliverable
    assert ref.refocusing_efficiency > 0.9


def test_sar_budget_lowers_amplitude():
    loose = _design()                             # peak-limited
    capped = _design(sar_headroom=8.0)
    assert capped.peak_B1 <= loose.peak_B1 + 1e-9
    assert capped.sar_ratio <= 8.0 * 1.02


def test_warm_start_from_array_is_refined():
    """A caller-supplied warm start (here a crude flat hard-180 array) is used and refined."""
    n = int(round(6e-3 / 2e-4))
    A = np.pi / (GAMMA * n * 2e-4)
    ws = np.full(n, A, dtype=complex)
    d = design_refocusing_rf(rf_duration=6e-3, dt=2e-4, B1_max=20e-6, n_b1=5, n_off_resonance=5,
                             warm_start=ws)
    assert np.isnan(d.mu)                                  # no HS backbone when warm-started
    assert d.B1.shape == (n,)
    assert d.refocusing_efficiency > d.refocusing_efficiency_hard   # refinement improved it


def test_deterministic():
    a = _design(); b = _design()
    np.testing.assert_allclose(a.B1, b.B1, rtol=1e-9, atol=0)
    assert a.mu == pytest.approx(b.mu, rel=1e-9)


def test_times_axis_is_centred():
    d = _design(rf_duration=4e-3)
    t = d.times()
    assert t.shape == d.B1.shape and t.mean() == pytest.approx(0.0, abs=1e-12)


# ── dmipy-sim bridge (needs the [sim] extra with B1Pulse) ─────────────────────
_HAS_B1PULSE = False
_HAS_BIR4 = False
try:
    from dmipy_sim.rf import B1Pulse, bloch_simulate  # noqa: F401
    _HAS_B1PULSE = True
    _HAS_BIR4 = hasattr(B1Pulse, "bir4")
except Exception:
    pass


@pytest.mark.skipif(not _HAS_BIR4, reason="dmipy-sim with the BIR-4 constructor not installed")
def test_warm_start_from_sim_bir4():
    """design can pull a dmipy-sim constructor (BIR-4) as the initial guess and refine it."""
    ws = B1Pulse.bir4(180, 6e-3, 1e-4, peak_b1=19e-6)
    d = design_refocusing_rf(rf_duration=6e-3, dt=1e-4, B1_max=19e-6, n_b1=7, n_off_resonance=7,
                             warm_start=ws)
    assert d.refocusing_efficiency > 0.9                   # already-robust warm start, refined


@pytest.mark.skipif(not _HAS_B1PULSE, reason="dmipy-sim with B1Pulse not installed")
def test_to_b1pulse_round_trips_into_sim():
    d = _design(n_b1=7, n_off_resonance=7, n_mu=14)
    p = d.to_b1pulse()
    np.testing.assert_allclose(p.b1, d.B1, rtol=0, atol=0)      # exact complex envelope
    assert p.dt == pytest.approx(d.dt, rel=1e-12)
    for b1 in (0.7, 1.0, 1.3):                                  # inverts across B1⁺ in sim
        _, Mz = bloch_simulate(p, df_hz=0.0, b1_scale=b1)
        assert float(Mz[0]) < -0.8
