"""
Acceptance tests for the tensor-valued gradient-waveform designer.

Three tiers (see the module docstring of ``waveform_designer``):

* Tier 1 — validity: the generated waveform respects the b-tensor shape,
  refocusing (q(TE)=0, the finite-180), and the Prisma hardware box
  (amplitude, slew, gradient-off-during-RF).
* Tier 2 — efficiency: the solver finds a *good* optimum — LTE recovers the
  analytic max-b (triangular q), and the STE/PTE efficiency ratios are sane.
* Tier 3 — physical truth (the arbiter): the waveform, run through the dmipy-sim
  Monte Carlo engine, reproduces the expected tensor-encoding signatures —
  exp(-bD) on free diffusion, and STE orientation-invariance vs LTE
  orientation-dependence on an anisotropic cylinder.  Self-consistent b-tensor
  math is necessary but not sufficient; this is the real check.

These tests design real waveforms (GPU optimization) and run MC, so they are
slow (~minutes).  The designs are built once per session and shared.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("jaxopt")

from dmipy_design.optimizers.waveform_designer import design_waveform, GAMMA

TE = 0.080
G_MAX = 0.08
S_MAX = 200.0
N_T = 140
N_RESTARTS = 32
N_OUTER = 12
D_ISO = 1.0e-9      # m²/s, free-diffusion test
B_TEST = 1.5e9      # s/m² (1500 s/mm²) — rescale target for clean MC


@pytest.fixture(scope="module")
def designs():
    """Design LTE / STE / PTE once (Prisma, TE=80 ms) and share across tests.

    Explicitly UNCONSTRAINED (M1/M2/Maxwell off): these tests validate the core
    b-tensor encoding, the analytic max-b recovery, and MC orientation
    invariance — all orthogonal to the robustness constraints, which are
    validated separately in the toggle tests.  (design_waveform defaults all
    three constraints ON.)"""
    kw = dict(G_max=G_MAX, slew_rate_max=S_MAX, TE=TE, n_t=N_T,
              n_restarts=N_RESTARTS, n_outer=N_OUTER,
              null_M1=False, null_M2=False, maxwell=False)
    return {
        'LTE': design_waveform(1.0, **kw),
        'STE': design_waveform(0.0, **kw),
        'PTE': design_waveform(-0.5, **kw),
    }


# ---------------------------------------------------------------------------
# Tier 1 — validity gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,target", [('LTE', 1.0), ('STE', 0.0), ('PTE', -0.5)])
def test_tier1_validity(designs, name, target):
    d = designs[name]
    assert d.feasible, f"{name}: no feasible restart ({d.report})"
    assert abs(d.b_delta - target) < 2e-2, f"{name}: b_delta={d.b_delta} != {target}"
    assert d.max_amplitude <= G_MAX * 1.01, f"{name}: |G| exceeds G_max"
    assert d.max_slew <= S_MAX * 1.02, f"{name}: slew exceeds S_max"
    assert d.refocus_residual < 1e-2, f"{name}: not refocused (q(TE)!=0)"
    assert d.rf_window_leak <= G_MAX * 0.02, f"{name}: gradient leaks into RF window"


# ---------------------------------------------------------------------------
# Tier 2 — efficiency
# ---------------------------------------------------------------------------

def test_tier2_lte_recovers_max_b(designs):
    """LTE must approach the analytic max-b: a full-amplitude bang-bang gives a
    triangular q with b = (gamma·G_max·TE/2)²·TE/3.  This is the calibration
    anchor — if the solver can't reach the closed-form optimum it is broken."""
    q_max = GAMMA * G_MAX * (TE / 2.0)
    b_tri = q_max ** 2 * TE / 3.0
    ratio = designs['LTE'].b_value / b_tri
    assert 0.80 < ratio < 1.10, (
        f"LTE b={designs['LTE'].b_value/1e6:.0f} vs analytic max "
        f"{b_tri/1e6:.0f} s/mm² (ratio {ratio:.2f})")


def test_tier2_shape_efficiency_ratios(designs):
    """STE/PTE are less efficient than LTE but not pathologically so."""
    lte, ste, pte = (designs[k].b_value for k in ('LTE', 'STE', 'PTE'))
    assert 0.10 < ste / lte < 0.6, f"STE/LTE ratio {ste/lte:.3f} out of range"
    assert ste < pte < lte, "expected b_STE < b_PTE < b_LTE"
    assert designs['STE'].report['n_feasible'] >= 1


# ---------------------------------------------------------------------------
# Tier 3 — physical truth via dmipy-sim Monte Carlo (the arbiter)
# ---------------------------------------------------------------------------

def _simulate(wf, geometry, n=40000, seed=1):
    from dmipy_sim import simulate
    return float(np.asarray(simulate(n, D_ISO, wf, geometry, seed=seed))[0])


def test_tier3_free_diffusion_exp_bD(designs):
    """On free isotropic diffusion every shape gives S = exp(-bD); the b read
    back by dmipy-sim must equal the design b (cross-engine agreement)."""
    from dmipy_sim import FreeDiffusion
    from dmipy_sim.sequences._helpers import _calc_b_from_waveform
    for name in ('STE', 'LTE'):
        wf = designs[name].to_sim_waveform(b_target=B_TEST)
        b_sim = float(_calc_b_from_waveform(np.asarray(wf.G), wf.dt)[0])
        assert abs(b_sim - B_TEST) / B_TEST < 0.02, (
            f"{name}: dmipy-sim b={b_sim/1e6:.0f} != target 1500 s/mm²")
        S = _simulate(wf, FreeDiffusion())
        assert abs(S - np.exp(-b_sim * D_ISO)) < 0.02, (
            f"{name}: S_MC={S:.4f} != exp(-bD)={np.exp(-b_sim*D_ISO):.4f}")


def test_tier3_ste_orientation_invariance(designs):
    """THE headline test: on an anisotropic cylinder the STE signal is
    orientation-invariant (isotropic encoding) while LTE varies strongly."""
    from dmipy_sim import Cylinder
    from dmipy_sim.waveforms import Waveform
    cyl = Cylinder(radius=3e-6, orientation=[0, 0, 1.0])

    def _rot(theta, phi):
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        Rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1.0]])
        Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        return Rz @ Ry

    frames = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2),
              (np.pi / 4, 0), (np.pi / 3, np.pi / 4)]
    cv = {}
    for name in ('STE', 'LTE'):
        G0 = np.asarray(designs[name].to_sim_waveform(b_target=B_TEST).G)[0]  # (n_t,3)
        dt = designs[name].dt
        echo = designs[name].G.shape[0] - 1
        sigs = []
        for th, ph in frames:
            Grot = (G0 @ _rot(th, ph).T)[None].astype(np.float32)
            sigs.append(_simulate(Waveform(G=Grot, dt=dt, echo_idx=echo), cyl))
        sigs = np.array(sigs)
        cv[name] = float(sigs.std() / sigs.mean())

    assert cv['STE'] < 0.05, f"STE not orientation-invariant: CV={cv['STE']:.3f}"
    assert cv['LTE'] > 0.10, f"LTE should be orientation-dependent: CV={cv['LTE']:.3f}"
    assert cv['STE'] < cv['LTE'] / 3.0, (
        f"STE CV ({cv['STE']:.3f}) must be far below LTE CV ({cv['LTE']:.3f})")


# ---------------------------------------------------------------------------
# Constraint toggling — M1/M2 moment nulling + Maxwell (concomitant) compensation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toggle_designs():
    """Baseline vs fully-constrained LTE on an ASYMMETRIC echo (frac=0.4), where
    M1 / Maxwell are naturally non-zero so the constraints have visible work to do."""
    kw = dict(b_delta=1.0, G_max=G_MAX, slew_rate_max=S_MAX, TE=TE, n_t=100,
              echo_frac=0.40, n_restarts=16, n_outer=16)
    return {
        'base': design_waveform(**kw, null_M1=False, null_M2=False, maxwell=False),
        'all': design_waveform(**kw, null_M1=True, null_M2=True, maxwell=True),
    }


def test_constraint_flags_select_active_set():
    """Flags toggle exactly which constraints enter the augmented Lagrangian, and
    all three default ON."""
    tiny = dict(TE=TE, n_t=64, n_restarts=4, n_outer=3, inner_maxiter=40)
    # default: all three robustness constraints active
    d_default = design_waveform(0.0, **tiny)
    assert {'M1', 'M2', 'maxwell'} <= set(d_default.active_constraints)
    # opt out of M2 only -> M1 + maxwell active, M2 not
    d = design_waveform(0.0, **tiny, null_M2=False)
    assert 'M1' in d.active_constraints and 'maxwell' in d.active_constraints
    assert 'M2' not in d.active_constraints
    # all indices reported regardless of which are constrained
    assert d.m1_index is not None and d.m2_index is not None and d.maxwell_index is not None


def test_moment_and_maxwell_nulling(toggle_designs):
    """Activating M1/M2/Maxwell drives each index toward zero — and costs b."""
    base, allc = toggle_designs['base'], toggle_designs['all']
    # asymmetric baseline is genuinely uncompensated (sanity: there is work to do)
    assert base.m1_index > 0.10, f"baseline M1 should be non-zero: {base.m1_index}"
    assert base.maxwell_index > 0.10, f"baseline Maxwell should be non-zero: {base.maxwell_index}"
    # the constrained design clearly nulls all three indices.  The relative
    # <base/2.5 is the substantive claim (a >2.5x reduction from the uncompensated
    # baseline); the absolute bound guards against a near-zero baseline.  (Tighter
    # nulling is achievable with more restarts/outer rounds; this is a fast test.)
    assert allc.m1_index < 0.08 and allc.m1_index < base.m1_index / 2.5
    assert allc.m2_index < 0.08 and allc.m2_index < base.m2_index / 2.5
    assert allc.maxwell_index < 0.08 and allc.maxwell_index < base.maxwell_index / 2.5
    # compensation is not free: b decreases relative to the unconstrained optimum
    assert allc.b_value < base.b_value, "expected a b-cost for the added constraints"


# ---------------------------------------------------------------------------
# SequenceTiming — encoding windows DERIVED from a real timing budget
# ---------------------------------------------------------------------------
from dmipy_design.optimizers import SequenceTiming


def test_sequence_timing_windows_are_asymmetric():
    """A real timing budget pins the encoding windows; an unequal lead-in vs
    readout-pre-echo makes the pre/post-180 windows asymmetric — a derived
    consequence, not a knob — with the gradient masked off in lead-in/180/readout."""
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3,
                                     readout_duration=30e-3, partial_fourier=0.75)
    assert abs(st.t_readout_pre_echo - 10e-3) < 1e-9          # 30ms·(0.25/0.75)
    TE, n_t = 0.080, 400
    mask, echo = st.masks(TE, n_t)
    on = mask[:, 0]
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    assert echo == round((TE / 2) / dt)                       # 180 at TE/2
    assert on[t < st.t_lead].sum() == 0                       # excitation lead-in off
    assert on[np.abs(t - TE / 2) <= st.t_refocus / 2].sum() == 0   # 180 off
    assert on[t > TE - st.t_readout_pre_echo].sum() == 0      # readout tail off
    pre, post = on[t < TE / 2].sum(), on[t > TE / 2].sum()
    assert pre > post * 1.1                                   # asymmetric, derived


def test_sequence_timing_symmetric_is_the_vanilla_waveform():
    """The VANILLA waveform: symmetric=True mirrors the encoding windows about the
    180 (equal pre/post durations) and dead-times the surplus of the longer window.
    Same TE and 180 position, but less total encoding -> the cost of refusing the
    budget's natural asymmetry (extra transverse time = T2 loss)."""
    b = dict(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=10e-3)
    TE, n_t = 0.050, 300
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    on_a, e_a = SequenceTiming(**b).masks(TE, n_t)
    on_s, e_s = SequenceTiming(**b, symmetric=True).masks(TE, n_t)
    assert e_a == e_s == round((TE / 2) / dt)                 # 180 stays at TE/2
    pre_a, post_a = on_a[t < TE / 2, 0].sum(), on_a[t > TE / 2, 0].sum()
    pre_s, post_s = on_s[t < TE / 2, 0].sum(), on_s[t > TE / 2, 0].sum()
    assert pre_a > post_a * 1.1                               # default is asymmetric
    assert pre_s == post_s                                    # vanilla is symmetric
    assert post_s == post_a                                   # both equal the shorter window
    assert on_s.sum() < on_a.sum()                            # vanilla encodes less (dead time)


def test_sequence_timing_min_te_guard():
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3,
                                     readout_duration=30e-3, partial_fourier=0.75)
    with pytest.raises(ValueError):                           # below min_TE windows vanish
        st.masks(st.min_TE() - 1e-3, 200)


def test_min_te_for_b_returns_smallest_feasible_te():
    """min_te_for_b bisects the max-b primitive and returns a FEASIBLE design that
    reaches the target b at a TE inside the search bracket (the minimum such TE).
    Uses the same restart/outer budget as the validated timing test so the designs
    are genuinely feasible (too-coarse settings give huge-b-but-infeasible designs,
    which the helper correctly rejects)."""
    from dmipy_design.optimizers import min_te_for_b
    kw = dict(G_max=G_MAX, slew_rate_max=S_MAX, n_t=150, n_restarts=16, n_outer=10,
              null_M1=False, null_M2=False, maxwell=False)
    b_target = 3.0e9                                          # 3000 s/mm^2 (well below max)
    d, te = min_te_for_b(b_target, 1.0, te_lo=0.038, te_hi=0.080, tol_te=7e-3, **kw)
    assert d.feasible and d.b_value >= b_target               # reaches the target, feasibly
    assert 0.038 <= te <= 0.080                               # min TE within the bracket


def test_design_with_timing_respects_windows():
    """design_waveform(timing=...) optimizes only within the derived encoding
    windows: feasible, and the gradient is ~0 in the lead-in / 180 / readout
    off-regions (no hand-set echo_frac)."""
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3,
                                     readout_duration=24e-3, partial_fourier=0.75)
    TE = 0.070
    d = design_waveform(0.0, G_max=G_MAX, slew_rate_max=S_MAX, TE=TE, n_t=160,
                        timing=st, null_M1=False, null_M2=False, maxwell=False,
                        n_restarts=16, n_outer=10)
    assert d.feasible, f"timing-driven design infeasible ({d.report})"
    t = np.arange(d.G.shape[0]) * d.dt
    gnorm = np.linalg.norm(d.G, axis=1)
    off = ((t < st.t_lead) | (np.abs(t - TE / 2) <= st.t_refocus / 2)
           | (t > TE - st.t_readout_pre_echo))
    assert gnorm[off].max() <= G_MAX * 0.02, "gradient leaks into an off-region"


def test_offcenter_180_is_guarded():
    """An off-centre 180 (misaligned with the readout — static field would refocus
    at 2·t_180, not TE) must RAISE unless explicitly opted into.  TE/2 is silent."""
    from dmipy_sim.sequences import Sequence
    n_t, dt = 200, 3e-4
    eff = np.zeros((n_t, 3), dtype=np.float32)
    eff[10:60, 0] = 0.03
    eff[110:170, 0] = -0.03
    i_off = int(round(0.35 * (n_t - 1)))
    with pytest.raises(ValueError, match="off-centre|TE/2|misaligned"):
        Sequence.from_btensor_waveform(eff, dt, echo_idx=i_off)
    # deliberate opt-in is allowed; TE/2 (default + explicit) is fine
    Sequence.from_btensor_waveform(eff, dt, echo_idx=i_off, allow_offcenter_180=True)
    Sequence.from_btensor_waveform(eff, dt)
    Sequence.from_btensor_waveform(eff, dt, echo_idx=n_t // 2)


# ---------------------------------------------------------------------------
# Spectral paradigm — encoding spectrum reporting + OGSE frequency targeting
# ---------------------------------------------------------------------------

def test_spectral_content_always_reported():
    """Every design reports its encoding spectrum (rms/centroid/bandwidth); the
    target is None when no spectral_freq is requested."""
    d = design_waveform(1.0, TE=TE, n_t=100, n_restarts=6, n_outer=5, inner_maxiter=60,
                        null_M1=False, null_M2=False, maxwell=False)
    for v in (d.spectral_rms_hz, d.spectral_centroid_hz, d.spectral_bandwidth_hz):
        assert v is not None and v >= 0
    assert d.spectral_target_hz is None


def test_spectral_freq_produces_ogse():
    """spectral_freq drives the RMS encoding frequency to the target (an OGSE-like
    oscillating waveform) vs a low-frequency PGSE-like baseline; oscillation costs
    b-efficiency, and the realized spectrum (incl. bandwidth) is reported."""
    kw = dict(b_delta=1.0, G_max=G_MAX, slew_rate_max=S_MAX, TE=TE, n_t=160,
              n_restarts=12, n_outer=12, null_M1=False, null_M2=False, maxwell=False)
    base = design_waveform(**kw)                            # no spectral -> PGSE-like
    ogse = design_waveform(**kw, spectral_freq=80.0)        # OGSE at 80 Hz
    assert base.spectral_rms_hz < 20.0, f"baseline not low-freq: {base.spectral_rms_hz}"
    assert abs(ogse.spectral_rms_hz - 80.0) / 80.0 < 0.15, (
        f"f_rms did not track target: {ogse.spectral_rms_hz}")
    assert ogse.spectral_rms_hz > 3 * base.spectral_rms_hz
    assert ogse.b_value < base.b_value, "oscillation should cost b-efficiency"
    assert ogse.spectral_bandwidth_hz > 0  # finite-duration spectral spread, reported
