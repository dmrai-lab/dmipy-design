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
    """Design LTE / STE / PTE once (Prisma, TE=80 ms) and share across tests."""
    kw = dict(G_max=G_MAX, slew_rate_max=S_MAX, TE=TE, n_t=N_T,
              n_restarts=N_RESTARTS, n_outer=N_OUTER)
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
