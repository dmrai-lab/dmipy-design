"""Tests for dmipy_design.mc_bridge — MC forward function for FIM computation.

All tests use small n_walkers (2000–5000) to stay under 60 s on CPU.
dmipy_sim is imported via pytest.importorskip so the whole module is
gracefully skipped if dmipy_sim is not installed.
"""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
import pytest

dmipy_sim = pytest.importorskip("dmipy_sim")

from dmipy_sim.geometries import Cylinder, FreeDiffusion  # noqa: E402
from dmipy_design.mc_bridge import (  # noqa: E402
    build_mc_forward_fn,
    build_mc_forward_fn_fixed,
    _scheme_to_sim_waveform,
)
from dmipy_design.jax_scheme_encoder import encode_pgse  # noqa: E402
from dmipy_design.fim import compute_fim_fd  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_DIRS = 10  # number of gradient directions per test
N_WALKERS_FAST = 2000  # fast: ~1 s per simulation on CPU
N_WALKERS_MEDIUM = 3000

DELTA = 0.013   # s
DELTA_CAP = 0.022   # s  (Delta in PGSE notation)
B_VALUE = 1000e6  # s/m²  (= 1000 s/mm²)
RADIUS = 3e-6   # m (3 µm cylinder)
D = 1.7e-9       # m²/s


def _make_perp_scheme(n_dirs=N_DIRS):
    """PGSE scheme with gradient perpendicular to cylinder axis (x-direction)."""
    bvecs = np.tile([1., 0., 0.], (n_dirs, 1))
    u = jnp.array([B_VALUE, DELTA, DELTA_CAP])
    return encode_pgse(u, jnp.array(bvecs, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# 1. Shape test
# ---------------------------------------------------------------------------

def test_mc_forward_fn_returns_correct_shape():
    """forward_fn(scheme) returns array of shape (N_meas,) matching scheme."""
    geom = Cylinder(radius=RADIUS, orientation=[0, 0, 1])
    mc_fwd = build_mc_forward_fn_fixed(geom, diffusivity=D,
                                        n_walkers=N_WALKERS_FAST, seed=0)
    scheme = _make_perp_scheme(n_dirs=N_DIRS)
    signals = mc_fwd(scheme)

    assert signals.shape == (N_DIRS,), (
        f"Expected shape ({N_DIRS},), got {signals.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Signal range test
# ---------------------------------------------------------------------------

def test_mc_signal_in_range():
    """All signal values in [0, 1] for a cylinder at b = 1000 s/mm²."""
    geom = Cylinder(radius=RADIUS, orientation=[0, 0, 1])
    mc_fwd = build_mc_forward_fn_fixed(geom, diffusivity=D,
                                        n_walkers=N_WALKERS_FAST, seed=1)
    scheme = _make_perp_scheme()
    signals = mc_fwd(scheme)

    assert np.all(signals >= 0.0), (
        f"Negative signals found: min={signals.min():.4f}"
    )
    assert np.all(signals <= 1.0 + 1e-5), (
        f"Signals > 1 found: max={signals.max():.4f}"
    )


# ---------------------------------------------------------------------------
# 3. FIM positive semi-definite test
# ---------------------------------------------------------------------------

def test_mc_fim_fd_is_psd():
    """compute_fim_fd with MC forward fn returns positive semi-definite FIM."""
    geom = Cylinder(radius=RADIUS, orientation=[0, 0, 1])
    mc_fwd = build_mc_forward_fn_fixed(geom, diffusivity=D,
                                        n_walkers=N_WALKERS_MEDIUM, seed=2)

    scheme = _make_perp_scheme()
    # theta is empty — we compute FIM over scheme space with fixed geometry.
    # compute_fim_fd expects forward_fn(theta, scheme), so wrap mc_fwd.
    wrapped = lambda theta, scheme: mc_fwd(scheme)
    theta = jnp.zeros(0)

    FIM = compute_fim_fd(wrapped, theta, scheme, sigma=0.05)
    # FIM should be (0, 0) for empty theta
    assert FIM.shape == (0, 0), (
        f"Expected shape (0, 0) for empty theta, got {FIM.shape}"
    )

    # --- Non-trivial FIM: use factory geometry to vary diffusivity proxy ---
    # Build a 1-parameter forward fn: theta[0] scales the signal amplitude
    # (not physically meaningful, but exercises the FD pathway end-to-end).
    def scaled_mc_fwd(theta, scheme):
        signals = mc_fwd(scheme)
        return signals * float(theta[0])  # theta[0] is a scale factor

    theta_1d = jnp.array([1.0])
    FIM_1d = compute_fim_fd(scaled_mc_fwd, theta_1d, scheme, sigma=0.05)

    assert FIM_1d.shape == (1, 1), f"Expected (1,1), got {FIM_1d.shape}"
    eigvals = np.linalg.eigvalsh(np.array(FIM_1d))
    assert np.all(eigvals >= -1e-10), (
        f"FIM has negative eigenvalues: {eigvals}"
    )


# ---------------------------------------------------------------------------
# 4. Fixed vs factory geometry agreement
# ---------------------------------------------------------------------------

def test_fixed_vs_factory_geometry_agree():
    """Fixed and factory variants produce similar signals (within MC noise).

    Uses n_walkers=2000 and atol=0.05 (5% tolerance) to account for MC noise.
    Both variants use the same seed so walker paths are identical, resulting
    in tighter agreement than the 5% tolerance.
    """
    radius = RADIUS
    geom_fixed = Cylinder(radius=radius, orientation=[0, 0, 1])
    geom_factory = lambda p: Cylinder(radius=p["radius"], orientation=[0, 0, 1])

    mc_fixed = build_mc_forward_fn_fixed(
        geom_fixed, diffusivity=D, n_walkers=N_WALKERS_FAST, seed=7
    )
    mc_factory = build_mc_forward_fn(
        geom_factory, diffusivity=D, n_walkers=N_WALKERS_FAST, seed=7
    )

    scheme = _make_perp_scheme()
    sig_fixed = mc_fixed(scheme)
    sig_factory = mc_factory({"radius": radius}, scheme)

    npt.assert_allclose(
        sig_fixed, sig_factory, atol=0.05,
        err_msg="Fixed and factory variants should agree within MC noise (5%)"
    )


# ---------------------------------------------------------------------------
# 5. Scheme-to-waveform conversion sanity check
# ---------------------------------------------------------------------------

def test_scheme_to_sim_waveform_shape():
    """_scheme_to_sim_waveform returns a Waveform with correct G shape."""
    from dmipy_sim.waveforms import Waveform

    scheme = _make_perp_scheme(n_dirs=5)
    wf = _scheme_to_sim_waveform(scheme, n_t=200)

    assert isinstance(wf, Waveform)
    n_meas, n_t, n_xyz = wf.G.shape
    assert n_meas == 5,  f"Expected n_meas=5, got {n_meas}"
    assert n_t   == 200, f"Expected n_t=200,  got {n_t}"
    assert n_xyz == 3,   f"Expected n_xyz=3,  got {n_xyz}"


def test_scheme_to_sim_waveform_b_values():
    """Waveform produced by _scheme_to_sim_waveform has correct b-values."""
    from dmipy_sim.waveforms import calc_b

    n_dirs = 8
    bvecs = np.tile([1., 0., 0.], (n_dirs, 1))
    u = jnp.array([B_VALUE, DELTA, DELTA_CAP])
    scheme = encode_pgse(u, jnp.array(bvecs, dtype=jnp.float32))
    wf = _scheme_to_sim_waveform(scheme, n_t=500)

    b_actual = calc_b(wf)
    b_expected = np.full(n_dirs, B_VALUE)

    npt.assert_allclose(
        b_actual, b_expected, rtol=0.02,
        err_msg="b-values from waveform should match scheme b-values within 2%"
    )
