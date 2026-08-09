"""Waveform optimization through replay packs — the shape/size discrimination slice.

Uses two synthetic slab packs of different thickness (a 1-D reflecting random walk, compressed to DCT
coefficients — no simulator needed), which is exactly the *size discrimination* case. Checks that the
optimizer finds a deliverable waveform whose signals separate the two substrates, beating a naive PGSE
baseline, and that the differentiable replay forward drives it. Skipped if jax/optax/dmipy_sim absent.
"""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")

from scipy.fft import dct
from dmipy_sim.replay import ReplayPack, compile_scheme, replay_signal
from dmipy_sim.constants import GAMMA
from dmipy_design.replay_design import design_discriminating_waveform

N_W, N_T, K, DT, D0 = 600, 120, 40, 1e-3, 2e-9


def _slab_pack(L, seed):
    """A synthetic replay pack: N_W walkers doing a 1-D reflecting walk in a slab of thickness L along x
    (free along y,z), stored as DCT-II trajectory coefficients. Narrower L -> more restricted."""
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * D0 * DT)
    x = rng.uniform(0, L, N_W)
    traj = np.zeros((N_W, N_T, 3), np.float64)
    for t in range(N_T):
        x = x + rng.normal(0, step, N_W)
        x = np.mod(x, 2 * L); x = np.where(x > L, 2 * L - x, x)     # reflect in [0,L]
        traj[:, t, 0] = x
    traj[:, :, 1:] = np.cumsum(rng.normal(0, step, (N_W, N_T, 2)), axis=1)   # free y,z
    traj -= traj.mean(1, keepdims=True)
    C = dct(traj, type=2, norm="ortho", axis=1)[:, :K, :]
    arrays = {"dct_coeffs": C.astype(np.float32), "spin_weights": np.ones(N_W, np.float32)}
    meta = {"n_t": N_T, "dt": DT, "walk_params": {"n_t": N_T, "dt_traj": DT}}
    return ReplayPack(arrays, meta)


@pytest.fixture(scope="module")
def packs():
    return _slab_pack(4e-6, 0), _slab_pack(8e-6, 1)      # 4 um vs 8 um slabs


def _pgse_contrast(pack_a, pack_b, b, delta=8e-3, Delta=None):
    "Contrast |E_a - E_b| of a plain PGSE at b (baseline to beat)."
    Delta = Delta or (N_T - 1) * DT * 0.9
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    g = np.zeros((1, N_T, 3)); nd = max(1, int(round(delta / DT))); ng = int(round(Delta / DT))
    g[0, :nd, 0] = np.sqrt(b / bu); g[0, ng:ng + nd, 0] = -np.sqrt(b / bu)
    ea = replay_signal(pack_a, compile_scheme(g, DT, pack_a.K, GAMMA))[0]
    eb = replay_signal(pack_b, compile_scheme(g, DT, pack_b.K, GAMMA))[0]
    return abs(ea - eb)


def test_discrimination_optimizer_separates_slabs(packs):
    pa, pb = packs
    res = design_discriminating_waveform(pa, pb, direction=(1., 0, 0), G_max=0.3,
                                         maxiter=250, n_restarts=4, seed=0)
    # a real separation, and the two substrates genuinely differ under the optimized waveform
    assert res.contrast > 0.05
    assert abs(res.E_A - res.E_B) == pytest.approx(res.contrast, abs=1e-9)
    # beats the best of a naive PGSE b-sweep (the point of optimizing the waveform)
    baseline = max(_pgse_contrast(pa, pb, b) for b in (1e9, 2e9, 4e9, 8e9))
    assert res.contrast >= baseline - 1e-3
    # deliverable: amplitude bounded and (near-)refocused
    assert np.abs(res.G).max() <= 0.3 + 1e-6
    assert abs(np.sum(res.G[:, 0]) * DT) < 0.3 * DT * N_T * 0.02    # |q(TE)| small


def test_forward_matches_engine(packs):
    "The signals the optimizer differentiates through equal the sim replay engine's forward."
    pa, _ = packs
    res = design_discriminating_waveform(pa, packs[1], G_max=0.3, maxiter=40, n_restarts=1, seed=1)
    W = compile_scheme(res.G[None], DT, pa.K, GAMMA)
    npt.assert_allclose(replay_signal(pa, W)[0], res.E_A, atol=1e-4)
