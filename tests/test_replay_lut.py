"""Parametric-family response LUT (T3): the precomputed table matches the exact replay forward, the
best-discriminating-b lookup agrees with a direct sweep, and the discriminability map orders by size
separation. Synthetic slab packs (no simulator). Skipped if dmipy_sim absent."""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")
from scipy.fft import dct
from dmipy_sim.replay import ReplayPack, compile_scheme, replay_signal
from dmipy_sim.constants import GAMMA
from dmipy_design.replay_lut import (pgse_response_lut, best_discriminating_pgse,
                                     discriminability_matrix)

N_W, N_T, K, DT, D0 = 500, 120, 40, 1e-3, 2e-9


def _slab_pack(L, seed):
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * D0 * DT); x = rng.uniform(0, L, N_W)
    traj = np.zeros((N_W, N_T, 3))
    for t in range(N_T):
        x = x + rng.normal(0, step, N_W); x = np.mod(x, 2 * L); x = np.where(x > L, 2 * L - x, x)
        traj[:, t, 0] = x
    traj[:, :, 1:] = np.cumsum(rng.normal(0, step, (N_W, N_T, 2)), axis=1)
    traj -= traj.mean(1, keepdims=True)
    C = dct(traj, type=2, norm="ortho", axis=1)[:, :K, :]
    return ReplayPack({"dct_coeffs": C.astype(np.float32), "spin_weights": np.ones(N_W, np.float32)},
                      {"n_t": N_T, "dt": DT, "walk_params": {"n_t": N_T, "dt_traj": DT}})


@pytest.fixture(scope="module")
def family():
    Ls = [3e-6, 5e-6, 8e-6]
    return [_slab_pack(L, s) for s, L in enumerate(Ls)], Ls


def test_lut_matches_replay(family):
    packs, _ = family
    b_grid = np.linspace(0, 4e9, 9)
    E = pgse_response_lut(packs, b_grid, delta=0.25 * (N_T - 1) * DT,
                          Delta=0.7 * (N_T - 1) * DT, direction=(1., 0, 0))
    # spot-check a node against a direct replay of the same PGSE
    dt = packs[0].dt; delta = 0.25 * (N_T - 1) * DT; Delta = 0.7 * (N_T - 1) * DT
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    g = np.zeros((1, N_T, 3)); nd = max(1, int(round(delta / dt))); ng = int(round(Delta / dt))
    b = b_grid[5]; g[0, :nd, 0] = np.sqrt(b / bu); g[0, ng:ng + nd, 0] = -np.sqrt(b / bu)
    direct = replay_signal(packs[0], compile_scheme(g, dt, packs[0].K, GAMMA))[0]
    npt.assert_allclose(E[0, 5], direct, atol=1e-9)


def test_best_discriminating_b(family):
    packs, _ = family
    b_grid = np.linspace(0, 4e9, 25)
    E = pgse_response_lut(packs, b_grid, direction=(1., 0, 0))
    b_star, contrast = best_discriminating_pgse(E, b_grid, 0, 2)     # 3um vs 8um
    assert contrast == pytest.approx(np.abs(E[0] - E[2]).max())
    assert contrast > 0.04 and b_star > 0                            # a real, non-trivial best b


def test_discriminability_orders_by_size(family):
    packs, Ls = family                                               # L = 3, 5, 8 um
    E = pgse_response_lut(packs, np.linspace(0, 4e9, 25), direction=(1., 0, 0))
    M = discriminability_matrix(E, None)
    npt.assert_allclose(np.diag(M), 0.0)
    npt.assert_allclose(M, M.T)
    assert M[0, 2] > M[0, 1]                                         # 3-vs-8 more separable than 3-vs-5
