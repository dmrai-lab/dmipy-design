"""Parametric-family response LUT (T3): the precomputed table matches the exact replay forward, the
best-discriminating-b lookup agrees with a direct sweep, and the discriminability map orders by size
separation. Slab packs from dmipy-sim's producer (the ``slab_packs`` fixture). Skipped if dmipy_sim absent."""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")
from dmipy_sim.replay import compile_scheme, replay_signal
from dmipy_sim.constants import GAMMA
from dmipy_design.replay_lut import (pgse_response_lut, lut_pgse, best_discriminating_pgse,
                                     discriminability_matrix)

from conftest import SLAB_T as T

DT = 1e-4          # the LUT's raster in these tests


@pytest.fixture(scope="module")
def family(slab_packs):
    Ls = [3e-6, 5e-6, 8e-6]
    return [slab_packs[L] for L in Ls], Ls


def test_lut_matches_replay(family):
    packs, _ = family
    b_grid = np.linspace(0, 4e9, 9)
    delta, Delta = 0.25 * T, 0.7 * T
    E = pgse_response_lut(packs, b_grid, delta=delta, Delta=Delta, direction=(1., 0, 0), dt=DT)
    # spot-check a node against a direct replay of the same PGSE (dmipy-sim's builder over the pack's window)
    seq = lut_pgse(packs[0], [b_grid[5]], delta=delta, Delta=Delta, direction=(1., 0, 0), dt=DT)
    npt.assert_allclose(seq.b(), [b_grid[5]], rtol=1e-6)
    assert seq.dt == pytest.approx(DT)
    direct = replay_signal(packs[0], compile_scheme(np.asarray(seq.G_eff, np.float64), seq.dt, packs[0].K, GAMMA, T=T))[0]
    npt.assert_allclose(E[0, 5], direct, atol=1e-9)
    assert packs[0].replay(seq, tissue=None)[0] == pytest.approx(direct, abs=1e-6)   # the pack reads the object too


def test_best_discriminating_b(family):
    packs, _ = family
    b_grid = np.linspace(0, 4e9, 25)
    E = pgse_response_lut(packs, b_grid, direction=(1., 0, 0), dt=DT)
    b_star, contrast = best_discriminating_pgse(E, b_grid, 0, 2)     # 3um vs 8um
    assert contrast == pytest.approx(np.abs(E[0] - E[2]).max())
    assert contrast > 0.04 and b_star > 0                            # a real, non-trivial best b


def test_discriminability_orders_by_size(family):
    packs, Ls = family                                               # L = 3, 5, 8 um
    E = pgse_response_lut(packs, np.linspace(0, 4e9, 25), direction=(1., 0, 0), dt=DT)
    M = discriminability_matrix(E, None)
    npt.assert_allclose(np.diag(M), 0.0)
    npt.assert_allclose(M, M.T)
    assert M[0, 2] > M[0, 1]                                         # 3-vs-8 more separable than 3-vs-5
