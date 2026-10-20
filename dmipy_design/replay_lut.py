"""Parametric-family response LUT — instant substrate-informed design over an acquisition family.

The replay forward for one candidate waveform is already a cheap matmul (``dmipy_design.replay_design``),
but *design* often sweeps a whole family of acquisitions against many substrates — e.g. "for every pair of
diameters, which PGSE b best separates them?". This module *lowers* each replay pack to a small table of
its signal over a parametric acquisition grid (the design analog of a fit LUT): precompute once, then any
design query is an O(1) lookup / interpolation instead of a per-candidate replay.

Scope: PGSE at fixed pulse timing over a b-grid (the workhorse), optionally per direction. The signal for
a pack is evaluated with the exact ``dmipy_sim.replay`` forward, so the LUT is exact at its grid nodes and
linearly interpolated between them. NumPy/SciPy only (no autodiff, no JAX), matching the package.
"""
import numpy as np

from .replay_design import DESIGN_RASTER

__all__ = ["pgse_response_lut", "lut_pgse", "best_discriminating_pgse", "discriminability_matrix"]


def lut_pgse(pack, b_grid, *, delta, Delta, direction=(1.0, 0.0, 0.0), dt=DESIGN_RASTER):
    """The PGSE family a LUT is built over: dmipy-sim's square PGSE, one measurement per b in ``b_grid``, the
    pair centred in the pack's window ``T`` (``pack.T_window``, the sequence's TE), sampled on the raster ``dt``."""
    from dmipy_sim.sequences import pgse
    T = float(pack.T_window)
    b_grid = np.asarray(b_grid, float)
    return pgse(np.tile(np.asarray(direction, float), (len(b_grid), 1)), delta, Delta, bvalues=b_grid,
                TE=T, n_t=int(round(T / float(dt))) + 1, slew_rate=np.inf)


def pgse_response_lut(packs, b_grid, *, delta=None, Delta=None, direction=(1.0, 0.0, 0.0), dt=DESIGN_RASTER):
    """Precompute the PGSE signal of each pack over ``b_grid`` at fixed ``(delta, Delta)`` along
    ``direction``. Returns ``E`` of shape ``(n_packs, n_b)`` (exact replay at each node). ``delta``/``Delta``
    default to a symmetric split of the packs' window (``delta = T / 5``, ``Delta = 3 T / 5``)."""
    from dmipy_sim.replay import compile_scheme, replay_signal
    from dmipy_sim.constants import GAMMA
    packs = list(packs)
    T = float(packs[0].T_window)
    if any(abs(float(p.T_window) - T) > 1e-12 * T for p in packs):
        raise ValueError("all packs must share the window T (pack.T_window)")
    delta = delta if delta is not None else 0.2 * T
    Delta = Delta if Delta is not None else 0.6 * T
    b_grid = np.asarray(b_grid, float)
    # one sequence, one measurement per b (shared across packs); compile per pack K
    seq = lut_pgse(packs[0], b_grid, delta=delta, Delta=Delta, direction=direction, dt=dt)
    G = np.asarray(seq.G_eff, np.float64)
    E = np.zeros((len(packs), len(b_grid)))
    for i, p in enumerate(packs):
        E[i] = replay_signal(p, compile_scheme(G, float(seq.dt), p.K, GAMMA, T=T))
    return E


def best_discriminating_pgse(E, b_grid, i, j):
    """Best PGSE b to separate pack ``i`` from pack ``j`` given a response LUT ``E`` (n_packs, n_b):
    the ``b`` maximizing ``|E_i(b) - E_j(b)|``. Returns ``(b_star, contrast)`` (grid-argmax)."""
    contrast = np.abs(E[i] - E[j])
    k = int(np.argmax(contrast))
    return float(np.asarray(b_grid)[k]), float(contrast[k])


def discriminability_matrix(E, b_grid):
    """Pairwise best-contrast matrix ``M[i,j] = max_b |E_i(b) - E_j(b)|`` over the LUT — e.g. a
    size-resolvability map over a diameter family. Symmetric, zero diagonal."""
    n = E.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = np.abs(E[i] - E[j]).max()
    return M
