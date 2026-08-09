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

__all__ = ["pgse_response_lut", "best_discriminating_pgse", "discriminability_matrix"]


def _pgse_G(n_t, dt, amp, delta, Delta, direction):
    g = np.zeros((n_t, 3)); nd = max(1, int(round(delta / dt))); ng = int(round(Delta / dt))
    u = np.asarray(direction, float); g[:nd] = amp * u; g[ng:ng + nd] = -amp * u
    return g


def pgse_response_lut(packs, b_grid, *, delta=None, Delta=None, direction=(1.0, 0.0, 0.0)):
    """Precompute the PGSE signal of each pack over ``b_grid`` at fixed ``(delta, Delta)`` along
    ``direction``. Returns ``E`` of shape ``(n_packs, n_b)`` (exact replay at each node). ``delta``/``Delta``
    default to a symmetric split of the packs' save window."""
    from dmipy_sim.replay import compile_scheme, replay_signal
    from dmipy_sim.constants import GAMMA
    packs = list(packs)
    dt, n_t = packs[0].dt, packs[0].n_t
    if any(abs(p.dt - dt) > 1e-12 or p.n_t != n_t for p in packs):
        raise ValueError("all packs must share the save grid (dt, n_t)")
    delta = delta if delta is not None else 0.25 * (n_t - 1) * dt
    Delta = Delta if Delta is not None else min((n_t - 2) * dt, delta + 0.5 * (n_t - 1) * dt)
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    b_grid = np.asarray(b_grid, float)
    # one waveform per b (shared across packs); compile per pack K
    G = np.stack([_pgse_G(n_t, dt, np.sqrt(b / bu), delta, Delta, direction) for b in b_grid])  # (n_b,n_t,3)
    E = np.zeros((len(packs), len(b_grid)))
    for i, p in enumerate(packs):
        E[i] = replay_signal(p, compile_scheme(G, dt, p.K, GAMMA))
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
