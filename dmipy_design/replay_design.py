"""Optimize gradient waveforms *through* replay packs — substrate-informed acquisition design.

The pivot: instead of re-simulating a substrate (or replaying raw ~GB trajectory arrays) for every
candidate waveform, we replay pre-computed ``.rpk`` replay packs. A pack is fixed; the waveform G(t) is
the optimization variable; the diffusion-weighted signal is a single matmul on the pack's DCT-compressed
trajectory (``dmipy_sim.replay``), and — crucially — it is differentiable in G, so waveform design becomes
gradient descent through the stored Monte-Carlo substrate.

This module provides the **shape/size discrimination** objective as the first use case: given two packs
(e.g. a cylinder and a sphere, or two diameters), find the deliverable waveform that maximally separates
their signals, ``max_G |E_A(G) - E_B(G)|``. The forward stays compressed (K DCT modes) and JAX-jittable,
so the optimizer never touches full trajectories.

Substrate set: the Substrate Commons canonical replay dataset (cylinders / spheres / planes,
0.1-20 um). Orientation convention here: anisotropic packs (cylinder/plane) are replayed in their
canonical frame (cylinder axis = z), i.e. the design gradient is taken perpendicular to the axis unless
a direction is given — the discriminating axis for shape/size contrast.
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["load_pack", "design_discriminating_waveform", "DiscriminationResult"]


def load_pack(path):
    """Load an ``.rpk`` replay pack (thin wrapper over :func:`dmipy_sim.replay.read_rpk`)."""
    from dmipy_sim.replay import read_rpk
    return read_rpk(path)


@dataclass
class DiscriminationResult:
    """Result of a discrimination design: the optimized waveform and the contrast it achieves."""
    G: np.ndarray            # (n_t, 3) optimized gradient waveform [T/m]
    dt: float                # save-grid interval [s]
    direction: np.ndarray    # (3,) encoding direction
    contrast: float          # |E_A - E_B| at the optimum
    E_A: float
    E_B: float
    b_value: float           # achieved b [s/m^2]
    history: list            # contrast per iteration


def _bvalue(g_axis, dt, gamma):
    q = gamma * np.cumsum(np.asarray(g_axis, float)) * dt
    return float(np.sum(q * q) * dt)


def design_discriminating_waveform(pack_a, pack_b, *, direction=(1.0, 0.0, 0.0), G_max=0.08,
                                   n_basis=16, n_restarts=4, maxiter=300, seed=0, refocus_weight=50.0):
    """Design a single deliverable gradient waveform that maximally discriminates ``pack_a`` from
    ``pack_b`` (both :class:`dmipy_sim.replay.ReplayPack`), i.e. ``max_G |E_A(G) - E_B(G)|``.

    The 1-axis waveform g(t) is expanded in a smooth low-order cosine (DCT) basis of ``n_basis`` modes
    (band-limited → deliverable; a free per-sample optimizer wanders into a high-frequency OGSE-like
    waveform with high nominal b but ~zero contrast — the repo's smooth-basis lesson). Coefficients are
    optimized through the differentiable replay forward by L-BFGS-B (JAX ``value_and_grad`` →
    ``scipy.optimize.minimize``, the repo's gradient-OED pattern), warm-started from the best plain PGSE
    and multi-restarted. The amplitude is bounded (``|g| <= G_max`` via a tanh squash) and the zeroth
    gradient moment nulled (refocused spin echo, ``q(TE)=0``) by a penalty. Returns a
    :class:`DiscriminationResult`.

    Both packs must share the save grid (``dt``, ``n_t``); ``K`` may differ. ``direction`` is the (fixed)
    encoding direction in the packs' canonical frame.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.fft import dct as jdct
    from scipy.optimize import minimize
    from dmipy_sim.constants import GAMMA

    if abs(pack_a.dt - pack_b.dt) > 1e-12 or pack_a.n_t != pack_b.n_t:
        raise ValueError("packs must share the save grid (dt, n_t)")
    dt, n_t = float(pack_a.dt), int(pack_a.n_t)
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)

    Ca = jnp.asarray(np.asarray(pack_a.dct_coeffs, np.float64)); Ka = int(Ca.shape[1])
    Cb = jnp.asarray(np.asarray(pack_b.dct_coeffs, np.float64)); Kb = int(Cb.shape[1])
    wa = jnp.asarray(np.asarray(pack_a.spin_weights, np.float64))
    wb = jnp.asarray(np.asarray(pack_b.spin_weights, np.float64))
    d_j = jnp.asarray(d)

    def signal(C, w, K, g_axis):
        G = g_axis[:, None] * d_j[None, :]                         # (n_t, 3)
        Ghat = jdct(G, type=2, norm="ortho", axis=0)[:K, :]        # (K, 3)
        W = (GAMMA * dt * Ghat).reshape(K * 3)                     # (3K,)
        phi = C.reshape(C.shape[0], K * 3) @ W                     # (n_walkers,)
        return jnp.abs((w * jnp.exp(1j * phi)).sum() / w.sum())

    # Smooth low-order cosine (DCT-III-like) synthesis basis: g(t) = sum_k c_k cos(pi k (t+0.5)/n_t).
    # Drop k=0 (the DC / zeroth-moment mode) so refocusing is structural, then band-limit to n_basis modes.
    tt = (np.arange(n_t) + 0.5) / n_t
    kk = np.arange(1, int(n_basis) + 1)
    B = jnp.asarray(np.cos(np.pi * np.outer(tt, kk)))              # (n_t, n_basis), no DC column

    def g_of(c):
        raw = B @ c                                               # smooth, zero-mean-ish
        return G_max * jnp.tanh(raw)                              # amplitude bound |g| <= G_max

    def _pair(c):
        g = g_of(c)
        return signal(Ca, wa, Ka, g), signal(Cb, wb, Kb, g)

    def loss(c):
        Ea, Eb = _pair(c)
        g = g_of(c)
        return -(Ea - Eb) ** 2 + refocus_weight * (jnp.sum(g) * dt) ** 2

    vg = jax.jit(jax.value_and_grad(loss))

    def scipy_vg(c):
        v, gr = vg(jnp.asarray(c))
        return float(v), np.asarray(gr, np.float64)

    # Warm start from the best plain PGSE (quick b-sweep), projected onto the smooth basis — from a random
    # init the signal is ~1 for both packs (flat contrast gradient) and the solver stalls at zero.
    delta = max(2 * dt, 0.25 * n_t * dt)
    Delta = min((n_t - 2) * dt, delta + 0.5 * n_t * dt)
    nd, ng = max(1, int(round(delta / dt))), int(round(Delta / dt))
    B_np = np.asarray(B)
    def _pgse_c(b):
        bu = (np.asarray(GAMMA) * delta) ** 2 * (Delta - delta / 3)
        amp = min(np.sqrt(b / bu), 0.9 * G_max)
        g = np.zeros(n_t); g[:nd] = amp; g[ng:ng + nd] = -amp
        return np.linalg.lstsq(B_np, np.arctanh(np.clip(g / G_max, -0.9, 0.9)), rcond=None)[0]
    def _contrast_c(c):
        Ea, Eb = _pair(jnp.asarray(c)); return abs(float(Ea) - float(Eb))
    warm = max((_pgse_c(b) for b in np.linspace(0.5e9, 12e9, 10)), key=_contrast_c)

    best = None
    rng = np.random.default_rng(seed)
    inits = [warm] + [warm + 0.3 * rng.standard_normal(len(warm)) for _ in range(max(0, int(n_restarts) - 1))]
    for c0 in inits:
        res = minimize(scipy_vg, c0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter)})
        if best is None or res.fun < best.fun:
            best = res
    c = jnp.asarray(best.x)
    Ea, Eb = (float(v) for v in _pair(c))
    g_final = np.asarray(g_of(c))
    G = g_final[:, None] * d[None, :]
    return DiscriminationResult(
        G=G, dt=dt, direction=d, contrast=abs(Ea - Eb),
        E_A=Ea, E_B=Eb, b_value=_bvalue(g_final, dt, GAMMA), history=[])
