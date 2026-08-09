"""Optimize gradient waveforms *through* replay packs — substrate-informed acquisition design.

The pivot: instead of re-simulating a substrate (or replaying raw ~GB trajectory arrays) for every
candidate waveform, we replay pre-computed ``.rpk`` replay packs. A pack is fixed; the waveform g(t) is
the optimization variable; the diffusion-weighted signal is a matmul on the pack's DCT-compressed
trajectory (``dmipy_sim.replay``), and its gradient w.r.t. the waveform is closed-form — so waveform
design is gradient descent through the stored Monte-Carlo substrate, with no autodiff (this package is
NumPy/SciPy-only by design: fully numpy-analytic derivatives → SciPy L-BFGS-B, per the repo philosophy).

First use case: **shape/size discrimination** — given two packs (e.g. a cylinder and a sphere, or two
diameters), find the deliverable waveform that maximally separates their signals, ``max_g |E_A - E_B|``.

The signal of one 1-axis waveform ``g(t)`` (direction ``d`` fixed) against a pack with DCT coefficients
``C`` (n_walkers, K, 3) and weights ``w`` is, with ``ghat = DCT(g)[:K]`` and ``Cd = C . d``,

    phi_i = gamma * dt * sum_k Cd_{i,k} ghat_k,   S = <w exp(i phi)>/<w>,   E = |S|,

and the analytic gradient chains back through the (linear) DCT to ``g``:

    dE/dghat_k = (gamma dt / |S|) * Re( conj(S) * i * <w exp(i phi) Cd_{:,k}> ),   dE/dg = idct(dE/dghat).

Substrate set: the Substrate Commons canonical replay dataset (cylinders / spheres / planes,
0.1-20 um). Orientation convention: anisotropic packs (cylinder/plane) are replayed in their canonical
frame (cylinder axis = z), i.e. the design gradient is perpendicular to the axis unless a direction is
given — the discriminating axis for shape/size contrast.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize

__all__ = ["load_pack", "design_discriminating_waveform", "DiscriminationResult"]


def load_pack(path):
    """Load an ``.rpk`` replay pack (thin wrapper over :func:`dmipy_sim.replay.read_rpk`)."""
    from dmipy_sim.replay import read_rpk
    return read_rpk(path)


@dataclass
class DiscriminationResult:
    G: np.ndarray            # (n_t, 3) optimized gradient waveform [T/m]
    dt: float                # save-grid interval [s]
    direction: np.ndarray    # (3,) encoding direction
    contrast: float          # |E_A - E_B| at the optimum
    E_A: float
    E_B: float
    b_value: float           # achieved b [s/m^2]


def _bvalue(g_axis, dt, gamma):
    q = gamma * np.cumsum(np.asarray(g_axis, float)) * dt
    return float(np.sum(q * q) * dt)


class _PackForward:
    """Analytic E(g) and dE/dg for one pack along a fixed direction, at fixed (dt, n_t, gamma)."""

    def __init__(self, pack, direction, gamma):
        C = np.asarray(pack.dct_coeffs, np.float64)               # (n_walkers, K, 3)
        self.K = C.shape[1]
        self.Cd = C @ np.asarray(direction, np.float64)           # (n_walkers, K)  = C . d
        self.w = np.asarray(pack.spin_weights, np.float64)
        self.W0 = self.w.sum()
        self.dt = float(pack.dt)
        self.n_t = int(pack.n_t)
        self.gamma = gamma

    def E_and_grad(self, g):
        "Return (E, dE/dg) for the 1-axis waveform g (n_t,)."
        ghat = dct(g, type=2, norm="ortho")[: self.K]             # (K,)
        phi = self.gamma * self.dt * (self.Cd @ ghat)             # (n_walkers,)
        e = np.exp(1j * phi)
        we = self.w * e
        S = we.sum() / self.W0
        absS = np.abs(S)
        if absS < 1e-12:
            return 0.0, np.zeros_like(g)
        # dE/dghat_k = (gamma dt / |S|) Re( conj(S) * i * sum_i we_i Cd_{i,k} )
        acc = (we[:, None] * self.Cd).sum(0)                      # (K,) = sum_i we_i Cd_{i,k}
        dE_dghat = (self.gamma * self.dt / absS) * np.real(np.conj(S) * 1j * acc) / self.W0
        # dE/dg = idct of the (K-truncated, zero-padded) dE/dghat  (DCT-II ortho is orthogonal)
        padded = np.zeros(self.n_t); padded[: self.K] = dE_dghat
        dE_dg = idct(padded, type=2, norm="ortho")
        return float(absS), dE_dg


def design_discriminating_waveform(pack_a, pack_b, *, direction=(1.0, 0.0, 0.0), G_max=0.08,
                                   n_basis=16, n_restarts=4, maxiter=300, seed=0, refocus_weight=50.0):
    """Design a single deliverable gradient waveform that maximally discriminates ``pack_a`` from
    ``pack_b`` (both :class:`dmipy_sim.replay.ReplayPack`), i.e. ``max_g |E_A(g) - E_B(g)|``.

    The 1-axis waveform g(t) is a smooth low-order cosine (DCT) synthesis of ``n_basis`` modes (no DC
    mode → refocusing is structural; band-limited → deliverable — a free per-sample optimizer wanders
    into a high-frequency OGSE-like waveform with high nominal b but ~zero contrast). Amplitude bounded
    (``|g| <= G_max`` via tanh). Optimized by SciPy L-BFGS-B with the analytic gradient (no autodiff),
    warm-started from the best plain PGSE (a cold start sits at ~zero contrast gradient), multi-restart.

    Both packs must share the save grid (``dt``, ``n_t``); ``K`` may differ. Returns a
    :class:`DiscriminationResult`.
    """
    from dmipy_sim.constants import GAMMA

    if abs(pack_a.dt - pack_b.dt) > 1e-12 or pack_a.n_t != pack_b.n_t:
        raise ValueError("packs must share the save grid (dt, n_t)")
    dt, n_t = float(pack_a.dt), int(pack_a.n_t)
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    fa, fb = _PackForward(pack_a, d, GAMMA), _PackForward(pack_b, d, GAMMA)

    # smooth cosine synthesis basis, no DC column (structural refocusing): g = G_max tanh(B c)
    tt = (np.arange(n_t) + 0.5) / n_t
    kk = np.arange(1, int(n_basis) + 1)
    B = np.cos(np.pi * np.outer(tt, kk))                          # (n_t, n_basis)

    def g_of(c):
        raw = B @ c
        return G_max * np.tanh(raw), raw

    def loss_and_grad(c):
        g, raw = g_of(c)
        Ea, dEa = fa.E_and_grad(g)
        Eb, dEb = fb.E_and_grad(g)
        diff = Ea - Eb
        # objective: -(Ea-Eb)^2 + refocus_weight (sum g dt)^2  (minimize)
        L = -diff ** 2 + refocus_weight * (np.sum(g) * dt) ** 2
        dL_dg = -2 * diff * (dEa - dEb) + refocus_weight * 2 * (np.sum(g) * dt) * dt
        dg_draw = G_max * (1.0 - np.tanh(raw) ** 2)               # d g / d raw
        dL_dc = B.T @ (dL_dg * dg_draw)
        return float(L), np.asarray(dL_dc, np.float64)

    # warm start: best plain PGSE projected onto the basis (via least squares on the pre-tanh signal)
    delta = max(2 * dt, 0.25 * n_t * dt)
    Delta = min((n_t - 2) * dt, delta + 0.5 * n_t * dt)
    nd, ng = max(1, int(round(delta / dt))), int(round(Delta / dt))
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    def _pgse_c(b):
        amp = min(np.sqrt(b / bu), 0.9 * G_max)
        g = np.zeros(n_t); g[:nd] = amp; g[ng:ng + nd] = -amp
        return np.linalg.lstsq(B, np.arctanh(np.clip(g / G_max, -0.9, 0.9)), rcond=None)[0]
    def _contrast_c(c):
        g, _ = g_of(c); return abs(fa.E_and_grad(g)[0] - fb.E_and_grad(g)[0])
    warm = max((_pgse_c(b) for b in np.linspace(0.5e9, 12e9, 10)), key=_contrast_c)

    best = None
    rng = np.random.default_rng(seed)
    inits = [warm] + [warm + 0.3 * rng.standard_normal(len(warm)) for _ in range(max(0, n_restarts - 1))]
    for c0 in inits:
        res = minimize(loss_and_grad, c0, jac=True, method="L-BFGS-B", options={"maxiter": int(maxiter)})
        if best is None or res.fun < best.fun:
            best = res
    g_final, _ = g_of(best.x)
    Ea = fa.E_and_grad(g_final)[0]; Eb = fb.E_and_grad(g_final)[0]
    G = g_final[:, None] * d[None, :]
    return DiscriminationResult(G=G, dt=dt, direction=d, contrast=abs(Ea - Eb),
                                E_A=Ea, E_B=Eb, b_value=_bvalue(g_final, dt, GAMMA))
