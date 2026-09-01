"""Optimize gradient waveforms *through* replay packs — substrate-informed acquisition design.

The pivot: instead of re-simulating a substrate (or replaying raw ~GB trajectory arrays) for every
candidate waveform, we replay pre-computed ``.rpk`` replay packs. A pack is fixed; the waveform g(t) is
the optimization variable; the diffusion-weighted signal is a matmul on the pack's compressed
trajectory (``dmipy_sim.replay``), and its gradient w.r.t. the waveform is closed-form — so waveform
design is gradient descent through the stored Monte-Carlo substrate, with no autodiff (this package is
NumPy/SciPy-only by design: fully numpy-analytic derivatives → SciPy L-BFGS-B, per the repo philosophy).

First use case: **shape/size discrimination** — given two packs (e.g. a cylinder and a sphere, or two
diameters), find the deliverable waveform that maximally separates their signals, ``max_g |E_A - E_B|``.

The signal of one 1-axis waveform ``g(t)`` (direction ``d`` fixed) against a pack with DCT coefficients
``C`` (n_walkers, K, 3) and weights ``w`` is, with ``ghat = DCT(g)[:K]`` and ``Cd = C . d``,

    phi_i = gamma * dt * sum_k Cd_{i,k} ghat_k,   S = <w exp(i phi)>/<w>,   E = |S|,

and the analytic gradient chains back through the (linear) projection to ``g``:

    dE/dghat_k = (gamma dt / |S|) * Re( conj(S) * i * <w exp(i phi) Cd_{:,k}> ),   dE/dg = P^T (dE/dghat), P the projection above.

Substrate set: the Substrate Commons canonical replay dataset (cylinders / spheres / planes,
0.1-20 um). Orientation convention: anisotropic packs (cylinder/plane) are replayed in their canonical
frame (cylinder axis = z), i.e. the design gradient is perpendicular to the axis unless a direction is
given — the discriminating axis for shape/size contrast.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import dst, idst
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
    te: float = None         # encoding window used [s] (None = full pack window)
    max_slew: float = None   # realized peak slew [T/m/s]


def _bvalue(g_axis, dt, gamma):
    q = gamma * np.cumsum(np.asarray(g_axis, float)) * dt
    return float(np.sum(q * q) * dt)


class _PackForward:
    """Analytic E(g) and dE/dg for one pack along a fixed direction, at fixed (dt, n_t, gamma).

    Packs store positions as ``bridge_dst``: per axis, two exact endpoints followed by ``K``
    sine bands of the residual pinned at both ends.  The waveform is therefore projected onto
    ``[M0, M1, sine bands]`` -- the two gradient moments first, which a motion-compensated
    waveform makes vanish -- and the gradient steps back through the adjoint of that same map.
    """

    def __init__(self, pack, direction, gamma):
        from dmipy_sim.compression import read_position_coeffs, require_position_method
        require_position_method(pack.method)
        C = read_position_coeffs(pack.arrays, dtype=np.float64)   # (n_walkers, K+2, 3)
        # bands, not the stored width: the leading two coefficients are endpoints, and treating
        # them as bands would project the waveform onto a basis it was never expanded in
        self.K = C.shape[1] - 2
        self.Cd = C @ np.asarray(direction, np.float64)           # (n_walkers, K+2) = C . d
        self.w = np.asarray(pack.spin_weights, np.float64)
        self.W0 = self.w.sum()
        self.dt = float(pack.dt)
        self.n_t = int(pack.n_t)
        self.tau = np.arange(self.n_t) / (self.n_t - 1.0)
        self.gamma = gamma

    def _project(self, g):
        """g (n_t,) -> [M0, M1, sine bands] (K+2,), the basis the coefficients live in."""
        bands = dst(g[1:-1], type=1, norm="ortho")[: self.K]
        return np.concatenate([[g.sum(), float(self.tau @ g)], bands])

    def _project_adjoint(self, y):
        """Transpose of :meth:`_project`: (K+2,) -> (n_t,), for the chain rule back to g."""
        out = y[0] * np.ones(self.n_t) + y[1] * self.tau
        padded = np.zeros(self.n_t - 2)
        padded[: self.K] = y[2:]
        out[1:-1] += idst(padded, type=1, norm="ortho")
        return out

    def E_and_grad(self, g):
        "Return (E, dE/dg) for the 1-axis waveform g (n_t,)."
        ghat = self._project(g)                                   # (K+2,)
        phi = self.gamma * self.dt * (self.Cd @ ghat)             # (n_walkers,)
        e = np.exp(1j * phi)
        we = self.w * e
        S = we.sum() / self.W0
        absS = np.abs(S)
        if absS < 1e-12:
            return 0.0, np.zeros_like(g)
        # dE/dghat_k = (gamma dt / |S|) Re( conj(S) * i * sum_i we_i Cd_{i,k} )
        acc = (we[:, None] * self.Cd).sum(0)                      # (K+2,)
        dE_dghat = (self.gamma * self.dt / absS) * np.real(np.conj(S) * 1j * acc) / self.W0
        return float(absS), self._project_adjoint(dE_dghat)


def design_discriminating_waveform(pack_a, pack_b, *, direction=(1.0, 0.0, 0.0), G_max=0.08,
                                   te=None, slew_max=None, n_basis=16, n_restarts=4, maxiter=300,
                                   seed=0, refocus_weight=50.0, slew_weight=1e-3):
    """Design a single deliverable gradient waveform that maximally discriminates ``pack_a`` from
    ``pack_b`` (both :class:`dmipy_sim.replay.ReplayPack`), i.e. ``max_g |E_A(g) - E_B(g)|``.

    The 1-axis waveform g(t) is a smooth low-order cosine (DCT) synthesis of ``n_basis`` modes (no DC
    mode → refocusing is structural; band-limited → deliverable — a free per-sample optimizer wanders
    into a high-frequency OGSE-like waveform with high nominal b but ~zero contrast). Amplitude bounded
    (``|g| <= G_max`` via tanh). Optimized by SciPy L-BFGS-B with the analytic gradient (no autodiff),
    warm-started from the best plain PGSE (a cold start sits at ~zero contrast gradient), multi-restart.

    ``te`` restricts the encoding to a window ``[0, te]`` (the waveform is zero after; the echo forms at
    ``te`` — the pack's TE-prefix property), which bounds ``b`` to a realistic range. ``slew_max`` [T/m/s]
    adds a soft slew penalty so the waveform is scanner-deliverable (the smooth basis already band-limits
    slew; this bounds it explicitly). Both packs must share the save grid (``dt``, ``n_t``); ``K`` may
    differ. Returns a :class:`DiscriminationResult`.
    """
    from dmipy_sim.constants import GAMMA

    if abs(pack_a.dt - pack_b.dt) > 1e-12 or pack_a.n_t != pack_b.n_t:
        raise ValueError("packs must share the save grid (dt, n_t)")
    dt, n_t = float(pack_a.dt), int(pack_a.n_t)
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    fa, fb = _PackForward(pack_a, d, GAMMA), _PackForward(pack_b, d, GAMMA)

    # smooth cosine synthesis basis, no DC column (structural refocusing): g = mask * G_max tanh(B c)
    tt = (np.arange(n_t) + 0.5) / n_t
    kk = np.arange(1, int(n_basis) + 1)
    B = np.cos(np.pi * np.outer(tt, kk))                          # (n_t, n_basis)
    mask = np.ones(n_t)
    if te is not None:
        te_idx = min(n_t, max(2, int(round(te / dt))))
        mask = np.zeros(n_t); mask[:te_idx] = 1.0                 # encode only within [0, te]

    def g_of(c):
        raw = B @ c
        return mask * (G_max * np.tanh(raw)), raw

    def _slew_pen_and_grad(g):
        "Soft penalty for |dg/dt| exceeding slew_max: sum relu(|s|-slew_max)^2, with its dpen/dg."
        if slew_max is None:
            return 0.0, np.zeros_like(g)
        s = np.diff(g) / dt                                       # (n_t-1,)
        over = np.maximum(np.abs(s) - slew_max, 0.0)
        pen = slew_weight * float(np.sum(over ** 2))
        ds = slew_weight * 2 * over * np.sign(s) / dt            # dpen/ds_t
        dg = np.zeros_like(g)                                     # adjoint of the difference operator
        dg[:-1] -= ds; dg[1:] += ds
        return pen, dg

    def loss_and_grad(c):
        g, raw = g_of(c)
        Ea, dEa = fa.E_and_grad(g)
        Eb, dEb = fb.E_and_grad(g)
        diff = Ea - Eb
        spen, dspen = _slew_pen_and_grad(g)
        # objective: -(Ea-Eb)^2 + refocus_weight (sum g dt)^2 + slew penalty  (minimize)
        L = -diff ** 2 + refocus_weight * (np.sum(g) * dt) ** 2 + spen
        dL_dg = -2 * diff * (dEa - dEb) + refocus_weight * 2 * (np.sum(g) * dt) * dt + dspen
        dg_draw = mask * G_max * (1.0 - np.tanh(raw) ** 2)        # d g / d raw (through mask)
        dL_dc = B.T @ (dL_dg * dg_draw)
        return float(L), np.asarray(dL_dc, np.float64)

    # warm start: best plain PGSE projected onto the basis (via least squares on the pre-tanh signal),
    # fitted within the encoding window (te, or the full pack window)
    win = (te if te is not None else (n_t - 1) * dt)
    delta = max(2 * dt, 0.25 * win)
    Delta = min(win - dt, delta + 0.5 * win)
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
    max_slew = float(np.abs(np.diff(g_final) / dt).max())
    return DiscriminationResult(G=G, dt=dt, direction=d, contrast=abs(Ea - Eb),
                                E_A=Ea, E_B=Eb, b_value=_bvalue(g_final, dt, GAMMA),
                                te=(te if te is not None else (n_t - 1) * dt), max_slew=max_slew)
