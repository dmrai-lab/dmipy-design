"""Optimize gradient waveforms *through* replay packs — substrate-informed acquisition design.

A pack is fixed; the waveform g(t) is the optimization variable; the diffusion-weighted signal is a matmul on
the pack's stored bands (``dmipy_sim.replay``), and its gradient w.r.t. the waveform is closed-form — so waveform
design is gradient descent through the stored Monte-Carlo substrate, with no autodiff (this package is
NumPy/SciPy-only: fully numpy-analytic derivatives → SciPy L-BFGS-B).

First use case: **shape/size discrimination** — given two packs (e.g. a cylinder and a sphere, or two
diameters), find the deliverable waveform that maximally separates their signals, ``max_g |E_A - E_B|``.

The signal of one 1-axis waveform ``g(t)`` (direction ``d`` fixed) against a pack is dmipy-sim's own replay
forward: ``phi_i = C_i . W(g)`` with ``C`` the pack's position coefficients (per axis the window's exact start and
displacement, then its ``K`` sine bands on the continuous basis) and ``W(g) = M g`` the waveform's band projection
(``dmipy_sim.replay.compression.band_projection_matrix`` along ``d``, LINEAR in ``g``), ``E = |<w exp(i phi)>| /
sum(w)``, and the analytic gradient chains back through ``M``:

    dE/dW = (1/|S|) Re( conj(S) * i * <w exp(i phi) C> ) / W0,   dE/dg = M^T dE/dW.

The waveform is synthesised in the pack's own band basis: ``n_basis`` of the window's sines ``sin(pi k t / T)``,
``k = 1..n_basis <= K``, so the design is band-limited to what the packs store by construction (no DC mode:
refocusing is structural), sampled on the design raster ``dt`` (the scanner's gradient raster, or
:data:`DESIGN_RASTER`). Orientation convention: anisotropic packs (cylinder/plane) are replayed in their canonical
frame (cylinder axis = z), i.e. the design gradient is perpendicular to the axis unless a direction is given — the
discriminating axis for shape/size contrast.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["load_pack", "design_discriminating_waveform", "DiscriminationResult", "DESIGN_RASTER", "design_raster"]

_MIN_SLEW_PENALTY = 1e-3

DESIGN_RASTER = 10e-6
"""The grid (s) a design is sampled on when the scanner limits state no gradient raster."""


def design_raster(limits, dt=None):
    """The design grid (s): ``dt`` when given, else the limits' gradient raster, else :data:`DESIGN_RASTER`."""
    if dt is not None:
        return float(dt)
    raster = getattr(limits, "grad_raster", None)
    return float(raster) if raster else DESIGN_RASTER


def load_pack(path):
    """Load an ``.rpk`` replay pack (thin wrapper over :meth:`dmipy_sim.replay.ReplayPack.load`)."""
    from dmipy_sim.replay import ReplayPack
    return ReplayPack.load(path)


@dataclass
class DiscriminationResult:
    G: np.ndarray            # (n_t, 3) optimized gradient waveform [T/m] on the design raster
    dt: float                # design raster [s]
    direction: np.ndarray    # (3,) encoding direction
    contrast: float          # |E_A - E_B| at the optimum
    E_A: float
    E_B: float
    b_value: float           # achieved b [s/m^2]
    te: float = None         # encoding window used [s] (the packs' window when none was given)
    max_slew: float = None   # realized peak slew [T/m/s]

    def to_sequence(self):
        """The designed waveform as a dmipy-sim ``ScannerSequence``: a self-refocusing gradient echo (no pulse
        is folded; the structural refocusing of the design makes it refocus on its own)."""
        from dmipy_sim.sequences import from_waveform
        return from_waveform(np.asarray(self.G, np.float32)[None], self.dt, self.direction[None])


class _PackForward:
    """Analytic E(g) and dE/dg for one pack along a fixed direction: dmipy-sim's own replay forward.

    The pack replays a waveform through its band projection, ``phi_i = C_i . W(g)`` with ``C`` the pack's position
    coefficients (``pack.position_coeffs``) and ``W(g) = M g`` the waveform's projection on the pack's bands
    (``dmipy_sim.replay.compression.band_projection_matrix`` over the window, times ``gamma``, along ``d``). So
    ``E(g)`` is exactly what ``pack.replay`` returns for the same waveform, and ``dE/dg = M^T dE/dW`` is
    closed-form -- no autodiff, no second copy of the replay mathematics here.
    """

    def __init__(self, pack, direction, gamma, dt, n_t):
        from dmipy_sim.replay.compression import band_projection_matrix
        self.dt, self.n_t, self.T = float(dt), int(n_t), float(pack.T_window)
        d = np.asarray(direction, np.float64)
        C = np.asarray(pack.position_coeffs, np.float64)                  # (n_walkers, K+2, 3)
        self.Cflat = C.reshape(C.shape[0], -1)                            # (n_walkers, 3(K+2))
        P = band_projection_matrix(self.n_t, self.dt, self.T, pack.K)     # (K+2, n_t): unit samples on the bands
        self.M = gamma * (P[:, None, :] * d[None, :, None]).reshape(-1, self.n_t)   # (3(K+2), n_t): W(g) = M g
        self.w = np.asarray(pack.spin_weights, np.float64)
        self.W0 = self.w.sum()

    def E_and_grad(self, g):
        "Return (E, dE/dg) for the 1-axis waveform g (n_t,)."
        phi = self.Cflat @ (self.M @ np.asarray(g, np.float64))     # (n_walkers,)
        we = self.w * np.exp(1j * phi)
        S = we.sum() / self.W0
        absS = np.abs(S)
        if absS < 1e-12:
            return 0.0, np.zeros(self.n_t)
        # dE/dW_c = (1/|S|) Re( conj(S) * i * sum_i we_i C_{i,c} ) / W0 ;  dE/dg = M^T dE/dW
        acc = self.Cflat.T @ we                                      # (3(K+2),)
        dE_dW = np.real(np.conj(S) * 1j * acc) / (absS * self.W0)
        return float(absS), self.M.T @ dE_dW


def design_discriminating_waveform(pack_a, pack_b, *, limits, direction=(1.0, 0.0, 0.0),
                                   te=None, dt=None, n_basis=16, n_restarts=4, maxiter=300,
                                   seed=0, refocus_weight=50.0, slew_weight=_MIN_SLEW_PENALTY):
    """Design a single deliverable gradient waveform that maximally discriminates ``pack_a`` from
    ``pack_b`` (both :class:`dmipy_sim.replay.ReplayPack`), i.e. ``max_g |E_A(g) - E_B(g)|``.

    The 1-axis waveform g(t) is a synthesis of ``n_basis`` of the packs' own band basis, the window's sines
    ``sin(pi k t / T)`` for ``k = 1..n_basis`` (``n_basis`` at most the smaller pack's ``K``: the design never
    leaves the band the packs store; no DC mode → refocusing is structural; band-limited → deliverable — a free
    per-sample optimizer wanders into a high-frequency OGSE-like waveform with high nominal b but ~zero contrast),
    sampled on the design raster ``dt`` (:func:`design_raster`). Amplitude bounded (``|g| <= G_max`` via tanh).
    Optimized by SciPy L-BFGS-B with the analytic gradient (no autodiff), warm-started from the best plain PGSE
    (a cold start sits at ~zero contrast gradient), multi-restart.

    ``limits`` (a :class:`~dmipy_sim.acquisition.scanners.ScannerLimits`, or anything ``ScannerLimits.of``
    resolves) bounds the amplitude (``|g| <= G_max`` via tanh) and adds a soft slew penalty at its slew rate so
    the waveform is scanner-deliverable (the smooth basis already band-limits slew; this bounds it explicitly).
    ``te`` restricts the encoding to a window ``[0, te]`` (the waveform is zero after; the echo forms at ``te`` --
    the pack's TE-prefix property), which bounds ``b`` to a realistic range. Both packs must share the window
    ``T`` (``pack.T_window``); ``K`` may differ. Returns a :class:`DiscriminationResult`.
    """
    from dmipy_sim.acquisition.scanners import ScannerLimits
    from dmipy_sim.acquisition.waveforms import b_from_gradient
    from dmipy_sim.constants import GAMMA
    limits = ScannerLimits.of(limits)
    G_max, slew_max = float(limits.G_max), float(limits.slew_max)

    T = float(pack_a.T_window)
    if abs(T - float(pack_b.T_window)) > 1e-12 * T:
        raise ValueError(f"packs must share the window T ({T:.6g} s vs {float(pack_b.T_window):.6g} s)")
    K_shared = min(int(pack_a.K), int(pack_b.K))
    if int(n_basis) > K_shared:
        raise ValueError(f"n_basis = {int(n_basis)} exceeds the packs' band (K = {K_shared}): the design basis is the "
                         f"packs' own sines, so a design cannot use bands they do not store")
    dt = design_raster(limits, dt)
    n_t = int(round(T / dt))                                      # sample-and-hold samples covering [0, T)
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    fa, fb = _PackForward(pack_a, d, GAMMA, dt, n_t), _PackForward(pack_b, d, GAMMA, dt, n_t)

    # the packs' band basis as the synthesis basis, no DC column (structural refocusing): g = mask * G_max tanh(B c)
    tt = (np.arange(n_t) + 0.5) * dt / T
    kk = np.arange(1, int(n_basis) + 1)
    B = np.sin(np.pi * np.outer(tt, kk))                          # (n_t, n_basis)
    mask = np.ones(n_t)
    if te is not None:
        te_idx = min(n_t, max(2, int(round(te / dt))))
        mask = np.zeros(n_t); mask[:te_idx] = 1.0                 # encode only within [0, te]

    def g_of(c):
        raw = B @ c
        return mask * (G_max * np.tanh(raw)), raw

    def _slew_pen_and_grad(g):
        "Soft penalty for |dg/dt| exceeding slew_max: sum relu(|s|-slew_max)^2, with its dpen/dg."
        if not np.isfinite(slew_max):
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
    # fitted within the encoding window (te, or the packs' window)
    win = (te if te is not None else T)
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
    # the soft refocusing term leaves q(TE) at ~1e-3 of its peak; the sequence a designer hands on must refocus
    # exactly, so the net moment is removed along the window's first sine (zero at both edges, so no step is
    # introduced) -- a change below the optimizer's own tolerance -- and the contrast reported is that of the
    # waveform actually returned
    on = mask > 0
    h = np.where(on, np.sin(np.pi * (np.arange(n_t) + 0.5) / max(1, on.sum())), 0.0)
    g_final = g_final - (g_final.sum() / h.sum()) * h
    Ea = fa.E_and_grad(g_final)[0]; Eb = fb.E_and_grad(g_final)[0]
    G = g_final[:, None] * d[None, :]
    max_slew = float(np.abs(np.diff(g_final) / dt).max())
    return DiscriminationResult(G=G, dt=dt, direction=d, contrast=abs(Ea - Eb),
                                E_A=Ea, E_B=Eb, b_value=float(b_from_gradient(G[None], dt)[0]),
                                te=(te if te is not None else T), max_slew=max_slew)
