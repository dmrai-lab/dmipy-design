"""Optimize gradient waveforms *through* replay packs — substrate-informed acquisition design.

The pivot: instead of re-simulating a substrate (or replaying raw ~GB trajectory arrays) for every
candidate waveform, we replay pre-computed ``.rpk`` replay packs. A pack is fixed; the waveform g(t) is
the optimization variable; the diffusion-weighted signal is a matmul on the pack's compressed
trajectory (``dmipy_sim.replay``), and its gradient w.r.t. the waveform is closed-form — so waveform
design is gradient descent through the stored Monte-Carlo substrate, with no autodiff (this package is
NumPy/SciPy-only by design: fully numpy-analytic derivatives → SciPy L-BFGS-B, per the repo philosophy).

First use case: **shape/size discrimination** — given two packs (e.g. a cylinder and a sphere, or two
diameters), find the deliverable waveform that maximally separates their signals, ``max_g |E_A - E_B|``.

The signal of one 1-axis waveform ``g(t)`` (direction ``d`` fixed) against a pack is dmipy-sim's own replay
forward: ``phi_i = C_i . W(g)`` with ``C`` the pack's position coefficients and ``W(g) = M g`` the waveform's
mode-space projection (``dmipy_sim.replay.compile_scheme``, LINEAR in ``g``), ``E = |<w exp(i phi)>| / sum(w)``,
and the analytic gradient chains back through ``M``:

    dE/dW = (1/|S|) Re( conj(S) * i * <w exp(i phi) C> ) / W0,   dE/dg = M^T dE/dW.

Substrate set: the Substrate Commons canonical replay dataset (cylinders / spheres / planes,
0.1-20 um). Orientation convention: anisotropic packs (cylinder/plane) are replayed in their canonical
frame (cylinder axis = z), i.e. the design gradient is perpendicular to the axis unless a direction is
given — the discriminating axis for shape/size contrast.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["load_pack", "design_discriminating_waveform", "DiscriminationResult"]

_MIN_SLEW_PENALTY = 1e-3


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

    def to_sequence(self):
        """The designed waveform as a dmipy-sim ``ScannerSequence``: a self-refocusing gradient echo (no pulse
        is folded; the structural refocusing of the design makes it refocus on its own)."""
        from dmipy_sim.sequences import from_waveform
        return from_waveform(np.asarray(self.G, np.float32)[None], self.dt, self.direction[None])


class _PackForward:
    """Analytic E(g) and dE/dg for one pack along a fixed direction: dmipy-sim's own replay forward.

    The pack replays a waveform through its compiled scheme, ``phi_i = C_i . W(g)`` with ``C`` the pack's
    position coefficients (``dmipy_sim.replay.compression.read_position_coeffs``) and ``W(g)`` the waveform's
    mode-space projection (``dmipy_sim.replay.compile_scheme``), which is LINEAR in ``g``: ``W(g) = M g`` with
    ``M`` the compiled unit impulses along the direction. So ``E(g)`` is exactly what ``pack.replay`` returns
    for the same waveform, and ``dE/dg = M^T dE/dW`` is closed-form -- no autodiff, no second copy of the
    replay mathematics here.
    """

    def __init__(self, pack, direction, gamma):
        from dmipy_sim.replay import compile_scheme
        from dmipy_sim.replay.compression import read_position_coeffs
        self.dt = float(pack.dt)
        self.n_t = int(pack.n_t)
        d = np.asarray(direction, np.float64)
        C = read_position_coeffs(pack.arrays, dtype=np.float64)       # (n_walkers, K+2, 3)
        self.Cflat = C.reshape(C.shape[0], -1)                       # (n_walkers, 3(K+2))
        basis = np.eye(self.n_t)[:, :, None] * d[None, None, :]      # the unit impulses along d, (n_t, n_t, 3)
        self.M = compile_scheme(basis, self.dt, pack.K, gamma, n_t=self.n_t)   # (3(K+2), n_t): W(g) = M g
        self.w = np.asarray(pack.spin_weights, np.float64)
        self.W0 = self.w.sum()

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
                                   te=None, n_basis=16, n_restarts=4, maxiter=300,
                                   seed=0, refocus_weight=50.0, slew_weight=_MIN_SLEW_PENALTY):
    """Design a single deliverable gradient waveform that maximally discriminates ``pack_a`` from
    ``pack_b`` (both :class:`dmipy_sim.replay.ReplayPack`), i.e. ``max_g |E_A(g) - E_B(g)|``.

    The 1-axis waveform g(t) is a smooth low-order cosine (DCT) synthesis of ``n_basis`` modes (no DC
    mode → refocusing is structural; band-limited → deliverable — a free per-sample optimizer wanders
    into a high-frequency OGSE-like waveform with high nominal b but ~zero contrast). Amplitude bounded
    (``|g| <= G_max`` via tanh). Optimized by SciPy L-BFGS-B with the analytic gradient (no autodiff),
    warm-started from the best plain PGSE (a cold start sits at ~zero contrast gradient), multi-restart.

    ``limits`` (a :class:`~dmipy_sim.acquisition.scanners.ScannerLimits`, or anything ``ScannerLimits.of``
    resolves) bounds the amplitude (``|g| <= G_max`` via tanh) and adds a soft slew penalty at its slew rate so
    the waveform is scanner-deliverable (the smooth basis already band-limits slew; this bounds it explicitly).
    ``te`` restricts the encoding to a window ``[0, te]`` (the waveform is zero after; the echo forms at ``te`` --
    the pack's TE-prefix property), which bounds ``b`` to a realistic range. Both packs must share the save
    grid (``dt``, ``n_t``); ``K`` may differ. Returns a :class:`DiscriminationResult`.
    """
    from dmipy_sim.acquisition.scanners import ScannerLimits
    from dmipy_sim.acquisition.waveforms import b_from_gradient
    from dmipy_sim.constants import GAMMA
    limits = ScannerLimits.of(limits)
    G_max, slew_max = float(limits.G_max), float(limits.slew_max)

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
    mask[-1] = 0.0                                                # the readout sample acts over nothing

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
    # the soft refocusing term leaves q(TE) at ~1e-3 of its peak; the sequence a designer hands on must refocus
    # exactly, so the net moment is removed along a smooth half-sine over the window (zero at both edges, so
    # no step is introduced) -- a change below the optimizer's own tolerance -- and the contrast reported is
    # that of the waveform actually returned
    on = mask > 0
    h = np.where(on, np.sin(np.pi * (np.arange(n_t) + 0.5) / max(1, on.sum())), 0.0)
    g_final = g_final - (g_final.sum() / h.sum()) * h
    Ea = fa.E_and_grad(g_final)[0]; Eb = fb.E_and_grad(g_final)[0]
    G = g_final[:, None] * d[None, :]
    max_slew = float(np.abs(np.diff(g_final) / dt).max())
    return DiscriminationResult(G=G, dt=dt, direction=d, contrast=abs(Ea - Eb),
                                E_A=Ea, E_B=Eb, b_value=float(b_from_gradient(G[None], dt)[0]),
                                te=(te if te is not None else (n_t - 1) * dt), max_slew=max_slew)
