"""
Tensor-valued diffusion gradient-waveform designer (NOW-style, JAX/GPU).

Generates a hardware-realizable *physical* gradient waveform ``g(t)`` with a
finite-180 spin echo built in, achieving a target b-tensor shape
(``b_delta``: LTE = 1, STE = 0, PTE = -0.5, or any value in [-0.5, 1]) while
maximizing the b-value under Prisma-class hardware (G_max, slew, TE).  Subsumes
the analytic placeholder STE encoding (``B = (b/3) I`` asserted, no waveform).

Optional, toggleable constraints (the full NOW objective set):
  * ``null_M1``  — velocity compensation:      ∫ t · g_eff dt = 0
  * ``null_M2``  — acceleration compensation:  ∫ t² · g_eff dt = 0
  * ``maxwell``  — concomitant-field (Maxwell) compensation:
                   M = ∫ s(t) · g gᵀ dt = 0   (Szczepankiewicz 2019)
Each flag adds its constraint to the augmented Lagrangian; *all* of the moment /
Maxwell indices are always reported (whether constrained or not), so toggling a
flag shows exactly what changed (and what it cost in b).

Physics (the metrics double as the constraint functions)
--------------------------------------------------------
Effective dephasing with the 180 sign flip folded in::

    s(t) = +1 for t < t_180,  -1 for t >= t_180
    g_eff = s * g
    q(t) = gamma * cumsum(g_eff) * dt                  (rad/m)
    B    = integral q q^T dt                           (s/m^2, the b-tensor)
    b    = trace(B)
    M_k  = integral t^k g_eff dt                       (gradient moments)
    Maxw = integral s g g^T dt                         (concomitant matrix)

Refocusing (echo, M0): ``q(TE) = 0`` — intrinsic because the sign flip enters
``q`` (this is what an effective-only loaded waveform, e.g. OPTICUBE, lacks).

Parameterization (both hardware limits structural)
--------------------------------------------------
slew = S_max·tanh(|raw|)/|raw|·raw·rf_off  ->  |dg/dt| ~<= S_max (radial)
g_raw = dt·cumsum(slew)                     ->  g(0) = 0
g    = G_max·tanh(|g_raw|/G_max)/|g_raw|·g_raw  ->  |g| <= G_max
so b can be driven to the amplitude wall without violating the box.  The
amplitude squash perturbs the slew slightly, so a residual slew constraint is
kept.  Equality constraints (refocus, shape, g(TE)=0, RF-window, residual slew,
and any active M1/M2/Maxwell) are driven to zero by an augmented Lagrangian.

Solver
------
JAX augmented Lagrangian: inner jaxopt L-BFGS minimizes ``-b + λ·c + (μ/2)|c|²``
(vmapped over restarts on GPU), outer loop updates ``λ += μ c`` and grows ``μ``.
Decoupling feasibility (multipliers) from the objective avoids penalty-weight
balancing that otherwise collapses b.  Best feasible restart (max b) wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False

GAMMA = 267.513e6  # rad/s/T — matches dmipy_sim.constants.GAMMA

# Fixed constraint order; the AL uses a flag-selected subset, the report shows all.
CONSTRAINT_NAMES = ('refocus', 'shape', 'g(TE)=0', 'RF-window', 'slew',
                    'M1', 'M2', 'maxwell')
_BASE_CONSTRAINTS = (0, 1, 2, 3, 4)   # always active (validity)


# ===========================================================================
# Differentiable physics — these ARE the success metrics / constraint functions
# ===========================================================================

def _echo_sign(n_t: int, echo_idx: int):
    return jnp.where(jnp.arange(n_t) < echo_idx, 1.0, -1.0)[:, None]   # (n_t,1)


def effective_q(g, dt: float, echo_idx: int):
    """Effective dephasing q(t) (rad/m), with the 180 sign flip folded in."""
    s = _echo_sign(g.shape[0], echo_idx)
    return GAMMA * dt * jnp.cumsum(s * g, axis=0)                       # (n_t,3)


def b_tensor(g, dt: float, echo_idx: int):
    """b-tensor B = ∫ q qᵀ dt  (3,3), matching dmipy_sim._btensor_from_waveform.

    Explicit outer-product sum, not einsum/matmul: the small (3×3) contraction
    triggers an XLA "too small divisible part of the contracting dimension"
    failure inside ``jax.vmap`` on GPU.
    """
    q = effective_q(g, dt, echo_idx)
    return dt * jnp.sum(q[:, :, None] * q[:, None, :], axis=0)


def b_value(B):
    return jnp.trace(B)


def b_delta_of(B):
    """Normalized b-tensor anisotropy: 1 LTE, 0 STE, -0.5 PTE (frame-invariant)."""
    w = jnp.linalg.eigvalsh(B)
    b = jnp.sum(w) + 1e-30
    idx = jnp.argmax(jnp.abs(w - b / 3.0))
    lam_u = w[idx]
    return (lam_u - (b - lam_u) / 2.0) / b


def _target_eigs(b_delta, b):
    lam_par = (b / 3.0) * (1.0 + 2.0 * b_delta)
    lam_perp = (b / 3.0) * (1.0 - b_delta)
    return jnp.sort(jnp.array([lam_par, lam_perp, lam_perp]))


def _shape_penalty(B, b_delta):
    w = jnp.sort(jnp.linalg.eigvalsh(B))
    b = jnp.sum(w) + 1e-30
    return jnp.sum((w - _target_eigs(b_delta, b)) ** 2) / (b ** 2)


# ===========================================================================
# Parameterization + full constraint vector
# ===========================================================================

def _waveform_from_raw(raw, dt, s_max, g_max, slew_off_mask):
    """raw (n_t,3) -> slew- and amplitude-bounded, RF-gated physical g (n_t,3)."""
    rn = jnp.sqrt(jnp.sum(raw ** 2, axis=1, keepdims=True) + 1e-12)
    slew = s_max * (jnp.tanh(rn) / rn) * raw * slew_off_mask            # |slew|<=S_max
    g_raw = dt * jnp.cumsum(slew, axis=0)                              # g(0)=0
    gn = jnp.sqrt(jnp.sum(g_raw ** 2, axis=1, keepdims=True) + 1e-12)
    return g_max * (jnp.tanh(gn / g_max) / gn) * g_raw                 # |g|<=G_max


def _b_and_constraints(raw, dt, echo_idx, s_max, g_max, b_delta, rf_mask,
                       slew_off_mask, t_arr, TE):
    """Return (b, c) where c is the full (8,) vector of equality violations.

    Each entry is a normalized squared violation (0 = satisfied); the AL selects
    an active subset and the report reads them all.
    """
    g = _waveform_from_raw(raw, dt, s_max, g_max, slew_off_mask)
    B = b_tensor(g, dt, echo_idx)
    b = b_value(B)
    s = _echo_sign(g.shape[0], echo_idx)                  # (n_t,1)
    geff = s * g                                          # effective gradient
    q = GAMMA * dt * jnp.cumsum(geff, axis=0)
    q2 = jnp.sum(q ** 2, axis=1)
    gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=1) + 1e-30)
    slewnorm = jnp.sqrt(jnp.sum((jnp.diff(g, axis=0) / dt) ** 2, axis=1) + 1e-30)

    # gradient moments (velocity/acceleration) and the Maxwell matrix
    M1 = jnp.sum(t_arr[:, None] * geff, axis=0) * dt                  # (3,)
    M2 = jnp.sum((t_arr ** 2)[:, None] * geff, axis=0) * dt           # (3,)
    Mmx = dt * jnp.sum(s[:, :, None] * g[:, :, None] * g[:, None, :], axis=0)  # (3,3)

    c = jnp.array([
        jnp.sum(q[-1] ** 2) / (jnp.max(q2) + 1e-30),                  # 0 refocus (M0)
        _shape_penalty(B, b_delta),                                  # 1 shape
        jnp.sum(g[-1] ** 2) / g_max ** 2,                            # 2 g(TE)=0
        jnp.mean((gnorm * rf_mask) ** 2) / g_max ** 2,               # 3 RF window
        jnp.max(jnp.maximum(slewnorm - s_max, 0.0) ** 2) / s_max ** 2,  # 4 slew
        jnp.sum(M1 ** 2) / (g_max * TE ** 2) ** 2,                   # 5 M1 (velocity)
        jnp.sum(M2 ** 2) / (g_max * TE ** 3) ** 2,                   # 6 M2 (accel)
        jnp.sum(Mmx ** 2) / (g_max ** 2 * TE) ** 2,                  # 7 Maxwell
    ])
    return b, c


# ===========================================================================
# Public result + designer
# ===========================================================================

@dataclass
class WaveformDesign:
    """An optimized tensor-valued gradient waveform + its acceptance metrics."""
    G: np.ndarray            # (n_t, 3) physical gradient, T/m
    dt: float
    echo_idx: int
    b_value: float           # s/m²
    b_delta: float
    b_delta_target: float
    max_amplitude: float     # max |G|, T/m
    max_slew: float          # max |dG/dt|, T/m/s
    refocus_residual: float  # |q(TE)| / max|q|
    rf_window_leak: float    # max |G| inside the RF window, T/m
    m1_index: float          # |M1| / (G_max·TE²)   — 0 when velocity-compensated
    m2_index: float          # |M2| / (G_max·TE³)   — 0 when accel-compensated
    maxwell_index: float     # ||M||_F / (G_max²·TE) — 0 when Maxwell-compensated
    active_constraints: tuple
    feasible: bool
    report: dict

    def effective_G(self) -> np.ndarray:
        """Effective (sign-folded) gradient — what dmipy-sim's b-from-waveform eats."""
        s = np.where(np.arange(self.G.shape[0]) < self.echo_idx, 1.0, -1.0)[:, None]
        return self.G * s

    def to_sim_waveform(self, b_target: float | None = None):
        """Build a dmipy-sim ``Waveform`` (effective gradient, echo at TE) for MC.

        Subsumes the analytic STE placeholder: hands the real hardware-realizable
        gradient to the forward/MC pipeline.  ``b_target`` (s/m²) optionally
        rescales the amplitude (b ∝ |G|²; b-tensor shape and refocusing invariant).
        """
        from dmipy_sim.waveforms import Waveform
        G = self.effective_G()
        if b_target is not None:
            G = G * np.sqrt(b_target / self.b_value)
        return Waveform(G=G[None].astype(np.float32), dt=self.dt,
                        echo_idx=self.G.shape[0] - 1)


def design_waveform(
    b_delta: float,
    *,
    G_max: float = 0.08,
    slew_rate_max: float = 200.0,
    TE: float = 0.080,
    n_t: int = 256,
    echo_frac: float = 0.5,
    rf_duration: float = 0.004,
    null_M1: bool = False,
    null_M2: bool = False,
    maxwell: bool = False,
    n_restarts: int = 48,
    seed: int = 0,
    inner_maxiter: int = 300,
    n_outer: int = 14,
    verbose: bool = False,
) -> WaveformDesign:
    """Design a hardware-realizable tensor-valued spin-echo gradient waveform.

    Parameters
    ----------
    b_delta : float
        Target b-tensor shape: 1 = LTE, 0 = STE, -0.5 = PTE.
    G_max, slew_rate_max, TE : float
        Hardware constraints (T/m, T/m/s, s).  Defaults are 3T Prisma.
    null_M1, null_M2 : bool
        Add velocity (M1) / acceleration (M2) moment-nulling constraints.
    maxwell : bool
        Add concomitant-field (Maxwell) compensation: ∫ s·g·gᵀ dt = 0.
    n_t, echo_frac, rf_duration : see module docstring.
    n_restarts, seed, inner_maxiter, n_outer : solver controls.

    Returns
    -------
    WaveformDesign
        ``.m1_index / .m2_index / .maxwell_index`` are always reported (0 ≈
        compensated), regardless of which flags were active — so toggling a flag
        shows what changed and what it cost in ``.b_value``.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX + jaxopt are required for design_waveform.")

    dt = TE / (n_t - 1)
    echo_idx = int(round(echo_frac * (n_t - 1)))
    t = np.arange(n_t) * dt
    rf_mask_np = (np.abs(t - echo_idx * dt) <= 0.5 * rf_duration).astype(np.float64)
    rf_mask = jnp.asarray(rf_mask_np)
    slew_off_mask = jnp.asarray(1.0 - rf_mask_np)[:, None]
    t_arr = jnp.asarray(t)
    b_scale = (GAMMA * G_max) ** 2 * TE ** 3 / 50.0       # ~ achievable LTE b

    # active constraint subset (validity always on; flags add M1/M2/Maxwell)
    active = list(_BASE_CONSTRAINTS)
    if null_M1:
        active.append(5)
    if null_M2:
        active.append(6)
    if maxwell:
        active.append(7)
    active_idx = jnp.asarray(active)
    active_names = tuple(CONSTRAINT_NAMES[i] for i in active)
    n_active = len(active)

    # structured (q-MAS-like) + random warm starts
    key = jax.random.PRNGKey(seed)
    kf, kp, ka, kx, kn = jax.random.split(key, 5)
    tt = (jnp.arange(n_t) / (n_t - 1))[None, :, None]
    freqs = jax.random.randint(kf, (n_restarts, 1, 3), 1, 7)
    phase = jax.random.uniform(kp, (n_restarts, 1, 3), minval=0.0, maxval=2 * np.pi)
    amp = jax.random.uniform(ka, (n_restarts, 1, 3), minval=1.0, maxval=3.0)
    axis_w = jax.random.uniform(kx, (n_restarts, 1, 3), minval=0.2, maxval=1.0)
    raw = (amp * axis_w * jnp.sin(2 * np.pi * freqs * tt + phase)
           + 0.3 * jax.random.normal(kn, (n_restarts, n_t, 3)))

    bc = lambda r: _b_and_constraints(r, dt, echo_idx, slew_rate_max, G_max,
                                      b_delta, rf_mask, slew_off_mask, t_arr, TE)

    # --- augmented Lagrangian, per-restart multipliers, vmapped on GPU ---
    lam = jnp.zeros((n_restarts, n_active))
    mu = 10.0
    for outer in range(n_outer):
        def al_loss(r, lam_r, mu_s):
            b, c_all = bc(r)
            c = c_all[active_idx]
            return -b / b_scale + jnp.sum(lam_r * c) + 0.5 * mu_s * jnp.sum(c ** 2)

        solver = LBFGS(fun=al_loss, maxiter=inner_maxiter, tol=1e-8,
                       jit=True, history_size=10)
        raw = jax.vmap(lambda r, l: solver.run(r, l, mu).params)(raw, lam)
        cs = jax.vmap(lambda r: bc(r)[1][active_idx])(raw)
        lam = lam + mu * cs
        mu = min(mu * 4.0, 1e6)
        if verbose:
            print(f"  outer {outer}: mu={mu:.0e}  max|c| best="
                  f"{float(jnp.min(jnp.max(cs, axis=1))):.2e}")

    # --- evaluate all metrics, pick best feasible (max b) ---
    def _metrics(r):
        g = _waveform_from_raw(r, dt, slew_rate_max, G_max, slew_off_mask)
        B = b_tensor(g, dt, echo_idx)
        q = effective_q(g, dt, echo_idx)
        gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=1) + 1e-30)
        slew = jnp.sqrt(jnp.sum((jnp.diff(g, axis=0) / dt) ** 2, axis=1) + 1e-30)
        refoc = jnp.sqrt(jnp.sum(q[-1] ** 2)) / (jnp.sqrt(jnp.max(jnp.sum(q ** 2, axis=1))) + 1e-30)
        _, c_all = bc(r)
        return jnp.array([b_value(B), b_delta_of(B), jnp.max(gnorm), jnp.max(slew),
                          refoc, jnp.max(gnorm * rf_mask),
                          jnp.sqrt(c_all[5]), jnp.sqrt(c_all[6]), jnp.sqrt(c_all[7])]), g

    metrics, gs = jax.vmap(_metrics)(raw)
    metrics = np.asarray(metrics)
    gs = np.asarray(gs)
    (b_all, bd_all, amp_all, slew_all, refoc_all, win_all,
     m1_all, m2_all, mx_all) = metrics.T

    feas = ((np.abs(bd_all - b_delta) < 2e-2) & (amp_all <= G_max * 1.01)
            & (slew_all <= slew_rate_max * 1.02) & (refoc_all < 1e-2)
            & (win_all <= G_max * 0.02))
    if null_M1:
        feas &= (m1_all < 5e-2)
    if null_M2:
        feas &= (m2_all < 5e-2)
    if maxwell:
        feas &= (mx_all < 5e-2)

    if feas.any():
        cand = np.where(feas)[0]
        best = int(cand[np.argmax(b_all[cand])])
        feasible = True
    else:
        viol = (np.maximum(np.abs(bd_all - b_delta) - 2e-2, 0)
                + np.maximum(amp_all - G_max, 0) / G_max
                + np.maximum(slew_all - slew_rate_max, 0) / slew_rate_max
                + refoc_all + win_all / G_max
                + (m1_all if null_M1 else 0) + (m2_all if null_M2 else 0)
                + (mx_all if maxwell else 0))
        best = int(np.argmin(viol))
        feasible = False

    return WaveformDesign(
        G=gs[best].astype(np.float64), dt=dt, echo_idx=echo_idx,
        b_value=float(b_all[best]), b_delta=float(bd_all[best]),
        b_delta_target=float(b_delta), max_amplitude=float(amp_all[best]),
        max_slew=float(slew_all[best]), refocus_residual=float(refoc_all[best]),
        rf_window_leak=float(win_all[best]), m1_index=float(m1_all[best]),
        m2_index=float(m2_all[best]), maxwell_index=float(mx_all[best]),
        active_constraints=active_names, feasible=bool(feasible),
        report={'n_restarts': n_restarts, 'n_feasible': int(feas.sum()),
                'b_scale': float(b_scale), 'dt': dt,
                'b_feasible_max': float(b_all[feas].max()) if feas.any() else None},
    )
