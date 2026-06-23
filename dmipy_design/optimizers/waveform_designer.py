"""
Tensor-valued diffusion gradient-waveform designer (NOW-style, JAX/GPU).

Generates a hardware-realizable *physical* gradient waveform ``g(t)`` with a
finite-180 spin echo built in, achieving a target b-tensor shape
(``b_delta``: LTE = 1, STE = 0, PTE = -0.5, or any value in [-0.5, 1]) while
maximizing the b-value under Prisma-class hardware (G_max, slew, TE).  Subsumes
the analytic placeholder STE encoding (``B = (b/3) I`` asserted, no waveform).

Physics (the success metrics double as the constraint functions)
----------------------------------------------------------------
Effective dephasing with the 180 sign flip folded in::

    s(t) = +1 for t < t_180,  -1 for t >= t_180
    q(t) = gamma * cumsum(s * g) * dt                  (rad/m)
    B    = integral q q^T dt                           (s/m^2, the b-tensor)
    b    = trace(B)
    b_delta = (lam_unique - mean(other two)) / b       (1 LTE, 0 STE, -0.5 PTE)

Refocusing (echo): ``q(TE) = 0`` — intrinsic here because the sign flip enters
``q`` (this is exactly what an effective-only waveform, e.g. OPTICUBE, lacks).

Parameterization (both hardware limits structural)
--------------------------------------------------
The optimization variable is the slew rate; a radial tanh bounds its norm and an
RF mask gates it off across the 180; integrating gives g, then a second radial
tanh caps the amplitude::

    slew(t) = S_max * tanh(|raw|)/|raw| * raw * rf_off    -> |dg/dt| ~<= S_max
    g_raw   = dt * cumsum(slew)                            -> g(0) = 0
    g(t)    = G_max * tanh(|g_raw|/G_max)/|g_raw| * g_raw  -> |g| <= G_max

so the optimizer can drive b to the amplitude wall without ever violating the
box limits.  The amplitude squash slightly perturbs the slew, so a residual slew
constraint is kept.  Remaining equality constraints — refocus ``q(TE)=0``,
``g(TE)=0``, ``g=0`` across the RF window, residual slew, and the b-tensor shape
— are driven to zero by an augmented Lagrangian; b is the objective.

Solver
------
JAX augmented Lagrangian: inner jaxopt L-BFGS minimizes ``-b + λ·c + (μ/2)|c|²``
(vmapped over restarts on GPU), outer loop updates multipliers ``λ += μ c`` and
grows ``μ``.  Best feasible restart wins.  Decoupling feasibility (multipliers)
from the objective avoids the penalty-weight balancing that collapses b.
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
N_CONSTRAINTS = 5  # (refocus, shape, g(TE)=0, RF-window, residual slew)


# ===========================================================================
# Differentiable physics — these ARE the Tier-1 success metrics
# ===========================================================================

def _echo_sign(n_t: int, echo_idx: int):
    return jnp.where(jnp.arange(n_t) < echo_idx, 1.0, -1.0)[:, None]   # (n_t,1)


def effective_q(g, dt: float, echo_idx: int):
    """Effective dephasing q(t) (rad/m), with the 180 sign flip folded in."""
    s = _echo_sign(g.shape[0], echo_idx)
    return GAMMA * dt * jnp.cumsum(s * g, axis=0)                       # (n_t,3)


def b_tensor(g, dt: float, echo_idx: int):
    """b-tensor B = ∫ q qᵀ dt  (3,3), matching dmipy_sim._btensor_from_waveform.

    Uses an explicit outer-product sum, not einsum/matmul: the small (3×3)
    contraction triggers an XLA "too small divisible part of the contracting
    dimension" failure inside ``jax.vmap`` on GPU.
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
    """Axially-symmetric target eigenvalues (ascending) for a given shape + b."""
    lam_par = (b / 3.0) * (1.0 + 2.0 * b_delta)
    lam_perp = (b / 3.0) * (1.0 - b_delta)
    return jnp.sort(jnp.array([lam_par, lam_perp, lam_perp]))


def _shape_penalty(B, b_delta):
    """Frame- and magnitude-invariant deviation of B's spectrum from the target."""
    w = jnp.sort(jnp.linalg.eigvalsh(B))
    b = jnp.sum(w) + 1e-30
    return jnp.sum((w - _target_eigs(b_delta, b)) ** 2) / (b ** 2)


# ===========================================================================
# Parameterization + constraint vector
# ===========================================================================

def _waveform_from_raw(raw, dt, s_max, g_max, slew_off_mask):
    """raw (n_t,3) -> slew- and amplitude-bounded, RF-gated physical g (n_t,3)."""
    rn = jnp.sqrt(jnp.sum(raw ** 2, axis=1, keepdims=True) + 1e-12)
    slew = s_max * (jnp.tanh(rn) / rn) * raw * slew_off_mask            # |slew|<=S_max
    g_raw = dt * jnp.cumsum(slew, axis=0)                              # g(0)=0
    gn = jnp.sqrt(jnp.sum(g_raw ** 2, axis=1, keepdims=True) + 1e-12)
    return g_max * (jnp.tanh(gn / g_max) / gn) * g_raw                 # |g|<=G_max


def _b_and_constraints(raw, dt, echo_idx, s_max, g_max, b_delta, rf_mask, slew_off_mask):
    """Return (b, c) where c is the (5,) vector of equality violations (want 0)."""
    g = _waveform_from_raw(raw, dt, s_max, g_max, slew_off_mask)
    B = b_tensor(g, dt, echo_idx)
    b = b_value(B)
    q = effective_q(g, dt, echo_idx)
    q2 = jnp.sum(q ** 2, axis=1)
    gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=1) + 1e-30)
    slew = jnp.diff(g, axis=0) / dt
    slewnorm = jnp.sqrt(jnp.sum(slew ** 2, axis=1) + 1e-30)
    c = jnp.array([
        jnp.sum(q[-1] ** 2) / (jnp.max(q2) + 1e-30),          # refocus q(TE)=0
        _shape_penalty(B, b_delta),                          # b-tensor shape
        jnp.sum(g[-1] ** 2) / g_max ** 2,                    # g(TE)=0
        jnp.mean((gnorm * rf_mask) ** 2) / g_max ** 2,       # g=0 across RF window
        jnp.max(jnp.maximum(slewnorm - s_max, 0.0) ** 2) / s_max ** 2,  # residual slew
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
    feasible: bool
    report: dict

    def effective_G(self) -> np.ndarray:
        """Effective (sign-folded) gradient — what dmipy-sim's b-from-waveform eats."""
        s = np.where(np.arange(self.G.shape[0]) < self.echo_idx, 1.0, -1.0)[:, None]
        return self.G * s

    def to_sim_waveform(self, b_target: float | None = None):
        """Build a dmipy-sim ``Waveform`` (effective gradient, echo at TE) for MC.

        Subsumes the analytic STE placeholder: instead of asserting ``B=(b/3)I``
        and a guessed b, this hands the real hardware-realizable gradient to the
        forward/MC pipeline.  ``b_target`` (s/m²) optionally rescales the
        amplitude (b ∝ |G|²; the b-tensor shape and refocusing are invariant).
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
        Target b-tensor shape: 1 = LTE, 0 = STE, -0.5 = PTE (any value in
        [-0.5, 1] allowed).
    G_max, slew_rate_max, TE : float
        Hardware constraints (T/m, T/m/s, s).  Defaults are 3T Prisma.
    n_t : int
        Waveform samples (dt = TE / (n_t - 1)).
    echo_frac : float
        Fractional position of the 180 RF within TE (0.5 = symmetric).
    rf_duration : float
        Finite-180 RF window (s); the gradient is held off across it.
    n_restarts : int
        Random restarts, vmapped on GPU; best feasible wins.
    inner_maxiter, n_outer : int
        L-BFGS iterations per inner solve, and augmented-Lagrangian outer rounds.

    Returns
    -------
    WaveformDesign
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX + jaxopt are required for design_waveform.")

    dt = TE / (n_t - 1)
    echo_idx = int(round(echo_frac * (n_t - 1)))

    t = np.arange(n_t) * dt
    rf_mask_np = (np.abs(t - echo_idx * dt) <= 0.5 * rf_duration).astype(np.float64)
    rf_mask = jnp.asarray(rf_mask_np)
    slew_off_mask = jnp.asarray(1.0 - rf_mask_np)[:, None]
    b_scale = (GAMMA * G_max) ** 2 * TE ** 3 / 50.0       # ~ achievable LTE b

    # --- structured + random initial guesses (warm starts for the 3D landscape) ---
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
                                      b_delta, rf_mask, slew_off_mask)

    # --- augmented Lagrangian, per-restart multipliers, vmapped on GPU ---
    lam = jnp.zeros((n_restarts, N_CONSTRAINTS))
    mu = 10.0
    for outer in range(n_outer):
        def al_loss(r, lam_r, mu_s):
            b, c = bc(r)
            return -b / b_scale + jnp.sum(lam_r * c) + 0.5 * mu_s * jnp.sum(c ** 2)

        solver = LBFGS(fun=al_loss, maxiter=inner_maxiter, tol=1e-8,
                       jit=True, history_size=10)
        raw = jax.vmap(lambda r, l: solver.run(r, l, mu).params)(raw, lam)
        cs = jax.vmap(lambda r: bc(r)[1])(raw)            # (n_restarts, 5)
        lam = lam + mu * cs                               # multiplier update
        mu = min(mu * 4.0, 1e6)
        if verbose:
            print(f"  outer {outer}: mu={mu:.0e}  max|c| best="
                  f"{float(jnp.min(jnp.max(cs, axis=1))):.2e}")

    # --- evaluate Tier-1 metrics, pick best feasible (max b) ---
    def _metrics(r):
        g = _waveform_from_raw(r, dt, slew_rate_max, G_max, slew_off_mask)
        B = b_tensor(g, dt, echo_idx)
        q = effective_q(g, dt, echo_idx)
        gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=1) + 1e-30)
        slew = jnp.sqrt(jnp.sum((jnp.diff(g, axis=0) / dt) ** 2, axis=1) + 1e-30)
        refoc = jnp.sqrt(jnp.sum(q[-1] ** 2)) / (jnp.sqrt(jnp.max(jnp.sum(q ** 2, axis=1))) + 1e-30)
        return jnp.array([b_value(B), b_delta_of(B), jnp.max(gnorm),
                          jnp.max(slew), refoc, jnp.max(gnorm * rf_mask)]), g

    metrics, gs = jax.vmap(_metrics)(raw)
    metrics = np.asarray(metrics)
    gs = np.asarray(gs)
    b_all, bd_all, amp_all, slew_all, refoc_all, win_all = metrics.T

    feas = ((np.abs(bd_all - b_delta) < 2e-2) & (amp_all <= G_max * 1.01)
            & (slew_all <= slew_rate_max * 1.02) & (refoc_all < 1e-2)
            & (win_all <= G_max * 0.02))
    if feas.any():
        cand = np.where(feas)[0]
        best = int(cand[np.argmax(b_all[cand])])
        feasible = True
    else:
        viol = (np.maximum(np.abs(bd_all - b_delta) - 2e-2, 0)
                + np.maximum(amp_all - G_max, 0) / G_max
                + np.maximum(slew_all - slew_rate_max, 0) / slew_rate_max
                + refoc_all + win_all / G_max)
        best = int(np.argmin(viol))
        feasible = False

    return WaveformDesign(
        G=gs[best].astype(np.float64), dt=dt, echo_idx=echo_idx,
        b_value=float(b_all[best]), b_delta=float(bd_all[best]),
        b_delta_target=float(b_delta), max_amplitude=float(amp_all[best]),
        max_slew=float(slew_all[best]), refocus_residual=float(refoc_all[best]),
        rf_window_leak=float(win_all[best]), feasible=bool(feasible),
        report={'n_restarts': n_restarts, 'n_feasible': int(feas.sum()),
                'b_scale': float(b_scale), 'dt': dt,
                'b_feasible_max': float(b_all[feas].max()) if feas.any() else None},
    )
