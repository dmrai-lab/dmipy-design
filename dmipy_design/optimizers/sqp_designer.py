"""NOW-style direct-gradient + SQP waveform designer (LTE), for comparison with the
augmented-Lagrangian designer in ``waveform_designer``.

Ports the structure of NOW (Sjölund et al., J. Magn. Reson. 261 (2015) 157-168;
github.com/jsjol/NOW): optimize the GRADIENT SAMPLES DIRECTLY (no nonlinear squash),
maximize the b-value b = gᵀQ g (a smoothing quadratic form), and hand the hardware/physics
constraints to a sequential-quadratic-programming solver (SciPy ``SLSQP`` — the method
NOW's Python port defaults to) which keeps them on their boundaries.  The AL designer
instead reparameterizes the box through ``raw → tanh → cumsum → tanh → g`` and uses a
penalty + L-BFGS; that squash is the suspected source of high-frequency local minima at
high slew.  This module isolates the parameterization/solver variable so the two can be
compared at matched hardware.

Scope: LTE (rank-1, single axis) with optional M1/M2/Maxwell nulling — the case under
comparison.  Objective and all constraints (+ their exact Jacobians) come from JAX; SLSQP
does the constrained optimization.  Off-region samples are fixed to 0 (the 180 / lead-in /
readout), so g(0)=g(TE)=0 and the RF window are exact by construction.
"""
from __future__ import annotations
import numpy as np

try:
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize
    _OK = True
except ImportError:                                            # pragma: no cover
    _OK = False

GAMMA = 267.513e6  # rad/s/T


def design_waveform_sqp(b_delta=1.0, *, G_max=0.08, slew_rate_max=200.0, TE=0.060,
                        n_t=200, timing, null_M1=True, null_M2=True, maxwell=True,
                        n_restarts=12, seed=0, maxiter=300, init_g=None):
    """NOW-style direct-g + SLSQP LTE design.  Returns a dict with G (n_t,3) and metrics."""
    if not _OK:
        raise ImportError("JAX + SciPy required for design_waveform_sqp.")
    if abs(b_delta - 1.0) > 1e-6:
        raise NotImplementedError("sqp_designer currently covers LTE (b_delta=1) only.")

    slew_off, echo_idx = timing.masks(TE, n_t)
    enc = np.asarray(slew_off)[:, 0] > 0.5                     # encoding samples (g free)
    dt = TE / (n_t - 1)
    s = np.where(np.arange(n_t) < echo_idx, 1.0, -1.0)         # spin-echo sign
    t = np.arange(n_t) * dt
    free = np.where(enc)[0]
    nf = len(free)
    s_j, t_j = jnp.asarray(s[free]), jnp.asarray(t[free])      # signs/times at free samples

    # everything is expressed on the FREE (encoding) samples xf; the off-regions are 0, so
    # g(0)=g(TE)=0 and the RF window hold exactly.  q is the running effective dephasing.
    def q_of(xf):
        return GAMMA * jnp.cumsum(s_j * xf) * dt               # (nf,) rad/m on free samples
    def negb(xf):                                              # maximize b -> minimize -b
        q = q_of(xf); return -jnp.sum(q ** 2) * dt
    # equality constraints g(xf)=0 (refocus + selected moments + Maxwell)
    def c_refoc(xf):  return jnp.sum(s_j * xf) * dt            # q(TE)=0  (M0)
    def c_m1(xf):     return jnp.sum(t_j * s_j * xf) * dt
    def c_m2(xf):     return jnp.sum(t_j ** 2 * s_j * xf) * dt
    def c_mxwl(xf):   return jnp.sum(s_j * xf ** 2) * dt / (G_max ** 2 * TE)  # ∫s·g² scaled O(1)

    b_scale = (GAMMA * G_max) ** 2 * TE ** 3 / 50.0
    jit = jax.jit
    f_val, f_jac = jit(negb), jit(jax.grad(negb))
    eqs = [(c_refoc, "refoc")]
    if null_M1: eqs.append((c_m1, "m1"))
    if null_M2: eqs.append((c_m2, "m2"))
    if maxwell: eqs.append((c_mxwl, "mxwl"))
    cons = [{"type": "eq", "fun": jit(fn), "jac": jit(jax.grad(fn))} for fn, _ in eqs]

    # slew inequality on the full (zero-padded) waveform: |Δg|/dt ≤ sMax (SLSQP ineq ≥ 0)
    full_idx = jnp.asarray(free)
    def g_full(xf):
        return jnp.zeros(n_t).at[full_idx].set(xf)
    def slew_margin(xf):
        d = jnp.diff(g_full(xf)) / dt
        return (slew_rate_max - jnp.abs(d)) / slew_rate_max    # (n_t-1,) ≥ 0, normalized O(1)
    cons.append({"type": "ineq", "fun": jit(slew_margin), "jac": jit(jax.jacfwd(slew_margin))})
    bounds = [(-G_max, G_max)] * nf                            # |g| ≤ G_max box

    # diverse warm starts.  SLSQP is LOCAL, so coverage decides whether we find the high-b
    # clean basin (NOW's solution) or a low-b local max: full-amplitude bipolars (toward NOW's
    # 1-2-lobe shape) at several pre/post split fractions, plus low-frequency random combos.
    rng = np.random.default_rng(seed)
    edge = np.sin(np.linspace(0.0, np.pi, nf))                 # 0 at the window edges
    inits = []
    for frac in (0.45, 0.5, 0.55, 0.6):
        h = max(1, int(nf * frac))
        inits.append(edge * np.concatenate([-np.ones(h), np.ones(nf - h)]) * G_max)
    for _ in range(max(0, n_restarts - len(inits))):           # low-frequency random
        kk = int(rng.integers(1, 4)); ph = rng.uniform(0, np.pi, kk); am = rng.uniform(0.4, 1.0, kk)
        x = sum(a * np.sin((j + 1) * np.linspace(0, np.pi, nf) + p) for j, (a, p) in enumerate(zip(am, ph)))
        inits.append(edge * x / (np.max(np.abs(x)) + 1e-9) * G_max)
    if init_g is not None:                                     # optional warm start (e.g. NOW wf)
        ig = np.asarray(init_g, float); ig = ig[:, 0] if ig.ndim == 2 else ig
        inits.insert(0, ig[free])

    def solve(x0, iters):                                      # scaled objective: O(1) vs O(1) cons
        return minimize(lambda x: float(f_val(x)) / b_scale, x0,
                        jac=lambda x: np.asarray(f_jac(x)) / b_scale,
                        method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": iters, "ftol": 1e-9}).x

    def evaluate(xf):
        G = np.zeros((n_t, 3)); G[free, 0] = xf
        q = np.asarray(q_of(jnp.asarray(xf)))
        b = float(np.sum(q ** 2) * dt)
        refoc = abs(float(c_refoc(jnp.asarray(xf)))) * GAMMA / (np.sqrt(np.max(q ** 2)) + 1e-30)
        slew_ok = float(np.max(np.abs(np.diff(G[:, 0]) / dt)))
        amp_ok = float(np.max(np.abs(xf)))
        mx = abs(float(c_mxwl(jnp.asarray(xf))))
        m1 = abs(float(c_m1(jnp.asarray(xf)))) / (G_max * TE ** 2)
        m2 = abs(float(c_m2(jnp.asarray(xf)))) / (G_max * TE ** 3)
        feas = (refoc < 1e-2 and slew_ok <= slew_rate_max * 1.02 and amp_ok <= G_max * 1.01
                and (not maxwell or mx < 2e-2) and (not null_M1 or m1 < 5e-2)
                and (not null_M2 or m2 < 5e-2))
        return dict(G=G, xf=xf, b_value=b, refoc=refoc, max_slew=slew_ok, max_amp=amp_ok,
                    maxwell_index=mx, m1_index=m1, m2_index=m2, feasible=feas,
                    echo_idx=int(echo_idx), dt=dt)

    best = last = None
    for x0 in inits:
        last = evaluate(solve(x0, maxiter))
        if last["feasible"] and (best is None or last["b_value"] > best["b_value"]):
            best = last
    if best is not None:                                       # polish the winner (2x iters)
        pol = evaluate(solve(best["xf"], maxiter * 2))
        if pol["feasible"] and pol["b_value"] >= best["b_value"] - 1.0:
            best = pol
    return best if best is not None else last
