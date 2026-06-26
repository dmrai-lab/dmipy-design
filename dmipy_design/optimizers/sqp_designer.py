"""NOW-style direct-gradient + SQP waveform designer (LTE / PTE / STE, + OGSE), for
comparison with — and as a lower-risk alternative to — the augmented-Lagrangian designer in
``waveform_designer``.

Ports NOW's structure (Sjölund et al., J. Magn. Reson. 261 (2015) 157-168;
github.com/jsjol/NOW): optimize the GRADIENT SAMPLES DIRECTLY (no nonlinear squash), maximize
b = trace ∫q·qᵀ (a smoothing quadratic in g), and hand the hardware/physics constraints to
SciPy ``SLSQP`` (the SQP method NOW's Python port uses).  Objective and ALL constraint
Jacobians are exact via JAX autodiff — identical to NOW's hand-derived analytic gradients to
machine precision (autodiff IS the analytic gradient; nothing to port for accuracy).  At
matched constraint strictness this beats the AL designer on b, cleanliness, and feasibility
precision (the AL's tanh-squash + penalty admits high-frequency local minima at high slew).

Shapes (rank of the target b-tensor sets the number of encoding axes):
  * LTE (b_delta=1)   -> rank 1 (1 axis); shape automatic.
  * PTE (b_delta=-0.5) -> rank 2 (plane); the two in-plane eigenvalues forced equal.
  * STE (b_delta=0)   -> rank 3 (ball); all three eigenvalues forced equal (isotropy).
OGSE: any shape + ``spectral_freq`` adds the equality f_rms(g) = f_target -> a CLEAN
oscillation at the target frequency (SQP smoothness makes it a clean OGSE, not noise).

Off-region samples (lead-in / 180 / readout) are fixed to 0, so g(0)=g(TE)=0 and the RF
window hold exactly by construction; the slew constraint then forces a feasible ramp to 0 at
the encoding-window edges.  Maxwell (concomitant) nulling is optional and B0-tied (does not
generalize to >3 axes).  Formulation verified vs Szczepankiewicz 2019 (b=∫q·qᵀ,
Maxwell M=∫s·g·gᵀ, moments ∫tᵏ·g_eff).
"""
from __future__ import annotations
import numpy as np

try:
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize, NonlinearConstraint, Bounds
    _OK = True
except ImportError:                                            # pragma: no cover
    _OK = False

GAMMA = 267.513e6  # rad/s/T


def _rank_of(b_delta):
    if abs(b_delta - 1.0) < 1e-6:
        return 1                                               # LTE
    if abs(b_delta + 0.5) < 1e-6:
        return 2                                               # PTE (planar)
    return 3                                                   # STE (b_delta=0) / intermediate


def design_waveform_sqp(b_delta=1.0, *, G_max=0.08, slew_rate_max=200.0, TE=0.060,
                        n_t=200, timing, null_M1=True, null_M2=True, maxwell=True,
                        spectral_freq=None, n_restarts=16, seed=0, maxiter=400, init_g=None,
                        method="slsqp", slew_mode="vector", slew_param=False):
    """NOW-style direct-g + SLSQP design for LTE/PTE/STE (+OGSE).  Returns a dict with G
    (n_t,3), b_value, the constraint indices, and feasibility."""
    if not _OK:
        raise ImportError("JAX + SciPy required for design_waveform_sqp.")
    na = _rank_of(b_delta)                                     # number of encoding axes
    slew_off, echo = timing.masks(TE, n_t)
    enc = np.asarray(slew_off)[:, 0] > 0.5
    dt = TE / (n_t - 1)
    # erode the encoding mask by the ramp length so g is pinned to 0 a few samples inside each
    # window edge -> g ramps from 0 STRUCTURALLY (edge slew <= G_max/(n_ramp*dt) <= S_max),
    # instead of relying on the slew inequality to discover the ramp (which it under-enforces
    # for multi-axis, leaving edge-step violations).
    n_ramp = max(1, int(np.ceil(G_max / (slew_rate_max * dt))))
    er = enc.copy()
    for j in range(1, n_ramp + 1):
        er &= np.roll(enc, j) & np.roll(enc, -j)               # keep only interior samples
    enc = er
    s = jnp.asarray(np.where(np.arange(n_t) < echo, 1.0, -1.0)[:, None])   # (n_t,1)
    t = jnp.asarray(np.arange(n_t) * dt)[:, None]
    free = np.where(enc)[0]
    nf = len(free)
    full_idx = jnp.asarray(free)
    f_target = float(spectral_freq) if spectral_freq else 0.0
    bscale = (GAMMA * G_max) ** 2 * TE ** 3 / 50.0
    iu = np.triu_indices(na)                                   # Maxwell upper-triangle (static)

    # slew_param=True: LINEAR per-axis slew -- the VARIABLE IS the slew, box-bounded [-Smax,Smax]
    # per axis (the physical per-amplifier limit; box bounds are enforced exactly by SLSQP),
    # g=cumsum(slew)*dt linear, amplitude as a per-axis linear inequality.
    # EMPIRICAL (the multi-axis story, established across 6 formulations -- vector slew, per-axis
    # slew, trust-constr, edge-erosion, objective-penalty, box-slew):
    #   * single-axis (LTE/OGSE): every form works.
    #   * multi-axis (PTE/STE): scipy SLSQP leaves the INEQUALITY constraints loose, whichever
    #     they are -- box-slew fixes slew (=Smax exactly) but then AMPLITUDE blows up (g 0.3-1.2
    #     T/m) and refoc fails; the tanh-structural form instead fails the squashed equalities.
    # ROOT CAUSE = the SOLVER, not the formulation: scipy SLSQP can't enforce the multi-axis
    # inequalities.  NOW does multi-axis via MATLAB fmincon (mature interior-point); our AL
    # designer via structural tanh + augmented-Lagrangian penalty.  A NOW-faithful multi-axis
    # SQP needs a fmincon-class interior-point (scipy trust-constr w/ cached/sparse Jacobians).
    # So: AL (waveform_designer) = the single unified all-shape solver; this SQP (slew_param=
    # False default, direct-g) = single-axis specialist (LTE/OGSE -- cleaner, beats AL, MC-valid).
    def gfull(x):                                              # x (nf*na,) -> g (n_t,na)
        raw = x.reshape(nf, na)
        if not slew_param:
            return jnp.zeros((n_t, na)).at[full_idx].set(raw)  # legacy direct-g (off=0)
        # LINEAR slew parameterization: the VARIABLE IS the slew (T/m/s), bounded per-axis by a
        # simple box [-Smax, Smax] (the physical per-amplifier limit), so slew feasibility is
        # exact & free (box bounds are enforced directly by SLSQP).  g = cumsum(slew)*dt is
        # LINEAR in the variable -> the equalities stay as tractable as direct-g (no tanh
        # squash).  Amplitude |g_k| <= Gmax is a separate per-axis LINEAR inequality.
        slew_full = jnp.zeros((n_t, na)).at[full_idx].set(raw)   # raw = slew, slew=0 off-regions
        return dt * jnp.cumsum(slew_full, axis=0)              # g(0)=0, linear in slew
    def qof(g):
        return GAMMA * jnp.cumsum(s * g, axis=0) * dt          # (n_t,na)
    def Bof(q):
        return jnp.sum(q[:, :, None] * q[:, None, :], axis=0) * dt   # (na,na) broadcast-sum

    def negb(x):
        return -jnp.trace(Bof(qof(gfull(x)))) / bscale
    qscale = GAMMA * G_max * TE                                # ~ max |q|; keep refoc O(1)
    def c_refoc(x):
        return qof(gfull(x))[-1] / qscale                      # (na,)  q(TE)=0, normalized
    def c_shape(x):
        B = Bof(qof(gfull(x)))
        w = jnp.linalg.eigvalsh(B + 1e-9 * jnp.trace(B) * jnp.eye(na))   # jitter: avoid eig non-conv
        return jnp.array([jnp.sum((w - jnp.mean(w)) ** 2) / (jnp.trace(B) ** 2 + 1e-30)])
    def c_m1(x):
        g = gfull(x); return jnp.sum(t * s * g, 0) * dt / (G_max * TE ** 2)        # (na,)
    def c_m2(x):
        g = gfull(x); return jnp.sum(t ** 2 * s * g, 0) * dt / (G_max * TE ** 3)
    def c_mxwl(x):
        g = gfull(x)
        M = jnp.sum(s[:, :, None] * g[:, :, None] * g[:, None, :], 0) * dt / (G_max ** 2 * TE)
        return M[iu]                                           # upper-triangle = 0
    def c_spec(x):
        g = gfull(x); q = qof(g)
        frms = jnp.sqrt(GAMMA ** 2 * jnp.sum(g ** 2) / (jnp.sum(q ** 2) + 1e-30)) / (2 * np.pi)
        return jnp.array([(frms - f_target) / (f_target + 1e-9)])
    # slew_param holds g constant through the off-regions (slew=0 there), so require g=0 at
    # the 180 (clean sign flip; g held there = g[echo]) and at TE -> each lobe ramps 0->0
    # within its window (the standard spin-echo structure).
    def c_g180(x):  return gfull(x)[int(echo)] / G_max          # (na,)  g=0 during the 180
    def c_gTE(x):   return gfull(x)[-1] / G_max                 # (na,)  g(TE)=0
    def slew_margin(x):                                       # vector-norm (nonlinear)
        d = jnp.diff(gfull(x), axis=0) / dt
        return (slew_rate_max - jnp.sqrt(jnp.sum(d ** 2, 1) + 1e-30)) / slew_rate_max   # ≥0
    # per-axis LINEAR slew: |dg_k/dt| <= S_max/sqrt(na) per axis  => vector |dg/dt| <= S_max.
    # Linear in x, so SLSQP's QP enforces it EXACTLY (the nonlinear vector-norm above gets
    # linearized and left loose for multi-axis); conservative by ~1/sqrt(na) but feasibility-
    # guaranteed.  Returns [c - dg_k, c + dg_k] over axes (both >=0 <=> |dg_k| <= c).
    cax = slew_rate_max / np.sqrt(na)
    def slew_margin_axis(x):
        d = jnp.diff(gfull(x), axis=0) / dt                   # (n_t-1, na)
        return jnp.concatenate([(cax - d).reshape(-1), (cax + d).reshape(-1)]) / slew_rate_max
    def amp_margin(x):
        g = gfull(x); return (G_max ** 2 - jnp.sum(g ** 2, 1)) / G_max ** 2             # ≥0
    def amp_axis(x):                                          # per-axis |g_k| <= Gmax (LINEAR)
        g = gfull(x)
        return jnp.concatenate([(G_max - g).reshape(-1), (G_max + g).reshape(-1)]) / G_max

    # slew_mode='penalty': fold the slew overshoot into the OBJECTIVE (SLSQP minimizes it as
    # part of the cost, which it does reliably) instead of a hard inequality (which SLSQP
    # leaves loose for the equality-heavy multi-axis problem, vector OR per-axis alike).
    def negb_pen(x):
        d = jnp.diff(gfull(x), axis=0) / dt
        sl = jnp.sqrt(jnp.sum(d ** 2, 1) + 1e-30)
        return negb(x) + 50.0 * jnp.sum(jnp.maximum(sl / slew_rate_max - 1.0, 0.0) ** 2)

    jit = jax.jit
    obj = negb_pen if (slew_mode == "penalty" and not slew_param) else negb
    f_val, f_jac = jit(obj), jit(jax.grad(obj))
    eqs = [c_refoc]
    if na >= 2:        eqs.append(c_shape)
    if null_M1:        eqs.append(c_m1)
    if null_M2:        eqs.append(c_m2)
    if maxwell:        eqs.append(c_mxwl)
    if f_target > 0.0: eqs.append(c_spec)
    if slew_param:
        # slew = box-bounded variable (exact, free); amplitude = per-axis LINEAR ineq; pin g=0
        # at 180 & TE.  All constraints linear or as-tractable-as-direct-g -> SLSQP-friendly.
        eqs += [c_g180, c_gTE]
        ineqs = [amp_axis]
    elif slew_mode == "penalty":
        ineqs = [amp_margin]
    else:
        ineqs = [slew_margin_axis if slew_mode == "per_axis" else slew_margin, amp_margin]
    if method == "trust-constr":
        cons = ([NonlinearConstraint(jit(fn), 0.0, 0.0, jac=jit(jax.jacfwd(fn))) for fn in eqs]
                + [NonlinearConstraint(jit(fn), 0.0, np.inf, jac=jit(jax.jacfwd(fn))) for fn in ineqs])
        bounds = None
    else:
        cons = [{"type": "eq", "fun": jit(fn), "jac": jit(jax.jacfwd(fn))} for fn in eqs]
        cons += [{"type": "ineq", "fun": jit(fn), "jac": jit(jax.jacfwd(fn))} for fn in ineqs]
        # slew_param: BOX-BOUND the slew variable to [-Smax,Smax] (per-axis slew, exact & free);
        # legacy direct-g: box on g samples to [-Gmax,Gmax].
        bnd = slew_rate_max if slew_param else G_max
        bounds = [(-bnd, bnd)] * (nf * na)

    rng = np.random.default_rng(seed)
    edge = np.sin(np.linspace(0.0, np.pi, nf))                 # 0 at window edges
    def bipolar(fr):
        h = max(1, int(nf * fr)); return edge * np.concatenate([-np.ones(h), np.ones(nf - h)])
    # init scale: slew_param raw is the slew (~S_max); legacy g-domain is ~G_max.
    sc = (slew_rate_max * 0.3) if slew_param else (G_max * 0.85 / np.sqrt(na))
    inits = []
    # structured: each axis a bipolar at a distinct split (distinct axes -> full-rank B for STE)
    for base_fr in (0.45, 0.5, 0.55, 0.6):
        x = np.zeros((nf, na))
        for k in range(na):
            x[:, k] = bipolar(min(0.75, base_fr + 0.07 * k)) * sc
        inits.append(x.reshape(-1))
    for _ in range(max(0, n_restarts - len(inits))):           # low-frequency random per axis
        x = np.zeros((nf, na))
        for k in range(na):
            kk = int(rng.integers(1, 4)); ph = rng.uniform(0, np.pi, kk); am = rng.uniform(0.4, 1.0, kk)
            v = sum(a * np.sin((j + 1) * np.linspace(0, np.pi, nf) + p) for j, (a, p) in enumerate(zip(am, ph)))
            x[:, k] = edge * v / (np.max(np.abs(v)) + 1e-9) * sc
        inits.append(x.reshape(-1))
    if init_g is not None:
        ig = np.asarray(init_g, float)
        ig = ig[:, :na] if ig.ndim == 2 else ig[:, None]
        inits.insert(0, ig[free].reshape(-1))

    def solve(x0, iters):                                     # negb is already O(1)-scaled
        if method == "trust-constr":
            return minimize(lambda x: float(f_val(x)), x0, jac=lambda x: np.asarray(f_jac(x)),
                            method="trust-constr", bounds=bounds, constraints=cons,
                            options={"maxiter": iters, "gtol": 1e-8, "xtol": 1e-10}).x
        return minimize(lambda x: float(f_val(x)), x0, jac=lambda x: np.asarray(f_jac(x)),
                        method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": iters, "ftol": 1e-9}).x

    def evaluate(x):
        g = np.asarray(gfull(jnp.asarray(x)))
        q = np.asarray(qof(jnp.asarray(g)))
        B = np.asarray(Bof(jnp.asarray(q)))
        b = float(np.trace(B))
        w = np.sort(np.linalg.eigvalsh(B))[::-1]
        shape = float(np.sum((w - w.mean()) ** 2) / (np.trace(B) ** 2 + 1e-30)) if na >= 2 else 0.0
        refoc = float(np.linalg.norm(q[-1]) / (np.sqrt(np.max((q ** 2).sum(1))) + 1e-30))
        amp = float(np.max(np.sqrt((g ** 2).sum(1))))
        sl = float(np.max(np.sqrt((np.diff(g, axis=0) / dt) ** 2).sum(1)) if g.shape[0] > 1 else 0.0)
        M = (np.asarray(s)[:, 0][:, None, None] * g[:, :, None] * g[:, None, :]).sum(0) * dt
        mx = float(np.sqrt(np.sum(M ** 2)) / (G_max ** 2 * TE))
        m1 = float(np.linalg.norm((np.asarray(t) * np.asarray(s) * g).sum(0) * dt) / (G_max * TE ** 2))
        m2 = float(np.linalg.norm((np.asarray(t) ** 2 * np.asarray(s) * g).sum(0) * dt) / (G_max * TE ** 3))
        frms = float(np.sqrt(GAMMA ** 2 * np.sum(g ** 2) / (np.sum(q ** 2) + 1e-30)) / (2 * np.pi))
        feas = (refoc < 1e-2 and amp <= G_max * 1.02 and sl <= slew_rate_max * 1.03
                and (na < 2 or shape < 5e-2) and (not null_M1 or m1 < 5e-2)
                and (not null_M2 or m2 < 5e-2) and (not maxwell or mx < 2e-2)
                and (f_target <= 0 or abs(frms - f_target) / f_target < 0.15))
        G3 = np.zeros((n_t, 3)); G3[:, :na] = g
        return dict(G=G3, x=x, b_value=b, b_delta=float(b_delta), shape=shape, refoc=refoc,
                    max_amp=amp, max_slew=sl, maxwell_index=mx, m1_index=m1, m2_index=m2,
                    f_rms=frms, feasible=feas, na=na, echo_idx=int(echo), dt=dt)

    best = last = None
    for x0 in inits:
        last = evaluate(solve(x0, maxiter))
        if last["feasible"] and (best is None or last["b_value"] > best["b_value"]):
            best = last
    if best is not None:                                       # polish the winner
        pol = evaluate(solve(best["x"], maxiter * 2))
        if pol["feasible"] and pol["b_value"] >= best["b_value"] - 1.0:
            best = pol
    return best if best is not None else last
