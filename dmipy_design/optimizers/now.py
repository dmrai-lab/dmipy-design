"""NOW — Numerical Optimization of gradient Waveforms, native in dmipy (NumPy + SciPy only).

A faithful port of NOW's solver recipe (Sjölund et al., J. Magn. Reson. 261 (2015);
github.com/jsjol/NOW): maximize the b-value of a direct gradient waveform under hardware and
physics constraints with sequential quadratic programming (SciPy SLSQP), the constraints
expressed the way NOW expresses them —

  * LINEAR constraints as constant matrices: per-axis slew-rate (|dG_k/dt| <= S_max), refocus
    (q(TE)=0), and motion moments (M1, M2 = 0).  Amplitude (|G_k| <= G_max) is a box bound on
    the free samples.  (g=0 in the off-regions / at the 180 / at TE holds by construction.)
  * NONLINEAR constraints for the b-tensor SHAPE (b_delta) and optional Maxwell (concomitant)
    compensation, as b-tensor component equalities.
  * ANALYTIC objective gradient (db/dg is a reverse-cumsum of q) -- NO autodiff.

Why NumPy + SciPy and not JAX here: SQP runs its loop in C and calls the objective/constraints
each iteration.  With JAX those calls cross a per-iteration numpy<->JAX bridge that dominates
the wall clock (minutes); with NumPy-analytic derivatives SciPy calls NumPy directly (LTE in
~1 s).  And because SQP handles the constraints EXACTLY (not via a penalty), the objective is
never dwarfed, so the b-value reaches the true optimum (rides slew=S_max and |G|=G_max).

Covers LTE / PTE / STE (b_delta = 1 / -0.5 / 0) and OGSE (pass ``spectral_freq``: a rank-1
shape plus one extra equality pinning the encoding's RMS frequency, f_rms = target).  Single
solver, all shapes -- the LTE problem is just the rank-1 case, OGSE the rank-1 + spectral case.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from dataclasses import dataclass

GAMMA = 267.513e6  # rad/s/T

# SAFE PNS coefficients (Hebrank/Szczepankiewicz model) -- pypulseq's representative
# Siemens-class example.  Per axis: tau1/2/3 (ms), a1/2/3, stim_limit, g_scale.  Canonical
# NumPy copy for the NOW solver (the AL designer carries a JAX twin); the in-design constraint
# shapes the waveform, pulseq_pns_report verifies the exact .asc on export.
_SAFE_HW = (
    dict(tau1=0.2, tau2=0.03, tau3=3.0, a1=0.4, a2=0.1, a3=0.5, stim_limit=30.0, g_scale=0.35),
    dict(tau1=1.5, tau2=2.5, tau3=0.15, a1=0.55, a2=0.15, a3=0.3, stim_limit=15.0, g_scale=0.31),
    dict(tau1=2.0, tau2=0.12, tau3=1.0, a1=0.42, a2=0.4, a3=0.18, stim_limit=25.0, g_scale=0.25),
)


def _safe_kernels(dt_ms, m):
    """(9, m) causal RC-lowpass kernels (3 axes x 3 taus): alpha*(1-alpha)^k, alpha=dt/(tau+dt)."""
    kers = []
    for hw in _SAFE_HW:
        for tau in (hw['tau1'], hw['tau2'], hw['tau3']):
            alpha = dt_ms / (tau + dt_ms)
            kers.append(alpha * (1.0 - alpha) ** np.arange(m))
    return np.asarray(kers)


def _pns_pct(G3, dt, kernels):
    """Per-timepoint SAFE PNS (% of stimulation limit) of physical gradient G3 (n_t,3 T/m):
    3 RC-lowpass terms per axis, normalized, L2-combined across axes -- same model as
    pypulseq's calculate_pns.  Returns the (m,) time series (caller takes max / constrains all)."""
    dgdt = np.diff(G3, axis=0) / dt                            # (m,3) T/m/s
    m = dgdt.shape[0]; pns_sq = np.zeros(m)
    for ax in range(3):
        hw = _SAFE_HW[ax]; d = dgdt[:, ax]; k = 3 * ax
        lp1 = np.convolve(d, kernels[k + 0])[:m]
        lp2 = np.convolve(np.abs(d), kernels[k + 1])[:m]
        lp3 = np.convolve(d, kernels[k + 2])[:m]
        stim = hw['a1'] * np.abs(lp1) + hw['a2'] * lp2 + hw['a3'] * np.abs(lp3)
        pns_sq += (stim / hw['stim_limit'] * hw['g_scale'] * 100.0) ** 2
    return np.sqrt(pns_sq)


def _rank_of(b_delta):
    if abs(b_delta - 1.0) < 1e-6:
        return 1
    if abs(b_delta + 0.5) < 1e-6:
        return 2
    return 3


@dataclass
class NowDesign:
    G: np.ndarray            # (n_t, 3) physical gradient, T/m
    dt: float
    echo_idx: int
    b_value: float           # s/m²
    b_delta: float
    n_axes: int
    max_slew: float
    max_amplitude: float
    refocus_residual: float
    shape_residual: float
    m1_index: float
    m2_index: float
    maxwell_index: float
    feasible: bool
    spectral_rms: float = 0.0   # Hz, RMS encoding frequency (OGSE); 0 for non-oscillating
    pns_pct: float = 0.0        # %, peak SAFE PNS (% of stimulation limit); 0 if not constrained
    heat_frac: float = 0.0      # mean gradient energy ⟨g²⟩ as a fraction of G_max²


def design_waveform_now(b_delta=1.0, *, G_max=0.08, slew_rate_max=200.0, TE=0.060, n_t=140,
                        timing=None, null_M1=True, null_M2=True, maxwell=False,
                        spectral_freq=None, pns=False, pns_target=80.0, heat_eta=None,
                        n_axes=None, n_restarts=8, maxiter=300, seed=0):
    """Design a max-b spin-echo gradient waveform via NOW's SQP recipe (LTE/PTE/STE).

    ``timing`` is a ``SequenceTiming``; if None a default Prisma budget is used.  Returns a
    ``NowDesign`` with the physical gradient and the (machine-precision) constraint residuals.
    """
    from dmipy_design.optimizers import SequenceTiming
    if timing is None:
        timing = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)
    na = _rank_of(b_delta) if n_axes is None else int(n_axes)
    slew_off, echo = timing.masks(TE, n_t)
    enc = np.asarray(slew_off)[:, 0] > 0.5
    dt = TE / (n_t - 1); echo = int(echo)
    # erode encoding windows by the ramp length so g ramps from 0 at the window edges
    nr = max(1, int(np.ceil(G_max / (slew_rate_max * dt))))
    er = enc.copy()
    for j in range(1, nr + 1):
        er &= np.roll(enc, j) & np.roll(enc, -j)
    free = np.where(er)[0]; nf = len(free); nvar = na * nf
    s = np.where(np.arange(n_t) < echo, 1.0, -1.0)[:, None]
    tt = (np.arange(n_t) * dt)[:, None]
    bscale = (GAMMA * G_max) ** 2 * TE ** 3 / 50.0

    def gof(x):
        g = np.zeros((n_t, na)); g[free, :] = x.reshape(na, nf).T; return g
    def qof(g):
        return GAMMA * np.cumsum(s * g, 0) * dt
    def Bof(q):
        return dt * (q.T @ q)

    def fun(x):                                               # -b/scale + analytic gradient
        g = gof(x); q = qof(g); b = np.trace(Bof(q))
        Qrev = np.flip(np.cumsum(np.flip(q, 0), 0), 0)        # Σ_{t>=a} q[t,:]
        gx = (2 * GAMMA * dt ** 2 * s * Qrev)[free, :].T.reshape(-1)
        return -b / bscale, -gx / bscale

    # ---- constant LINEAR-constraint matrices (x axis-major: x[k*nf + i]) ----
    Sel = sp.csr_matrix((np.ones(nf), (free, np.arange(nf))), shape=(n_t, nf))
    D = sp.diags([-np.ones(n_t), np.ones(n_t - 1)], [0, 1], shape=(n_t - 1, n_t))
    A_slew = sp.block_diag([(D @ Sel) / dt] * na).toarray()
    cons = [{"type": "ineq", "fun": lambda x: slew_rate_max - A_slew @ x, "jac": lambda x: -A_slew},
            {"type": "ineq", "fun": lambda x: slew_rate_max + A_slew @ x, "jac": lambda x: A_slew},
            {"type": "eq", "fun": (lambda A: (lambda x: A @ x))(sp.block_diag([(GAMMA * dt * s[free, 0]).reshape(1, -1)] * na).toarray()),
             "jac": (lambda A: (lambda x: A))(sp.block_diag([(GAMMA * dt * s[free, 0]).reshape(1, -1)] * na).toarray())}]
    if null_M1:
        A = sp.block_diag([((tt * s)[free, 0] * dt).reshape(1, -1)] * na).toarray()
        cons.append({"type": "eq", "fun": (lambda A: lambda x: A @ x)(A), "jac": (lambda A: lambda x: A)(A)})
    if null_M2:
        A = sp.block_diag([((tt ** 2 * s)[free, 0] * dt).reshape(1, -1)] * na).toarray()
        cons.append({"type": "eq", "fun": (lambda A: lambda x: A @ x)(A), "jac": (lambda A: lambda x: A)(A)})

    # ---- nonlinear: b-tensor SHAPE + optional Maxwell, with ANALYTIC Jacobians ----
    # Constant normalization (not /trace) so the constraint is a pure quadratic form in g and
    # its Jacobian is a linear combination of dB_jk/dg = γ dt² s[a]·(δ_jm Qrev[a,k]+δ_km Qrev[a,j]),
    # Qrev[a,k]=Σ_{t>=a} q[t,k] (the SAME reverse-cumsum as the objective gradient).
    bcoef = GAMMA * dt ** 2 * s[free, 0]                       # (nf,)  for dB/dg at free samples
    SHP = ([((0, 0), (1, 1)), ((0, 1), None)] if na == 2 else
           [((0, 0), (1, 1)), ((1, 1), (2, 2)), ((0, 1), None), ((0, 2), None), ((1, 2), None)])
    def _dBdx(Qrev, j, k):
        v = np.zeros(nvar)
        if j == k: v[j * nf:(j + 1) * nf] = 2 * bcoef * Qrev[free, j]
        else:
            v[j * nf:(j + 1) * nf] += bcoef * Qrev[free, k]; v[k * nf:(k + 1) * nf] += bcoef * Qrev[free, j]
        return v
    def c_shape(x):
        B = Bof(qof(gof(x)))
        return np.array([(B[p] - B[q]) if q else B[p] for p, q in SHP]) / bscale
    def j_shape(x):
        q = qof(gof(x)); Qrev = np.flip(np.cumsum(np.flip(q, 0), 0), 0)
        return np.array([(_dBdx(Qrev, *p) - _dBdx(Qrev, *q)) if q else _dBdx(Qrev, *p) for p, q in SHP]) / bscale
    if na >= 2:
        cons.append({"type": "eq", "fun": c_shape, "jac": j_shape})
    if maxwell:
        iu = list(zip(*np.triu_indices(na))); mcoef = dt / (G_max ** 2 * TE) * s[free, 0]
        def c_mx(x):
            g = gof(x); M = (s[:, :, None] * g[:, :, None] * g[:, None, :]).sum(0) * dt / (G_max ** 2 * TE)
            return np.array([M[j, k] for j, k in iu])
        def j_mx(x):                                          # dM_jk/dg[a,m]=mcoef[a](δ_jm g[a,k]+δ_km g[a,j])
            g = gof(x); rows = []
            for j, k in iu:
                v = np.zeros(nvar)
                if j == k: v[j * nf:(j + 1) * nf] = 2 * mcoef * g[free, j]
                else:
                    v[j * nf:(j + 1) * nf] += mcoef * g[free, k]; v[k * nf:(k + 1) * nf] += mcoef * g[free, j]
                rows.append(v)
            return np.array(rows)
        cons.append({"type": "eq", "fun": c_mx, "jac": j_mx})
    if spectral_freq is not None:                             # OGSE: pin RMS encoding frequency
        # ω_rms² = γ²·Σg²/Σq²  (Parseval: ∫(dq/dt)²dt = γ²∫g²dt, ∫q²dt the denominator).
        # Equality on ω_rms²/ω_t² so f_rms = spectral_freq; analytic Jacobian (quotient rule).
        wt2 = (2 * np.pi * spectral_freq) ** 2
        def c_spec(x):
            g = gof(x); q = qof(g)
            return np.array([(GAMMA ** 2 * np.sum(g ** 2) / (np.sum(q ** 2) + 1e-30) - wt2) / wt2])
        def j_spec(x):
            g = gof(x); q = qof(g); Qrev = np.flip(np.cumsum(np.flip(q, 0), 0), 0)
            N = np.sum(g ** 2); Dq = np.sum(q ** 2) + 1e-30
            dN = np.zeros(nvar); dD = np.zeros(nvar)
            for k in range(na):
                dN[k * nf:(k + 1) * nf] = 2 * g[free, k]
                dD[k * nf:(k + 1) * nf] = 2 * GAMMA * dt * s[free, 0] * Qrev[free, k]
            return (GAMMA ** 2 * (dN * Dq - N * dD) / Dq ** 2 / wt2)[None, :]
        cons.append({"type": "eq", "fun": c_spec, "jac": j_spec})
    if pns:                                                   # SAFE PNS <= pns_target % at every t
        pkern = _safe_kernels(dt * 1e3, n_t - 1)              # physical gradient = g (not s·g)
        def c_pns(x):                                         # one row per timepoint, all >= 0
            G3 = np.zeros((n_t, 3)); G3[:, :na] = gof(x)
            return pns_target - _pns_pct(G3, dt, pkern)       # FD Jacobian (nvar evals, cheap conv)
        cons.append({"type": "ineq", "fun": c_pns})
    if heat_eta is not None:                                  # coil heating: ⟨g²⟩ <= heat_eta·G_max²
        hscale = n_t * heat_eta * G_max ** 2
        def c_heat(x):
            g = gof(x); return np.array([1.0 - np.sum(g ** 2) / hscale])
        def j_heat(x):
            g = gof(x); v = np.zeros(nvar)
            for k in range(na): v[k * nf:(k + 1) * nf] = -2 * g[free, k] / hscale
            return v[None, :]
        cons.append({"type": "ineq", "fun": c_heat, "jac": j_heat})
    bounds = [(-G_max, G_max)] * nvar                         # amplitude box

    rng = np.random.default_rng(seed); edge = np.sin(np.linspace(0, np.pi, nf))
    # OGSE init oscillates near the target; bracket the lobe count symmetrically about the rough
    # estimate (≈ 2·f·T_enc + 1 half-sines) so low-frequency targets get low-frequency starts too.
    center = max(2, 2 * int(round(spectral_freq * nf * dt)) + 1) if spectral_freq is not None else 0
    best = None
    for r in range(n_restarts):
        x0 = np.zeros((na, nf))
        for k in range(na):
            f_lobes = max(1, center - n_restarts // 2 + r) if spectral_freq is not None else (k + 1 + r)
            x0[k] = edge * np.sin(f_lobes * np.linspace(0, np.pi, nf) + rng.uniform(0, 1)) * G_max * 0.5 / np.sqrt(na)
        res = minimize(fun, x0.reshape(-1), jac=True, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": maxiter, "ftol": 1e-9})
        g = gof(res.x); q = qof(g); B = Bof(q); b = float(np.trace(B))
        refoc = float(np.linalg.norm(q[-1]) / (np.sqrt(np.max((q ** 2).sum(1))) + 1e-30))
        sl = float(np.max(np.abs(np.diff(g, axis=0) / dt))); amp = float(np.max(np.abs(g)))
        w = np.sort(np.linalg.eigvalsh(B))[::-1]
        shape = float(np.sum((w - w.mean()) ** 2) / (b ** 2 + 1e-30)) if na >= 2 else 0.0
        m1 = float(np.linalg.norm((tt * s * g).sum(0) * dt) / (G_max * TE ** 2))
        m2 = float(np.linalg.norm((tt ** 2 * s * g).sum(0) * dt) / (G_max * TE ** 3))
        mx = float(np.sqrt(np.sum(((s[:, :, None] * g[:, :, None] * g[:, None, :]).sum(0) * dt) ** 2)) / (G_max ** 2 * TE))
        frms = float(GAMMA * np.sqrt(np.sum(g ** 2) / (np.sum(q ** 2) + 1e-30)) / (2 * np.pi))
        spec_ok = spectral_freq is None or abs(frms - spectral_freq) / spectral_freq < 5e-2
        G3 = np.zeros((n_t, 3)); G3[:, :na] = g
        pns_pk = float(np.max(_pns_pct(G3, dt, _safe_kernels(dt * 1e3, n_t - 1)))) if pns else 0.0
        heatf = float(np.sum(g ** 2) / (n_t * G_max ** 2))
        feas = (refoc < 1e-2 and sl <= slew_rate_max * 1.02 and amp <= G_max * 1.02
                and (na < 2 or shape < 5e-2) and (not null_M1 or m1 < 5e-2)
                and (not null_M2 or m2 < 5e-2) and (not maxwell or mx < 2e-2) and spec_ok
                and (not pns or pns_pk <= pns_target * 1.02)
                and (heat_eta is None or heatf <= heat_eta * 1.02))
        cand = NowDesign(G=G3, dt=dt, echo_idx=echo, b_value=b, b_delta=float(b_delta), n_axes=na,
                         max_slew=sl, max_amplitude=amp, refocus_residual=refoc, shape_residual=shape,
                         m1_index=m1, m2_index=m2, maxwell_index=mx, feasible=feas, spectral_rms=frms,
                         pns_pct=pns_pk, heat_frac=heatf)
        if feas and (best is None or b > best.b_value):
            best = cand
    return best if best is not None else cand
