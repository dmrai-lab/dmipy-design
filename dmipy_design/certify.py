"""Eq. (4): certify the K of a replay pack against the deliverable waveform set.

A replay pack stores the lowest ``K`` temporal bands of each walker's Brownian bridge, so K is
not a storage knob but a **claim**: that no waveform the consumer can physically produce reads
the truncated walk differently from the full one by more than ``eps``,

    K_min = min K  s.t.  sup_{G in G_scanner} |S_full(G) - S_K(G)| <= eps                (4)

The claim has to be settled here, at build time, because the failure is silent. A waveform
outside the certified band replays to a smooth, plausible, wrong number -- no NaN, no warning,
no shape anomaly -- so a consumer cannot tell from their own output that they have left the
band. Either the pack carries the answer or nobody has it.

This lives in dmipy-design, not dmipy-sim, because ``G_scanner`` does: the module that decides
which acquisitions are deliverable is the one that can say what replaying its own designs
costs. It shares :func:`dmipy_design.constraints.waveform_problem` with the NOW designer, and
that sharing is load-bearing rather than tidy -- a certificate taken over a SMALLER set than
the designer can reach is unsound, because a user could then design a waveform the pack was
never certified for.

Two things make the sup tractable.

*The constraints are handled exactly.* SQP holds the slew band, amplitude box, refocusing and
moment nulls as constraints rather than penalties, so the adversary rides ``|G| = G_max`` and
``|dG/dt| = slew_max`` instead of stopping short of them.

*Only the bands above K create the error, but every band steers it.* With
``d_phi = gamma dt <G, e_n>`` and ``e_n = r_n - r_n^K``,

    |S_full - S_K| = | < exp(i phi_K) (exp(i d_phi) - 1) > |

``e_n`` has support only above K, so only those bands drive ``d_phi``; but ``phi_K`` depends on
every band and decides whether the per-walker errors add or cancel. The adversary therefore
uses the low bands to align the ensemble and the high bands to break it, which is why a
one-mode-at-a-time scan understates the sup by one to two orders of magnitude -- worst exactly
where the per-mode amplitude ceiling is flat (preclinical inserts), where the tail behaves as
``sum 1/m`` rather than ``sum 1/m^2``.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
from dmipy_sim.constants import GAMMA
from .constraints import waveform_problem

__all__ = ["ReplayEnvelope", "replay_envelope", "SupResult", "Certificate",
           "PoolCertificate", "sup_replay_error", "k_min", "certify",
           "certify_pools", "k_for_walk"]


@dataclass(frozen=True)
class ReplayEnvelope:
    """A named scanner class, as the deliverable-waveform set Eq. (4) takes its sup over."""

    name: str
    G_max: float
    slew_rate_max: float
    raster: float = 10e-6
    null_M1: bool = False
    null_M2: bool = False
    n_axes: int = 3
    provenance: dict = field(default_factory=dict)

    def problem(self, n_t, dt, *, echo=None, encoding_mask=None):
        """The shared :class:`~dmipy_design.constraints.WaveformProblem` for this class.

        ``echo=None`` means the trajectory is read by an ALREADY-EFFECTIVE waveform (the 180 is
        folded in, as ``dmipy_sim.replay.compression.acquisition_battery`` builds them), so the sign is
        +1 throughout and the refocusing constraint reduces to nulling M0. Pass an echo index
        for a literal spin-echo layout.
        """
        return waveform_problem(n_t=n_t, n_axes=self.n_axes, dt=dt,
                                echo=(n_t if echo is None else echo),
                                encoding_mask=encoding_mask, G_max=self.G_max,
                                slew_rate_max=self.slew_rate_max,
                                null_M1=self.null_M1, null_M2=self.null_M2)

    def describe(self):
        return dict(name=self.name, G_max=self.G_max, slew_rate_max=self.slew_rate_max,
                    raster=self.raster, null_M1=self.null_M1, null_M2=self.null_M2,
                    n_axes=self.n_axes, **self.provenance)


def replay_envelope(model=None, *, G_max=None, slew_rate_max=None, raster=10e-6,
                    null_M1=False, null_M2=False, n_axes=3, regime="diffusion", name=None):
    """Build a :class:`ReplayEnvelope` from the cited scanner catalogue, or from explicit limits.

    ``regime='diffusion'`` takes the PNS-derated diffusion slew where a model publishes one, and
    that is the limit that actually binds a diffusion sequence rather than a detail: Connectom
    is 200 T/m/s hardware but 62.5 during diffusion encoding, which puts its 300 mT/m out of
    reach at every mode that matters and makes it the EASIEST class to certify, not the hardest.

    Preclinical inserts have no cited catalogue entry, so they are given explicitly, e.g.
    ``replay_envelope(G_max=1.0, slew_rate_max=1e4, name='insert-1T/m')``.
    """
    if model is not None:
        from dmipy_sim.acquisition.scanners import ScannerLimits
        lim = ScannerLimits.of(model, regime=regime)
        return ReplayEnvelope(name or model, lim.G_max, lim.slew_max,
                              raster if lim.grad_raster is None else lim.grad_raster, null_M1, null_M2, n_axes,
                              dict(source="dmipy_sim.acquisition.scanners", model=lim.name, slew_regime=regime))
    if G_max is None or slew_rate_max is None:
        raise ValueError("give a catalogue `model`, or both `G_max` and `slew_rate_max`")
    return ReplayEnvelope(name or f"custom-{G_max*1e3:.0f}mT/m", float(G_max),
                          float(slew_rate_max), float(raster), null_M1, null_M2, n_axes,
                          dict(source="explicit"))


@dataclass
class SupResult:
    """The sup of Eq. (4) -- and the two numbers, which answer different questions.

    ``sup`` is the CERTIFICATE. The ascent runs on the pack's own walkers and is scored on them,
    which is exactly what a consumer experiences, because the pack *is* the ensemble: they
    replay G against these walkers, and the truth is S_full on these walkers. No waveform does
    worse on this pack.

    ``held_out`` is the typical case: fitted on half the walkers, scored on the other half, so a
    waveform tuned to this particular realisation gets no credit for it. It runs 4-20x below the
    certificate and the gap narrows with N -- the adversary partly memorises the ensemble, and
    more walkers leave it less room. Certify on ``sup``; quote ``held_out`` so the gap is
    visible rather than implied.
    """

    sup: float
    held_out: float | None = None
    waveform: np.ndarray | None = None
    n_walkers: int = 0
    b_value: float | None = None

    def __float__(self):
        return float(self.sup)

    def __le__(self, other):
        return self.sup <= float(other)

    def __gt__(self, other):
        return self.sup > float(other)


def _truncate(X, K):
    """The bridge-truncated walk: exact endpoints plus the lowest K DST-I bands.

    Returned as float32. The objective streams the whole trajectory array on every evaluation
    and is memory-bandwidth bound, not compute bound, so halving the traffic nearly halves the
    wall clock. The cost is a ~1e-5 relative phase error, three orders below the eps this
    certifies against.
    """
    from dmipy_sim.replay.compression import encode_bridge_dst, decode_bridge_dst
    arrays, meta, _ = encode_bridge_dst(np.asarray(X, np.float64), K)
    return np.asarray(decode_bridge_dst(arrays, meta), np.float32)


def _objective(prob, Xf, Xk, dt):
    """``-|S_full - S_K|`` and its analytic gradient in the flat variable layout."""
    n = Xf.shape[0]
    sgn = prob.sign

    def fun(x):
        Geff = prob.effective(x)                          # (n_t, n_axes), 180 folded in
        pf = (GAMMA * dt) * np.einsum("td,ntd->n", Geff, Xf[:, :, :prob.n_axes])
        pk = (GAMMA * dt) * np.einsum("td,ntd->n", Geff, Xk[:, :, :prob.n_axes])
        ef, ek = np.exp(1j * pf), np.exp(1j * pk)
        D = ef.mean() - ek.mean()
        a = abs(D)
        if a < 1e-300:
            return 0.0, np.zeros(prob.n_var)
        c = np.conj(D) / a
        # d(mean exp(i phi))/dg[t,d] = i gamma dt sign[t] <exp(i phi) r[:,t,d]> / N
        gf = (1j * GAMMA * dt) * np.einsum("n,ntd->td", ef, Xf[:, :, :prob.n_axes]) / n
        gk = (1j * GAMMA * dt) * np.einsum("n,ntd->td", ek, Xk[:, :, :prob.n_axes]) / n
        dJ = np.real(c * (gf - gk)) * sgn
        return -a, -prob.flatten(dJ)

    return fun


def _starts(prob, n_restarts, rng):
    """PGSE and OGSE warm starts, not noise.

    A cold random start projects into a smooth low-amplitude waveform and the ascent stalls in a
    local optimum -- which is what made the prototype's sup non-monotone in K. Diffusion
    waveforms are oscillatory, so start from oscillatory: one lobe (PGSE-like) upward through
    the OGSE family.
    """
    nf, na = prob.n_free, prob.n_axes
    edge = np.sin(np.linspace(0, np.pi, nf))
    for r in range(n_restarts):
        x0 = np.zeros((na, nf))
        for k in range(na):
            lobes = 1 + r + k
            x0[k] = (edge * np.sin(lobes * np.linspace(0, np.pi, nf) + rng.uniform(0, 0.5))
                     * prob.G_max * 0.7 / np.sqrt(na))
        yield x0.reshape(-1)


def _moment_basis(prob):
    """Orthonormal basis of the moment subspace the waveform must be orthogonal to."""
    t = np.arange(prob.n_t) * prob.dt
    cols = [prob.sign[:, 0]]
    for order, on in ((1, prob._null_M1), (2, prob._null_M2)):
        if on:
            cols.append((t ** order) * prob.sign[:, 0])
    Q, _ = np.linalg.qr(np.stack(cols, 1))
    return Q


def _solve(Xf, Xk, envelope, dt, steps, restarts, rng, warm_start=None, echo=None, n_jobs=1):
    """Take the sup by projected Adam ascent, all restarts advanced together.

    Not SQP, and not one restart at a time -- both for measured reasons.

    SQP is what the NOW designer wants, because its waveforms are ~140 samples, i.e. ~420
    variables, and SLSQP then holds the constraints exactly and cheaply. A certificate is taken
    over the WHOLE walk: 601-1001 samples, 1800-3000 variables, where the QP subproblem is
    O(n^3) and a single 40-iteration solve measured over 7 minutes at n_t=201. The feasible set
    is identical either way -- `WaveformProblem` exposes it as SQP constraints for NOW and as a
    projection here -- so only the interface differs, not what is being certified.

    Restarts are BATCHED rather than looped because the objective is memory-bandwidth bound, not
    compute bound: it streams the whole trajectory array per evaluation. Stacking R restarts
    turns the per-restart GEMV into one GEMM, so the walk is read once for all of them. Measured
    on a 20k-walker, 1001-step walk with 8 restarts: 396 ms/step looping on CPU, 1.4 ms/step
    batched on GPU -- 280x, which is what turns a full class x K sweep from hours into seconds.
    """
    prob = envelope.problem(Xf.shape[1], dt, echo=echo)
    starts = ([np.asarray(warm_start, float)] if warm_start is not None else [])
    starts += list(_starts(prob, restarts, rng))
    G0 = np.stack([prob.gradient(x) for x in starts if np.shape(x) == (prob.n_var,)])

    try:
        import jax
        import jax.numpy as jnp
        backend = jax.default_backend()
    except Exception:
        jax = None
        backend = "numpy"

    Q = _moment_basis(prob)
    mask = np.zeros((prob.n_t, 1)); mask[prob.free] = 1.0
    lim = prob.slew_rate_max * dt

    if jax is None:
        xp, jit = np, (lambda f: f)
    else:
        xp, jit = jnp, jax.jit


    # explicit, jittable projection (POCS: moments affine, slew a band, amplitude a box)
    def _proj(G):
        Qx = xp.asarray(Q); Mk = xp.asarray(mask)
        for _ in range(6):
            G = G - xp.einsum("tk,rkd->rtd", Qx, xp.einsum("tk,rtd->rkd", Qx, G))
            for _ in range(3):
                d = xp.clip(xp.diff(G, axis=1), -lim, lim)
                G = xp.concatenate([G[:, :1], G[:, :1] + xp.cumsum(d, axis=1)], axis=1)
            G = xp.clip(G, -prob.G_max, prob.G_max)
        G = G - xp.einsum("tk,rkd->rtd", Qx, xp.einsum("tk,rtd->rkd", Qx, G))
        return G * Mk[None]

    def _val_grad(G, Xf_, Xk_):
        pf = (GAMMA * dt) * xp.einsum("rtd,ntd->rn", G, Xf_)
        pk = (GAMMA * dt) * xp.einsum("rtd,ntd->rn", G, Xk_)
        ef, ek = xp.exp(1j * pf), xp.exp(1j * pk)
        Dv = ef.mean(1) - ek.mean(1)
        a = xp.abs(Dv)
        c = xp.conj(Dv) / xp.maximum(a, 1e-30)
        gf = (1j * GAMMA * dt) * xp.einsum("rn,ntd->rtd", ef, Xf_) / Xf_.shape[0]
        gk = (1j * GAMMA * dt) * xp.einsum("rn,ntd->rtd", ek, Xk_) / Xk_.shape[0]
        return a, xp.real(c[:, None, None] * (gf - gk)) * xp.asarray(prob.sign)[None]

    Xa, Xb = xp.asarray(Xf), xp.asarray(Xk)
    G = _proj(xp.asarray(G0, dtype=Xa.dtype))
    m = xp.zeros_like(G); v = xp.zeros_like(G)
    lr = prob.G_max * 0.05

    # Xa/Xb are ARGUMENTS, not closure captures: closed over, XLA treats the walk as a compile
    # -time constant and constant-folds a (n_walkers, n_t*3) complex array on every trace, which
    # cost more than the ascent it was compiling.
    @jit
    def _step(G, m, v, Xa, Xb):
        a, g = _val_grad(G, Xa, Xb)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        return _proj(G + lr * m / (xp.sqrt(v) + 1e-30)), m, v, a

    best = np.zeros(G.shape[0]); bestG = np.array(G, copy=True)
    for _ in range(int(steps)):
        G_scored = G                                      # `a` is the objective AT this iterate; `_step` returns the next
        G, m, v, a = _step(G, m, v, Xa, Xb)
        an = np.asarray(a)
        imp = an > best
        if imp.any():
            Gs = np.asarray(G_scored)
            best = np.where(imp, an, best)
            bestG[imp] = Gs[imp]
    j = int(np.argmax(best))
    return float(best[j]), prob.flatten(bestG[j]), prob


def sup_replay_error(traj, K, envelope, dt, *, steps=300, restarts=64, seed=0,
                     warm_start=None, decoded=None, holdout=False, echo=None, n_jobs=1):
    """The sup of Eq. (4): worst-case ``|S_full(G) - S_K(G)|`` over ``G_scanner``.

    Returns a :class:`SupResult` whose ``sup`` is a **lower bound** on the true sup -- ascent on
    a non-concave objective finds a local optimum, so more restarts can only raise it. It
    therefore PROVES K insufficient when it exceeds ``eps``, and is evidence rather than proof
    when it does not.

    ``restarts`` defaults high because batching made it free: the restarts advance together in
    one GEMM, so 64 of them cost the same wall clock as 6 (measured: 39.1 s vs 39.0 s on a
    601-step, 5k-walker walk). Sweeping Magnus from 6 to 64 restarts raised the K=16 sup from
    5.5e-2 to 5.3e-1 -- the low counts were simply missing the good optima -- while leaving the
    K that decides the certificate unmoved (K_min = 96 throughout) and turning the
    non-monotonicity warning off. A cheap search is the difference between a bound you can
    defend and one you hope holds.
    """
    X = np.asarray(traj, np.float32)
    Xk = _truncate(X, K) if decoded is None else np.asarray(decoded, np.float32)
    if Xk.shape != X.shape:
        raise ValueError(f"decoded walk {Xk.shape} does not match trajectory {X.shape}")
    dt = float(dt)
    best, x, prob = _solve(X, Xk, envelope, dt, steps, restarts,
                           np.random.default_rng(seed), warm_start, echo, n_jobs)

    ho = None
    if holdout:
        h = X.shape[0] // 2
        if h < 1:
            raise ValueError("holdout needs at least 2 walkers")
        _, xa, pa = _solve(X[:h], Xk[:h], envelope, dt, steps, restarts,
                           np.random.default_rng(seed + 1), warm_start, echo, n_jobs)
        if xa is not None:
            fb = _objective(pa, X[h:], Xk[h:], dt)
            ho = -fb(xa)[0]

    G = prob.effective(x) if x is not None else None
    b = None
    if G is not None:
        q = GAMMA * dt * np.cumsum(G, axis=0)
        b = float(dt * (q ** 2).sum())
    return SupResult(float(best), ho, G, int(X.shape[0]), b), x


def k_min(traj, envelope, dt, *, eps=5e-3, K_grid=(8, 16, 32, 48, 64, 96, 128, 192, 256),
          steps=300, restarts=64, seed=0, holdout=False, echo=None, n_jobs=1, verbose=False):
    """Smallest K in ``K_grid`` whose certificate meets ``eps``, with the per-K evidence.

    K is decided by ``sup`` (the certificate), never by ``held_out``: the pack has to be safe
    for the waveform a consumer actually sends, not for an average one. Each K warm-starts from
    the previous winner -- that waveform stays feasible -- so the estimate is near-monotone. A
    rise in the reported sup as K grows is impossible for the true sup (the residual only
    shrinks), so it is warned about rather than smoothed away.
    """
    X = np.asarray(traj, np.float64)
    sups, x, prev, prev_K = {}, None, None, None
    for K in sorted(K_grid):
        if K >= X.shape[1] - 2:
            continue
        r, x = sup_replay_error(X, K, envelope, dt, steps=steps, restarts=restarts, seed=seed,
                                warm_start=x, holdout=holdout, echo=echo, n_jobs=n_jobs)
        if prev is not None and r.sup > prev * (1 + 1e-6):
            warnings.warn(
                f"sup rose from {prev:.3e} (K={prev_K}) to {r.sup:.3e} (K={K}); the true sup "
                f"cannot rise with K, so the search is landing in different local optima. "
                f"Raise `restarts`/`steps` before trusting K_min from this run.",
                RuntimeWarning, stacklevel=2)
        sups[K], prev, prev_K = r, r.sup, K
        if verbose:
            ho = "" if r.held_out is None else f" held_out={r.held_out:.3e}"
            print(f"  K={K:4d}  sup={r.sup:.3e}{ho}  {'ok' if r.sup <= eps else 'FAIL'}",
                  flush=True)
        if r.sup <= eps:
            return K, sups
    return None, sups


# ------------------------------------------------------------- certified bandwidth
def k_for_walk(f_c, T, dt_save=None):
    """Modes a walk of duration ``T`` needs to carry a certified bandwidth ``f_c`` [Hz].

    Mode ``m`` of the bridge is the sine at ``f_m = m/(2T)``, so ``K = ceil(2 T f_c)`` and the
    mode count is just the bandwidth expressed on this particular walk's grid. Raises if the
    walk cannot represent ``f_c`` at all: a pack saved at ``dt_save`` has Nyquist
    ``1/(2 dt_save)`` and cannot certify above it however strong the scanner.
    """
    if dt_save is not None and f_c > 1.0 / (2.0 * dt_save) + 1e-9:
        raise ValueError(
            f"f_c = {f_c:.0f} Hz exceeds the walk's Nyquist {1/(2*dt_save):.0f} Hz "
            f"(dt_save = {dt_save*1e3:.3g} ms): the trajectory does not resolve it, so no K "
            f"certifies it. Re-walk with a finer dt_save.")
    return int(np.ceil(2.0 * float(T) * float(f_c)))


@dataclass
class Certificate:
    """What a pack promises, in units that do not depend on how long the walk was.

    ``f_c`` is the certified BANDWIDTH in Hz. It is the invariant: mode ``m`` sits at
    ``f_m = m/(2T)``, so a mode count means nothing without the duration it was measured over,
    while ``f_c`` transfers between walks unchanged. Measured on one substrate at T = 15/30/60 ms
    a Prisma certifies at 133 Hz with K = 4/8/16 -- the same bandwidth, three different K.

    So a pack declares ``f_c`` and each consumer derives ``K = ceil(2 T f_c)`` for their own
    walk, and the admissibility question a user actually asks -- "can I run this waveform?" --
    becomes a comparison of bandwidths with no reference to duration at all.
    """

    f_c: float                      # Hz, the certified bandwidth
    K: int                          # modes, on the walk this was measured over
    T: float                        # s, that walk's duration
    eps: float
    envelope: dict
    nyquist: float                  # Hz, 1/(2 dt_save) -- the ceiling f_c can never pass
    sups: dict = field(default_factory=dict)
    n_walkers: int = 0

    def k_for(self, T, dt_save=None):
        """Modes another walk of duration ``T`` needs to carry this same certificate."""
        return k_for_walk(self.f_c, T, dt_save)

    def admits(self, f_hz):
        """Whether a waveform whose highest meaningful frequency is ``f_hz`` is covered."""
        return float(f_hz) <= self.f_c

    def describe(self):
        return dict(certified_bandwidth_hz=self.f_c, K_at_measurement=self.K, T_s=self.T,
                    eps=self.eps, nyquist_hz=self.nyquist, n_walkers=self.n_walkers,
                    envelope=self.envelope)


def certify(traj, envelope, dt, *, eps=5e-3, K_grid=(4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256),
            subsample=None, seed=0, **kw):
    """Certify a walk against ``envelope``: the bandwidth it can be replayed at within ``eps``.

    ``subsample`` takes a random subset of walkers for the SEARCH. The codec error is a property
    of K and the substrate rather than of the ensemble size, so a few thousand walkers locate
    the right K far more cheaply than the full pack -- but the returned certificate is the sup
    over the walkers actually passed, so re-run without ``subsample`` to certify a pack for
    publication. Fewer walkers also give the adversary more room to fit the particular
    realisation, so a subsampled search is conservative, not optimistic.
    """
    X = np.asarray(traj)
    if subsample and subsample < X.shape[0]:
        idx = np.random.default_rng(seed).choice(X.shape[0], int(subsample), replace=False)
        X = np.ascontiguousarray(X[np.sort(idx)])
    T = (X.shape[1] - 1) * float(dt)
    if X.shape[0] < _N_SUP_CONVERGED:
        warnings.warn(
            f"certifying from {X.shape[0]:,} walkers, below the ~{_N_SUP_CONVERGED:,} where the "
            f"sup was measured to converge. The adversary partly fits the particular realisation, "
            f"so f_c here is INFLATED, not merely noisy (measured 1.5-4x on 3-pool white matter), "
            f"and K will be over-provisioned with nothing else to signal it. A pack built to a "
            f"5e-3 Monte-Carlo floor carries ~116,000 walkers and clears this comfortably; "
            f"diagnostic walks do not.", RuntimeWarning, stacklevel=2)
    K, sups = k_min(X, envelope, dt, eps=eps, K_grid=K_grid, seed=seed, **kw)
    if K is None:
        raise ValueError(
            f"no K in {tuple(K_grid)} certifies {envelope.name} at eps={eps:g} on this walk "
            f"(best {min(r.sup for r in sups.values()):.3e} at K={max(sups)}). Extend K_grid, "
            f"loosen eps, or declare this scanner class out of scope for the pack.")
    return Certificate(f_c=K / (2.0 * T), K=int(K), T=T, eps=float(eps),
                       envelope=envelope.describe(), nyquist=1.0 / (2.0 * float(dt)),
                       sups=sups, n_walkers=int(X.shape[0]))


# ─────────────────────────────────────────────────────────── per-pool certificates
# The sup converges in walker count, and below this it is INFLATED rather than merely noisy:
# measured on a 3-pool CACTUS walk (connectom / prisma / magnus, mixture), f_c reads
# 480/1918/>3836 Hz at N=1000 and 320/320/959 Hz once converged, so an under-sampled certificate
# silently over-provisions K by 1.5-4x with nothing to signal it. Convergence lands by ~5,000 on
# every class measured. A pack built to a Monte-Carlo floor of sigma* = 5e-3 needs ~116,000
# walkers (the floor falls as 1/sqrt(N); 0.0155 measured at 12,042), so production packs clear
# this by more than 20x -- it is diagnostic walks that get caught.
_N_SUP_CONVERGED = 5000


@dataclass
class PoolCertificate:
    """What a pack promises for the WHOLE ensemble and for each subset a consumer can select.

    A single mixture certificate is not enough, because the mixture is genuinely easier than its
    parts. Two effects stack. The obvious one is dilution: a pool's error enters the pack signal
    weighted by its walker fraction. The subtler one is that the adversary must choose ONE
    waveform, and each pool's worst case is a different waveform, so

        sup_G |sum_p w_p e_p(G)|  <  sum_p w_p sup_G |e_p(G)|

    strictly, whenever the argmaxes differ. Measured on 3-pool white matter at K=48 (Prisma):
    pools 6.1e-3 (extra) and 7.1e-3 (intra) give a diluted sum of 5.9e-3, against a mixture sup
    of 3.7e-3 -- another 37% below, which is the pools' errors partly cancelling in phase.

    So a consumer who filters to one compartment is NOT covered by the mixture row: on that same
    class the pools need 480 Hz where the mixture needs 240. Packs carry compartment labels and
    support filtered replay, so the certificate is a table and ``k_store`` is sized by its worst
    row, not by the ensemble.
    """

    rows: dict                      # name -> Certificate ("mixture" plus one per pool)
    n_walkers: int

    @property
    def binding(self):
        """The row that sizes the pack: the worst selectable subset, not the ensemble."""
        return max(self.rows.items(), key=lambda kv: kv[1].f_c)

    @property
    def f_c(self):
        return self.binding[1].f_c

    def for_pool(self, name):
        """The certificate a consumer replaying only ``name`` is entitled to quote."""
        if name not in self.rows:
            raise KeyError(f"no certificate for {name!r}; have {sorted(self.rows)}")
        return self.rows[name]

    def k_store(self, T, dt_save=None):
        """Modes a pack over ``T`` must store so EVERY selectable subset is covered."""
        return k_for_walk(self.f_c, T, dt_save)

    def describe(self):
        name, cert = self.binding
        return dict(binding_pool=name, f_c_hz=self.f_c, n_walkers=self.n_walkers,
                    rows={k: v.describe() for k, v in self.rows.items()})


def certify_pools(traj, envelope, dt, *, labels, names=None, eps=5e-3, min_walkers=200, **kw):
    """Certify the mixture AND each selectable pool, returning the table a pack should carry.

    ``labels`` is the per-walker compartment id (the pack's ``comp``/``comp0`` channel); ``names``
    maps id -> name, defaulting to the id. Pools smaller than ``min_walkers`` are skipped rather
    than certified badly.

    Sizing a pack from the mixture alone under-certifies compartment-filtered replay -- see
    :class:`PoolCertificate`. Use ``result.k_store(T)``, which reads the worst row.
    """
    traj = np.asarray(traj)
    labels = np.asarray(labels)
    if len(labels) != len(traj):
        raise ValueError(f"labels {labels.shape} do not match {len(traj)} walkers")
    names = names or {}
    rows = {"mixture": certify(traj, envelope, dt, eps=eps, **kw)}
    for lab in np.unique(labels):
        m = labels == lab
        if int(m.sum()) < min_walkers:
            warnings.warn(
                f"pool {names.get(int(lab), int(lab))!r} has {int(m.sum())} walkers, below "
                f"min_walkers={min_walkers}; skipped rather than certified from too few.",
                RuntimeWarning, stacklevel=2)
            continue
        nm = names.get(int(lab), str(int(lab)))
        try:
            rows[nm] = certify(np.ascontiguousarray(traj[m]), envelope, dt, eps=eps, **kw)
        except ValueError as e:
            raise ValueError(f"pool {nm!r}: {e}") from None
    return PoolCertificate(rows=rows, n_walkers=int(len(traj)))
