"""Eq. (4): certify the band of a replay pack against the deliverable waveform set.

A replay pack stores the lowest ``K`` temporal bands of each walker's Brownian bridge on the
window's continuous sine basis, so K is not a storage knob but a **claim**: that no waveform the
consumer can physically produce reads the band-limited walk differently from the full one by
more than ``eps``,

    K_min = min K  s.t.  sup_{G in G_scanner} |S_full(G) - S_K(G)| <= eps                (4)

The claim has to be settled at build time, because the failure is silent. A waveform outside
the certified band replays to a smooth, plausible, wrong number -- no NaN, no warning, no shape
anomaly -- so a consumer cannot tell from their own output that they have left the band. Either
the pack carries the answer or nobody has it.

This lives in dmipy-design, not dmipy-sim, because ``G_scanner`` does: the module that decides
which acquisitions are deliverable is the one that can say what replaying its own designs
costs. It shares :func:`dmipy_design.constraints.waveform_problem` with the NOW designer, and
that sharing is load-bearing rather than tidy -- a certificate taken over a SMALLER set than
the designer can reach is unsound, because a user could then design a waveform the pack was
never certified for.

A pack has no raw walk: what it stores IS the band-limited walk. The certificate therefore has
two parts, and they are stated separately.

*Beyond the stored band* the error is walk-free: the bands beyond ``K`` of a free walk at the
pack's diffusivity carry the phase variance ``gamma^2 sum_{k > K} W_k^2 var_k`` with
``var_k = 2 D T^2 / (pi k)^2`` (:func:`dmipy_sim.replay.compression.band_variances`; a wall only
lowers a band's variance), and a dropped independent phase costs the signal half of it. The pack
states this over its envelope's battery as ``fidelity.terms.truncation`` (RPK.md 9.5.5), and a
certificate quotes it: a pack cannot be certified tighter than its own band.

*Within the stored band* the sup is measured on the pack's own walkers: the stored ``K`` bands
against their lowest ``K'``, over every waveform the scanner can deliver,

    |S_K - S_K'| = | < exp(i phi_K') (exp(i d_phi) - 1) > |,   d_phi = gamma sum_{K' < k <= K} W_k . beta_k .

Only the bands above ``K'`` create the error, but every band steers it: ``phi_K'`` decides
whether the per-walker errors add or cancel, so the adversary uses the low bands to align the
ensemble and the high bands to break it, which is why a one-mode-at-a-time scan understates the
sup by one to two orders of magnitude -- worst exactly where the per-mode amplitude ceiling is
flat (preclinical inserts), where the tail behaves as ``sum 1/m`` rather than ``sum 1/m^2``. The
constraints are handled exactly: the slew band, amplitude box, refocusing and moment nulls are
held as constraints rather than penalties, so the adversary rides ``|G| = G_max`` and
``|dG/dt| = slew_max`` instead of stopping short of them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from dmipy_sim.constants import GAMMA
from .constraints import waveform_problem

__all__ = ["ReplayEnvelope", "replay_envelope", "BandWalk", "SupResult", "Certificate",
           "PoolCertificate", "sup_replay_error", "k_min", "certify",
           "certify_pools", "k_for_walk"]


@dataclass(frozen=True)
class ReplayEnvelope:
    """A named scanner class, as the deliverable-waveform set Eq. (4) takes its sup over.

    ``raster`` (s) is the grid the adversary's waveforms are sampled on: the class's gradient raster.
    """

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

        ``echo=None`` means the walk is read by an ALREADY-EFFECTIVE waveform (the 180 folded in, as a pack's
        replay reads ``G_eff``), so the sign is +1 throughout and the refocusing constraint reduces to nulling
        M0. Pass an echo index for a literal spin-echo layout.
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


@dataclass(frozen=True)
class BandWalk:
    """A walk as the bands a pack stores: the ensemble a certificate is taken over.

    ``C`` ``(n_walkers, K + 2, 3)`` float32 holds per axis the window's exact start and displacement, then its
    ``K`` sine bands on the continuous basis (``pack.position_coeffs``, ``CoefficientWalk.position_coeffs``);
    ``T`` (s) is the window the bands are on; ``D`` (m^2/s) the walk's diffusivity (None when unrecorded);
    ``labels`` the pool label per walker (None on a walk without a static one); ``truncation`` the walk-free
    error a pack states beyond its band over its envelope's battery (``fidelity.terms.truncation``, RPK.md
    9.5.5; None for a walk not yet packed). Of a walk stored in windows this is window 0: each window is
    certified on its own (RPK.md 4.3).
    """

    C: np.ndarray
    T: float
    D: float | None = None
    labels: np.ndarray | None = None
    truncation: float | None = None

    @property
    def K(self):
        return int(self.C.shape[1]) - 2

    @property
    def n_walkers(self):
        return int(self.C.shape[0])

    @classmethod
    def of(cls, walk):
        """``walk`` as a :class:`BandWalk`: a :class:`~dmipy_sim.replay.ReplayPack`, a
        :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`, or one already."""
        if isinstance(walk, cls):
            return walk
        from dmipy_sim.replay import ReplayPack
        if isinstance(walk, ReplayPack):
            labels = walk.arrays.get("comp_static")
            terms = (walk.fidelity or {}).get("terms") or {}
            trunc = terms.get("truncation")
            return cls(np.ascontiguousarray(walk.position_coeffs, dtype=np.float32), float(walk.T_window),
                       None if walk.diffusivity is None else float(walk.diffusivity),
                       None if labels is None else np.asarray(labels), None if trunc is None else float(trunc))
        from dmipy_sim.coefficient_walk import CoefficientWalk
        if isinstance(walk, CoefficientWalk):
            lab = walk.compartment
            return cls(np.ascontiguousarray(np.asarray(walk.position_coeffs)[:, 0], dtype=np.float32),
                       float(walk.T_window), None if walk.diffusivity is None else float(walk.diffusivity),
                       lab if (lab is not None and np.ndim(lab) == 1) else None, None)
        raise TypeError(f"a certificate is taken over a ReplayPack, a CoefficientWalk or a BandWalk; got "
                        f"{type(walk).__name__}")

    def take(self, idx):
        """The walkers ``idx`` (an index array or boolean mask) as a :class:`BandWalk` of their own."""
        return BandWalk(np.ascontiguousarray(self.C[idx]), self.T, self.D,
                        None if self.labels is None else np.asarray(self.labels)[idx], self.truncation)


@dataclass
class SupResult:
    """The sup of Eq. (4) within the stored band -- and the numbers beside it, which answer different questions.

    ``sup`` is the CERTIFICATE: the largest ``|S_K - S_K'|`` found over the deliverable set, with ``K`` the
    stored band and ``K'`` the band asked about. The ascent runs on the walk's own walkers and is scored on
    them, which is exactly what a consumer experiences, because the pack *is* the ensemble: they replay G
    against these walkers. No waveform does worse on this pack.

    ``held_out`` is the typical case: fitted on half the walkers, scored on the other half, so a
    waveform tuned to this particular realisation gets no credit for it. It runs 4-20x below the
    certificate and the gap narrows with N -- the adversary partly memorises the ensemble, and
    more walkers leave it less room. Certify on ``sup``; quote ``held_out`` so the gap is
    visible rather than implied.

    ``bound`` is the walk-free statement of the same error at the winning waveform, what a pack's own
    certificate states for a band (RPK.md 9.5.5): half the phase variance the bands ``K' < k <= K`` carry for a
    free walk at the pack's diffusivity (:func:`dmipy_sim.replay.compression.band_variances`), the bias of the
    ensemble-mean magnitude, which a restricted walk's bands stay below. The measured ``sup`` on a finite
    ensemble also carries the realisation's noise the adversary fits, so below the Monte-Carlo floor it sits
    above ``bound`` and converges onto it with the walker count. None when the walk records no diffusivity.
    """

    sup: float
    held_out: float | None = None
    waveform: np.ndarray | None = None
    n_walkers: int = 0
    b_value: float | None = None
    bound: float | None = None

    def __float__(self):
        return float(self.sup)

    def __le__(self, other):
        return self.sup <= float(other)

    def __gt__(self, other):
        return self.sup > float(other)


def _projection(n_t, dt, T, K):
    """``(K + 2, n_t)``: a unit sample held over its interval, on the window's bands, times ``gamma`` -- the matrix
    a pack's replay contracts a waveform with (:func:`dmipy_sim.replay.compression.band_projection_matrix`)."""
    from dmipy_sim.replay.compression import band_projection_matrix
    return GAMMA * band_projection_matrix(int(n_t), float(dt), float(T), int(K))


def _objective(prob, C, K, P):
    """``-|S_stored - S_K|`` and its analytic gradient in the flat variable layout: the stored bands of ``C``
    against their lowest ``K``, the waveform entering through ``P`` (:func:`_projection`)."""
    n = C.shape[0]
    sgn = prob.sign
    Cf = np.asarray(C, np.float64)[:, :, :prob.n_axes]              # (n_w, K_stored + 2, n_axes)
    keep = np.zeros(P.shape[0]); keep[:2 + int(K)] = 1.0

    def fun(x):
        Geff = prob.effective(x)                                     # (n_t, n_axes), 180 folded in
        W = P @ Geff                                                 # (K_stored + 2, n_axes)
        pf = np.einsum("ikc,kc->i", Cf, W)
        pk = np.einsum("ikc,kc->i", Cf, W * keep[:, None])
        ef, ek = np.exp(1j * pf), np.exp(1j * pk)
        D = ef.mean() - ek.mean()
        a = abs(D)
        if a < 1e-300:
            return 0.0, np.zeros(prob.n_var)
        c = np.conj(D) / a
        # d(mean exp(i phi))/dG[t,d] = i sign[t] sum_k P[k,t] <exp(i phi) C[:,k,d]> / N
        Af = np.einsum("i,ikc->kc", ef, Cf) / n
        Ak = np.einsum("i,ikc->kc", ek, Cf) * keep[:, None] / n
        dJ = np.real(c * 1j * (P.T @ (Af - Ak))) * sgn
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


def _solve(C, K, P, envelope, dt, steps, restarts, rng, warm_start=None, echo=None):
    """Take the sup by projected Adam ascent, all restarts advanced together.

    Not SQP, and not one restart at a time -- both for measured reasons.

    SQP is what the NOW designer wants, because its waveforms are ~140 samples, i.e. ~420
    variables, and SLSQP then holds the constraints exactly and cheaply. A certificate is taken
    over the WHOLE window on the class's raster: thousands of samples, where the QP subproblem is
    O(n^3) and a single 40-iteration solve measured over 7 minutes at 201 samples. The feasible set
    is identical either way -- `WaveformProblem` exposes it as SQP constraints for NOW and as a
    projection here -- so only the interface differs, not what is being certified.

    Restarts are BATCHED rather than looped: the objective reads the whole band array per
    evaluation, so stacking R restarts turns the per-restart GEMV into one GEMM and the walk is
    read once for all of them. The waveform enters only through its band projection ``P G``
    ``(K + 2, n_axes)``, so the walk is contracted in band space, never decoded onto the grid.
    """
    n_t = P.shape[1]
    prob = envelope.problem(n_t, dt, echo=echo)
    starts = ([np.asarray(warm_start, float)] if warm_start is not None else [])
    starts += list(_starts(prob, restarts, rng))
    G0 = np.stack([prob.gradient(x) for x in starts if np.shape(x) == (prob.n_var,)])

    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        jax = None

    Q = _moment_basis(prob)
    mask = np.zeros((prob.n_t, 1)); mask[prob.free] = 1.0
    lim = prob.slew_rate_max * dt
    keep = np.zeros(P.shape[0], np.float32); keep[:2 + int(K)] = 1.0

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

    def _val_grad(G, C_, P_, keep_):
        W = xp.einsum("kt,rtd->rkd", P_, G * xp.asarray(prob.sign)[None])         # (R, K+2, n_axes)
        pf = xp.einsum("rkd,ikd->ri", W, C_)
        pk = xp.einsum("rkd,ikd->ri", W * keep_[None, :, None], C_)
        ef, ek = xp.exp(1j * pf), xp.exp(1j * pk)
        Dv = ef.mean(1) - ek.mean(1)
        a = xp.abs(Dv)
        c = xp.conj(Dv) / xp.maximum(a, 1e-30)
        Af = xp.einsum("ri,ikd->rkd", ef, C_) / C_.shape[0]
        Ak = xp.einsum("ri,ikd->rkd", ek, C_) * keep_[None, :, None] / C_.shape[0]
        g = xp.einsum("kt,rkd->rtd", P_, 1j * (Af - Ak))
        return a, xp.real(c[:, None, None] * g) * xp.asarray(prob.sign)[None]

    Ca = xp.asarray(np.asarray(C, np.float32)[:, :, :prob.n_axes])
    Pa = xp.asarray(np.asarray(P, np.float32))
    ka = xp.asarray(keep)
    G = _proj(xp.asarray(G0, dtype=Ca.dtype))
    m = xp.zeros_like(G); v = xp.zeros_like(G)
    lr = prob.G_max * 0.05

    # the walk and the projection are ARGUMENTS, not closure captures: closed over, XLA treats them as compile
    # -time constants and constant-folds the band array on every trace, which cost more than the ascent.
    @jit
    def _step(G, m, v, Ca, Pa, ka):
        a, g = _val_grad(G, Ca, Pa, ka)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        return _proj(G + lr * m / (xp.sqrt(v) + 1e-30)), m, v, a

    best = np.zeros(G.shape[0]); bestG = np.array(G, copy=True)
    for _ in range(int(steps)):
        G_scored = G                                      # `a` is the objective AT this iterate; `_step` returns the next
        G, m, v, a = _step(G, m, v, Ca, Pa, ka)
        an = np.asarray(a)
        imp = an > best
        if imp.any():
            Gs = np.asarray(G_scored)
            best = np.where(imp, an, best)
            bestG[imp] = Gs[imp]
    j = int(np.argmax(best))
    return float(best[j]), prob.flatten(bestG[j]), prob


def _free_bound(G, dt, T, K_stored, K, D):
    """Half the phase variance the bands ``K < k <= K_stored`` carry for the effective waveform ``G`` ``(n_t,
    n_axes)`` on a free walk at ``D`` (:func:`dmipy_sim.replay.compression.band_variances`): the walk-free
    statement of ``|S_stored - S_K|`` at this waveform."""
    from dmipy_sim.replay.compression import band_projection, band_variances
    W = band_projection(np.asarray(G, np.float64)[None], dt, T, int(K_stored))[0, 2 + int(K):, :]
    return float(0.5 * GAMMA ** 2 * ((W ** 2).sum(axis=1) * band_variances(int(K_stored), T, D)[int(K):]).sum())


def _grid(walk, envelope, dt):
    """``(dt, n_t)``: the raster the adversary's waveforms are sampled on (the envelope's unless given) and the
    sample-and-hold samples that cover the window."""
    dt = float(envelope.raster if dt is None else dt)
    n_t = int(round(float(walk.T) / dt))
    if n_t < 2:
        raise ValueError(f"the window T = {walk.T:.6g} s holds fewer than two samples at the raster {dt:.3g} s")
    return dt, n_t


def sup_replay_error(walk, K, envelope, *, dt=None, steps=300, restarts=64, seed=0,
                     warm_start=None, holdout=False, echo=None):
    """The sup of Eq. (4) within the stored band: worst-case ``|S_stored(G) - S_K(G)|`` over ``G_scanner``.

    ``walk`` is a :class:`~dmipy_sim.replay.ReplayPack`, a :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`
    or a :class:`BandWalk`; ``K`` is a band below the stored one (the stored band against itself is zero by
    construction, and what lies beyond it is the pack's walk-free term, :class:`Certificate`). The adversary's
    waveforms are sampled on the envelope's raster (``dt`` overrides it).

    Returns ``(SupResult, x)`` with ``x`` the winning waveform in the problem's flat layout (a warm start for
    the next band). ``SupResult.sup`` is a **lower bound** on the true sup -- ascent on a non-concave
    objective finds a local optimum, so more restarts can only raise it. It therefore PROVES K insufficient
    when it exceeds ``eps``, and is evidence rather than proof when it does not.

    ``restarts`` defaults high because batching made it free: the restarts advance together in one GEMM, so
    64 of them cost the same wall clock as 6. Sweeping Magnus from 6 to 64 restarts raised the K=16 sup from
    5.5e-2 to 5.3e-1 -- the low counts were simply missing the good optima -- while leaving the K that decides
    the certificate unmoved (K_min = 96 throughout) and turning the non-monotonicity warning off. A cheap
    search is the difference between a bound you can defend and one you hope holds.
    """
    bw = BandWalk.of(walk)
    K = int(K)
    if not 0 <= K < bw.K:
        raise ValueError(f"K = {K} is not below the stored band K = {bw.K}: within the stored band the sup is taken "
                         f"of the stored bands against their lowest K, and beyond it the error is the pack's "
                         f"walk-free truncation term")
    dt, n_t = _grid(bw, envelope, dt)
    P = _projection(n_t, dt, bw.T, bw.K)
    best, x, prob = _solve(bw.C, K, P, envelope, dt, steps, restarts, np.random.default_rng(seed), warm_start, echo)

    ho = None
    if holdout:
        h = bw.n_walkers // 2
        if h < 1:
            raise ValueError("holdout needs at least 2 walkers")
        _, xa, pa = _solve(bw.C[:h], K, P, envelope, dt, steps, restarts,
                           np.random.default_rng(seed + 1), warm_start, echo)
        if xa is not None:
            fb = _objective(pa, bw.C[h:], K, P)
            ho = -fb(xa)[0]

    G = prob.effective(x) if x is not None else None
    b = bound = None
    if G is not None:
        q = GAMMA * dt * np.cumsum(G, axis=0)
        b = float(dt * (q ** 2).sum())
        if bw.D is not None:
            bound = _free_bound(G, dt, bw.T, bw.K, K, bw.D)
    return SupResult(float(best), ho, G, bw.n_walkers, b, bound), x


def k_min(walk, envelope, *, eps=5e-3, K_grid=(8, 16, 32, 48, 64, 96, 128, 192, 256),
          dt=None, steps=300, restarts=64, seed=0, holdout=False, echo=None, verbose=False):
    """Smallest K in ``K_grid`` below the stored band whose measured sup meets ``eps``, with the per-K evidence.

    K is decided by ``sup`` (the certificate), never by ``held_out``: the pack has to be safe
    for the waveform a consumer actually sends, not for an average one. Each K warm-starts from
    the previous winner -- that waveform stays feasible -- so the estimate is near-monotone. A
    rise in the reported sup as K grows is impossible for the true sup (the residual only
    shrinks), so it is warned about rather than smoothed away. ``(None, sups)`` when no band of
    the grid meets ``eps``.
    """
    bw = BandWalk.of(walk)
    sups, x, prev, prev_K = {}, None, None, None
    for K in sorted(set(int(k) for k in K_grid)):
        if K >= bw.K:
            continue
        r, x = sup_replay_error(bw, K, envelope, dt=dt, steps=steps, restarts=restarts, seed=seed,
                                warm_start=x, holdout=holdout, echo=echo)
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
def k_for_walk(T, *, scanner=None, bandwidth_hz=None):
    """Bands a window of duration ``T`` (s) needs to carry a band: ``K = ceil(2 T f)``, the one derivation every
    producer reads (:func:`dmipy_sim.replay.bank.mode_count`), with ``f`` the storage band of the scanner class
    ``scanner`` (its certified band times the storage margin, dmipy-sim#643; a class the catalogue holds no
    certified band for is refused by name) or a stated ``bandwidth_hz``. Mode ``k`` of the bridge is the sine at
    ``f_k = k / (2T)``, so the band count is just a bandwidth expressed on this particular window."""
    from dmipy_sim.replay.bank import mode_count
    if scanner is None and bandwidth_hz is None:
        raise ValueError("give the scanner class whose band the walk carries, or a bandwidth_hz")
    return int(mode_count(float(T), scanner, temporal_bandwidth_hz=bandwidth_hz))


@dataclass
class Certificate:
    """What a pack promises, in units that do not depend on how long the window was.

    ``f_c`` is the certified BANDWIDTH in Hz. It is the invariant: mode ``k`` sits at
    ``f_k = k/(2T)``, so a band count means nothing without the window it was measured over,
    while ``f_c`` transfers between walks unchanged. Measured on one substrate at T = 15/30/60 ms
    a Prisma certifies at 133 Hz with K = 4/8/16 -- the same bandwidth, three different K.

    So a pack declares ``f_c`` and each consumer derives ``K = ceil(2 T f_c)`` for their own
    window, and the admissibility question a user actually asks -- "can I run this waveform?" --
    becomes a comparison of bandwidths with no reference to duration at all.

    ``K_stored`` is the band the walk was certified from, and ``truncation`` the walk-free error the pack
    states beyond it over its envelope's battery (``fidelity.terms.truncation``): the part of Eq. (4) no
    measurement within the pack reaches, quoted beside the measured sups.
    """

    f_c: float                      # Hz, the certified bandwidth
    K: int                          # bands, on the window this was measured over
    T: float                        # s, that window's duration
    eps: float
    envelope: dict
    K_stored: int                   # bands the walk stores
    truncation: float | None = None   # the pack's walk-free term beyond K_stored (None for an unpacked walk)
    sups: dict = field(default_factory=dict)
    n_walkers: int = 0

    def k_for(self, T):
        """Bands another window of duration ``T`` needs to carry this same certificate."""
        return k_for_walk(T, bandwidth_hz=self.f_c)

    def admits(self, f_hz):
        """Whether a waveform whose highest meaningful frequency is ``f_hz`` is covered."""
        return float(f_hz) <= self.f_c

    def describe(self):
        return dict(certified_bandwidth_hz=self.f_c, K_at_measurement=self.K, T_s=self.T,
                    eps=self.eps, K_stored=self.K_stored, truncation=self.truncation,
                    n_walkers=self.n_walkers, envelope=self.envelope)


def certify(walk, envelope, *, eps=5e-3, K_grid=(4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256),
            subsample=None, seed=0, **kw):
    """Certify a walk against ``envelope``: the bandwidth it can be replayed at within ``eps``.

    The bands of ``K_grid`` below the stored one are measured (:func:`k_min`); when none meets ``eps`` the
    stored band itself is the certificate, which holds when the pack's walk-free truncation term beyond it is
    within ``eps`` -- a pack cannot be certified tighter than its own band, and a walk not yet packed states
    no such term, so it is refused rather than certified at a band nothing bounds.

    ``subsample`` takes a random subset of walkers for the SEARCH. The codec error is a property
    of K and the substrate rather than of the ensemble size, so a few thousand walkers locate
    the right K far more cheaply than the full pack -- but the returned certificate is the sup
    over the walkers actually passed, so re-run without ``subsample`` to certify a pack for
    publication. Fewer walkers also give the adversary more room to fit the particular
    realisation, so a subsampled search is conservative, not optimistic.
    """
    bw = BandWalk.of(walk)
    if subsample and subsample < bw.n_walkers:
        idx = np.random.default_rng(seed).choice(bw.n_walkers, int(subsample), replace=False)
        bw = bw.take(np.sort(idx))
    T = float(bw.T)
    if bw.n_walkers < _N_SUP_CONVERGED:
        warnings.warn(
            f"certifying from {bw.n_walkers:,} walkers, below the ~{_N_SUP_CONVERGED:,} where the "
            f"sup was measured to converge. The adversary partly fits the particular realisation, "
            f"so f_c here is INFLATED, not merely noisy (measured 1.5-4x on 3-pool white matter), "
            f"and K will be over-provisioned with nothing else to signal it. A pack built to a "
            f"5e-3 Monte-Carlo floor carries ~116,000 walkers and clears this comfortably; "
            f"diagnostic walks do not.", RuntimeWarning, stacklevel=2)
    K, sups = k_min(bw, envelope, eps=eps, K_grid=K_grid, seed=seed, **kw)
    if K is None:
        tried = f"no K in {tuple(k for k in sorted(K_grid) if k < bw.K)} below the stored band certifies "
        if bw.truncation is None:
            raise ValueError(
                f"{tried}{envelope.name} at eps={eps:g} on this walk, and the walk states no truncation term beyond "
                f"its stored band K = {bw.K} (it is not a pack): build the pack, whose certificate bounds that band, "
                f"or extend K_grid below it")
        if bw.truncation > eps:
            raise ValueError(
                f"{tried}{envelope.name} at eps={eps:g} on this walk, and the pack's own band K = {bw.K} does not "
                f"either: its walk-free truncation term is {bw.truncation:.3e} > eps. Loosen eps, or walk the "
                f"substrate at a wider band.")
        K = bw.K
    return Certificate(f_c=K / (2.0 * T), K=int(K), T=T, eps=float(eps), envelope=envelope.describe(),
                       K_stored=int(bw.K), truncation=bw.truncation, sups=sups, n_walkers=int(bw.n_walkers))


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

    def k_store(self, T):
        """Bands a window of ``T`` must store so EVERY selectable subset is covered."""
        return k_for_walk(T, bandwidth_hz=self.f_c)

    def describe(self):
        name, cert = self.binding
        return dict(binding_pool=name, f_c_hz=self.f_c, n_walkers=self.n_walkers,
                    rows={k: v.describe() for k, v in self.rows.items()})


def certify_pools(walk, envelope, *, labels=None, names=None, eps=5e-3, min_walkers=200, **kw):
    """Certify the mixture AND each selectable pool, returning the table a pack should carry.

    ``labels`` is the per-walker compartment id, by default the walk's own static pool label (a pack's
    ``comp_static`` column, a coefficient walk's ``compartment``); a walk whose walkers change pool carries
    none and must be given one. ``names`` maps id -> name, defaulting to the id. Pools smaller than
    ``min_walkers`` are skipped rather than certified badly.

    Sizing a pack from the mixture alone under-certifies compartment-filtered replay -- see
    :class:`PoolCertificate`. Use ``result.k_store(T)``, which reads the worst row.
    """
    bw = BandWalk.of(walk)
    labels = bw.labels if labels is None else np.asarray(labels)
    if labels is None:
        raise ValueError("the walk carries no static pool label per walker; pass labels=")
    if len(labels) != bw.n_walkers:
        raise ValueError(f"labels {labels.shape} do not match {bw.n_walkers} walkers")
    names = names or {}
    rows = {"mixture": certify(bw, envelope, eps=eps, **kw)}
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
            rows[nm] = certify(bw.take(m), envelope, eps=eps, **kw)
        except ValueError as e:
            raise ValueError(f"pool {nm!r}: {e}") from None
    return PoolCertificate(rows=rows, n_walkers=int(bw.n_walkers))
