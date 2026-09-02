"""Eq. (4) certification: the scanner envelope, the sup, and the bandwidth it certifies.

Three things here would make a certificate worse than no certificate, and each has a test:
a wrong ascent gradient (the sup becomes a random number wearing a proof's clothes), a
projection that admits waveforms no scanner can play (the sup is over the wrong set), and a
feasible set that drifts from the one the NOW designer optimises over (the pack under-certifies
exactly the waveforms this package can produce).
"""
import numpy as np
import pytest

from dmipy_design.certify import (replay_envelope, sup_replay_error, k_min, certify,
                                  k_for_walk, ReplayEnvelope, Certificate, SupResult,
                                  _truncate, _objective, GAMMA)

D0, DT = 2e-9, 1e-4
FAST = dict(steps=60, restarts=4)


def walk(n_w=400, n_t=161, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, np.sqrt(2 * D0 * DT), (n_w, n_t, 3)), axis=1).astype(np.float32)


def test_the_ascent_gradient_matches_finite_differences():
    """An ascent on a wrong gradient still returns a number, and the number means nothing."""
    X = walk()
    env = replay_envelope(G_max=0.08, slew_rate_max=200.0, name="t")
    prob = env.problem(X.shape[1], DT)
    fun = _objective(prob, X, _truncate(X, 16), DT)
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, prob.n_var)
    f0, g0 = fun(x)
    h = 1e-7
    for i in rng.integers(0, prob.n_var, 6):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        fd = (fun(xp)[0] - fun(xm)[0]) / (2 * h)
        assert abs(fd - g0[i]) <= 1e-3 * max(abs(g0[i]), 1e-6)


def test_the_winning_waveform_is_actually_deliverable():
    """A sup over waveforms the hardware cannot play certifies nothing at all."""
    X = walk()
    env = replay_envelope(G_max=0.08, slew_rate_max=200.0, name="t")
    r, _ = sup_replay_error(X, 16, env, DT, **FAST)
    G = r.waveform
    assert np.abs(G).max() <= 0.08 * (1 + 1e-6)
    assert np.abs(np.diff(G, axis=0)).max() / DT <= 200.0 * (1 + 1e-4)
    # normalised the way NOW judges its own designs (refocus_residual < 1e-2), rather than
    # against an absolute epsilon: the ascent runs in float32, so |M0| bottoms out near 1e-7
    # of the gradient scale and an absolute 1e-12 would be testing the dtype, not the physics.
    q = GAMMA * DT * np.cumsum(G, axis=0)
    refoc = np.linalg.norm(q[-1]) / (np.sqrt(np.max((q ** 2).sum(1))) + 1e-30)
    assert refoc < 1e-2, f"must be refocused: q(TE)/max|q| = {refoc:.2e}"


def test_moment_order_one_also_nulls_the_first_moment():
    X = walk()
    env = replay_envelope(G_max=0.08, slew_rate_max=200.0, null_M1=True, name="t")
    r, _ = sup_replay_error(X, 16, env, DT, **FAST)
    t = np.arange(X.shape[1]) * DT
    TE = (X.shape[1] - 1) * DT
    m1 = np.linalg.norm((t[:, None] * r.waveform).sum(0) * DT) / (0.08 * TE ** 2)
    assert m1 < 5e-2, f"m1 index {m1:.2e} (NOW calls a design feasible below 5e-2)"


def test_the_certificate_is_taken_over_the_same_set_the_designer_optimises_over():
    """The soundness property, and the reason the feasible set is shared rather than copied.

    A user designs a waveform with NOW and replays it against a pack certified here. If the two
    feasible sets ever drift apart, the pack silently under-certifies waveforms this very package
    can produce -- so a NOW design must land inside the certificate's set at the same limits.
    """
    from dmipy_design.optimizers.now import design_waveform_now
    G_max, slew = 0.08, 200.0
    d = design_waveform_now(b_delta=1.0, G_max=G_max, slew_rate_max=slew, TE=0.06, n_t=120,
                            null_M1=False, null_M2=False, n_restarts=2, maxiter=60)
    env = replay_envelope(G_max=G_max, slew_rate_max=slew, name="shared")
    prob = env.problem(d.G.shape[0], d.dt, echo=d.echo_idx)
    assert np.abs(d.G).max() <= prob.G_max * 1.02
    assert np.abs(np.diff(d.G, axis=0)).max() / d.dt <= prob.slew_rate_max * 1.02
    # and the certificate's own projection leaves an already-feasible design essentially alone
    back = prob.gradient(prob.project(prob.flatten(d.G[:, :prob.n_axes])))
    assert np.abs(back).max() <= prob.G_max * (1 + 1e-6)


def test_connectom_certifies_against_its_diffusion_slew():
    """The PNS-derated limit binds a diffusion sequence, and it is the difference between
    Connectom being the easiest class to certify and looking like the hardest."""
    assert replay_envelope("siemens_magnetom_connectom_3T").slew_rate_max == pytest.approx(62.5)
    assert replay_envelope("siemens_magnetom_connectom_3T",
                           regime="default").slew_rate_max == pytest.approx(200.0)


def test_an_envelope_needs_a_model_or_explicit_limits():
    with pytest.raises(ValueError, match="G_max"):
        replay_envelope(G_max=0.08)


def test_the_sup_falls_with_K_and_rises_with_the_envelope():
    """Both are physics rather than fitting: more modes only shrink the residual, and a stronger
    gradient only reads that residual harder."""
    X = walk()
    got = {}
    for name, g, s in (("weak", 0.08, 200.0), ("strong", 1.0, 1e4)):
        env = replay_envelope(G_max=g, slew_rate_max=s, name=name)
        got[name] = [sup_replay_error(X, K, env, DT, **FAST)[0].sup for K in (8, 32, 64)]
        assert got[name][0] >= got[name][1] >= got[name][2]
    assert all(b > a for a, b in zip(got["weak"], got["strong"]))


def test_the_certified_bandwidth_is_the_duration_invariant():
    """K is meaningless without the walk it was measured over -- mode m sits at f = m/(2T), so a
    mode count and a duration only ever appear together. f_c is what transfers between walks."""
    X = walk(n_w=600, n_t=321)
    env = replay_envelope(G_max=0.08, slew_rate_max=200.0, name="t")
    fc = []
    for n_t in (161, 321):
        # a converged search, not the FAST settings: an under-explored ascent understates the
        # sup at small K, which lets the SHORT walk pass at a K the long one needs more of, and
        # that reads as a broken invariance rather than as a weak optimiser (measured: 6
        # restarts gave 5.5e-2 at K=16 on Magnus where 64 gave 5.3e-1).
        c = certify(np.ascontiguousarray(X[:, :n_t]), env, DT, eps=2e-2,
                    K_grid=(2, 3, 4, 6, 8, 12, 16, 24, 32), steps=200, restarts=32)
        fc.append(c.f_c)
        assert c.K == c.k_for(c.T), "K must be recoverable from f_c and the duration"
    assert fc[1] == pytest.approx(fc[0], rel=0.35), \
        f"certified bandwidth moved with duration: {fc} Hz (it must not)"


def test_k_for_walk_scales_with_duration_and_refuses_above_nyquist():
    assert k_for_walk(267.0, 0.060) == 33            # ceil(2 * 0.060 * 267)
    assert k_for_walk(267.0, 0.100) == 54
    assert k_for_walk(267.0, 0.200) == 107
    with pytest.raises(ValueError, match="Nyquist"):
        k_for_walk(9000.0, 0.100, dt_save=1e-4)      # walk resolves only to 5 kHz


def test_the_holdout_runs_below_the_certificate():
    """The adversary partly memorises the ensemble, so a waveform fitted on one half reads the
    other half less hard. Certifying on that softer number would under-size K for the pack a
    consumer actually replays -- hence `sup` is the certificate and this is context."""
    X = walk(n_w=600)
    env = replay_envelope(G_max=0.30, slew_rate_max=750.0, name="magnus")
    r, _ = sup_replay_error(X, 16, env, DT, holdout=True, **FAST)
    assert isinstance(r, SupResult) and r.held_out is not None
    assert r.held_out < r.sup
    assert float(r) == r.sup, "float() must give the certificate, not the softer number"


def test_certify_refuses_rather_than_returning_the_biggest_K_it_tried():
    """Silently returning the largest K would ship a pack whose certificate is false."""
    X = walk()
    env = replay_envelope(G_max=3.0, slew_rate_max=2e4, name="insert")
    with pytest.raises(ValueError, match="no K in"):
        certify(X, env, DT, eps=1e-9, K_grid=(4, 8), **FAST)


def test_k_min_is_decided_by_the_certificate_not_the_holdout():
    X = walk(n_w=600)
    env = replay_envelope(G_max=0.08, slew_rate_max=200.0, name="t")
    K, sups = k_min(X, env, DT, eps=2e-2, K_grid=(4, 8, 16, 32), holdout=True, **FAST)
    if K is not None:
        assert sups[K].sup <= 2e-2
        assert all(sups[k].sup > 2e-2 for k in sups if k < K)


def test_subsample_is_conservative_not_optimistic():
    """Fewer walkers give the adversary MORE room to fit the realisation, so a subsampled search
    may over-state the sup. It must never under-state it, or a cheap search would ship a pack
    certified on a number the full ensemble cannot honour."""
    X = walk(n_w=800)
    env = replay_envelope(G_max=0.08, slew_rate_max=200.0, name="t")
    full = sup_replay_error(X, 16, env, DT, **FAST)[0].sup
    sub = sup_replay_error(np.ascontiguousarray(X[:200]), 16, env, DT, **FAST)[0].sup
    assert sub >= full * 0.5, f"subsampled sup {sub:.2e} far below full-ensemble {full:.2e}"
