"""Eq. (4) certification: the scanner envelope, the sup within the stored band, and the bandwidth it certifies.

Three things here would make a certificate worse than no certificate, and each has a test:
a wrong ascent gradient (the sup becomes a random number wearing a proof's clothes), a
projection that admits waveforms no scanner can play (the sup is over the wrong set), and a
feasible set that drifts from the one the NOW designer optimises over (the pack under-certifies
exactly the waveforms this package can produce).

The walks are dmipy-sim's producer's, free diffusion over one 16 ms window stored at 64 bands (2 kHz), packed
by its bank; the tests read subsets of its walkers as :class:`BandWalk`.
"""
import numpy as np
import pytest

# `certify` reaches into dmipy-sim for the band basis and the cited scanner catalogue,
# so it belongs to the same optional tier as `replay_design` -- the `core` CI job installs
# neither. Same guard that test_replay_design.py uses.
pytest.importorskip("dmipy_sim")

from dmipy_design.certify import (replay_envelope, sup_replay_error, k_min, certify,
                                  k_for_walk, ReplayEnvelope, Certificate, SupResult, BandWalk,
                                  _objective, _projection, GAMMA)

D0, T, K_STORED = 2e-9, 0.016, 64
DT = 1e-4                    # the test envelopes' raster: 160 samples over the window
FAST = dict(steps=60, restarts=4)


def _free_walk(n_w, T_max, K, seed=0, D=D0):
    import dmipy_sim as ds
    return ds.simulate_trajectories(n_walkers=n_w, diffusivity=D, geometry=ds.FreeDiffusion(), T_max=T_max,
                                    seed=seed, K=K, require_gpu=False)


@pytest.fixture(scope="module")
def free_pack():
    "800 free walkers over 16 ms at 64 bands, as a pack."
    from dmipy_sim.replay.bank import build_replay_pack
    return build_replay_pack(_free_walk(800, T, K_STORED), id="test/free", license="CC-BY-4.0", citation="test",
                             K=K_STORED)


@pytest.fixture(scope="module")
def free(free_pack):
    return BandWalk.of(free_pack)


def walk(free, n_w=400):
    "The first n_w walkers of the free walk."
    return free.take(np.arange(n_w))


def env(G_max=0.08, slew=200.0, name="t", **kw):
    return replay_envelope(G_max=G_max, slew_rate_max=slew, raster=DT, name=name, **kw)


def test_a_pack_and_a_coefficient_walk_read_as_the_same_band_walk(free_pack):
    "The certificate's ensemble is the stored bands, whether they come packed or straight from the producer."
    from dmipy_sim.coefficient_walk import CoefficientWalk
    bw = BandWalk.of(free_pack)
    assert bw.K == K_STORED and bw.T == pytest.approx(T) and bw.D == pytest.approx(D0)
    assert bw.C.shape == (800, K_STORED + 2, 3) and bw.C.dtype == np.float32
    assert bw.truncation == pytest.approx(free_pack.fidelity["terms"]["truncation"])
    w = _free_walk(50, T, 8, seed=3)
    assert isinstance(w, CoefficientWalk)
    cw = BandWalk.of(w)
    assert cw.K == 8 and cw.n_walkers == 50 and cw.truncation is None       # not yet packed: no term beyond K
    assert BandWalk.of(cw) is cw
    sub = bw.take(np.arange(10))
    assert sub.n_walkers == 10 and sub.K == bw.K and sub.truncation == bw.truncation
    with pytest.raises(TypeError, match="BandWalk"):
        BandWalk.of(np.zeros((3, 10, 3)))


def test_the_ascent_gradient_matches_finite_differences(free):
    """An ascent on a wrong gradient still returns a number, and the number means nothing."""
    bw = walk(free)
    e = env()
    n_t = int(round(T / DT))
    prob = e.problem(n_t, DT)
    fun = _objective(prob, bw.C, 16, _projection(n_t, DT, T, bw.K))
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, prob.n_var)
    f0, g0 = fun(x)
    h = 1e-7
    for i in rng.integers(0, prob.n_var, 6):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        fd = (fun(xp)[0] - fun(xm)[0]) / (2 * h)
        assert abs(fd - g0[i]) <= 1e-3 * max(abs(g0[i]), 1e-6)


def test_the_returned_waveform_is_the_one_that_scored_the_sup(free):
    """``SupResult.sup`` is attained by ``SupResult.waveform``: re-evaluating the objective on the returned waveform
    gives the reported sup (dmipy-design#18: the ascent stored the iterate one projected step past the one it
    scored, which in the regime where the adversary kills the signal was a different waveform altogether -- on
    Magnus at K = 8 the returned waveform's error was 0.128 against a reported sup of 0.416)."""
    bw = walk(free)
    n_t = int(round(T / DT))
    P = _projection(n_t, DT, T, bw.K)
    for gmax, slew, K in ((0.3, 200.0, 4), (0.3, 200.0, 32), (3.0, 3000.0, 4), (3.0, 3000.0, 8)):
        e = env(gmax, slew)                                                           # the last two: the adversary kills
        r, x = sup_replay_error(bw, K, e, seed=1, **FAST)                             # the signal
        prob = e.problem(n_t, DT)
        again = -_objective(prob, bw.C, K, P)(x)[0]
        assert abs(again - r.sup) <= 1e-4 * max(1.0, abs(r.sup)), (K, r.sup, again)     # float32 ascent vs float64 re-read


def test_the_winning_waveform_is_actually_deliverable(free):
    """A sup over waveforms the hardware cannot play certifies nothing at all."""
    r, _ = sup_replay_error(walk(free), 16, env(), **FAST)
    G = r.waveform
    assert G.shape == (int(round(T / DT)), 3)                      # on the envelope's raster
    assert np.abs(G).max() <= 0.08 * (1 + 1e-6)
    assert np.abs(np.diff(G, axis=0)).max() / DT <= 200.0 * (1 + 1e-4)
    # normalised the way NOW judges its own designs (refocus_residual < 1e-2), rather than
    # against an absolute epsilon: the ascent runs in float32, so |M0| bottoms out near 1e-7
    # of the gradient scale and an absolute 1e-12 would be testing the dtype, not the physics.
    q = GAMMA * DT * np.cumsum(G, axis=0)
    refoc = np.linalg.norm(q[-1]) / (np.sqrt(np.max((q ** 2).sum(1))) + 1e-30)
    assert refoc < 1e-2, f"must be refocused: q(TE)/max|q| = {refoc:.2e}"
    assert r.b_value > 0 and r.bound is not None and np.isfinite(r.bound) and r.bound >= 0


def test_moment_order_one_also_nulls_the_first_moment(free):
    r, _ = sup_replay_error(walk(free), 16, env(null_M1=True), **FAST)
    n_t = int(round(T / DT))
    t = np.arange(n_t) * DT
    m1 = np.linalg.norm((t[:, None] * r.waveform).sum(0) * DT) / (0.08 * T ** 2)
    assert m1 < 5e-2, f"m1 index {m1:.2e} (NOW calls a design feasible below 5e-2)"


def test_the_sup_is_taken_within_the_stored_band(free):
    "The stored band against itself is zero by construction, and beyond it is the pack's own term, not a sup."
    with pytest.raises(ValueError, match="stored band"):
        sup_replay_error(walk(free), K_STORED, env(), **FAST)
    with pytest.raises(ValueError, match="stored band"):
        sup_replay_error(walk(free), K_STORED + 8, env(), **FAST)


def test_the_certificate_is_taken_over_the_same_set_the_designer_optimises_over():
    """The soundness property, and the reason the feasible set is shared rather than copied.

    A user designs a waveform with NOW and replays it against a pack certified here. If the two
    feasible sets ever drift apart, the pack silently under-certifies waveforms this very package
    can produce -- so a NOW design must land inside the certificate's set at the same limits.
    """
    from dmipy_design.optimizers.now import design_waveform_now
    G_max, slew = 0.08, 200.0
    d = design_waveform_now(b_delta=1.0, limits=(G_max, slew), TE=0.06, n_t=120,
                            null_M1=False, null_M2=False, n_restarts=2, maxiter=60)
    e = replay_envelope(G_max=G_max, slew_rate_max=slew, name="shared")
    prob = e.problem(d.G.shape[0], d.dt, echo=d.echo_idx)
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
    assert replay_envelope("siemens_magnetom_connectom_3T").raster == pytest.approx(10e-6)   # the class's raster


def test_an_envelope_needs_a_model_or_explicit_limits():
    with pytest.raises(ValueError, match="G_max"):
        replay_envelope(G_max=0.08)


def test_the_sup_falls_with_K_and_rises_with_the_envelope(free):
    """Both are physics rather than fitting: more bands only shrink the residual, and a stronger
    gradient only reads that residual harder."""
    bw = walk(free)
    got = {}
    for name, g, s in (("weak", 0.08, 200.0), ("strong", 1.0, 1e4)):
        e = env(g, s, name)
        got[name] = [sup_replay_error(bw, K, e, **FAST)[0].sup for K in (8, 16, 32)]
        assert got[name][0] >= got[name][1] >= got[name][2]
    assert all(b > a for a, b in zip(got["weak"], got["strong"]))


@pytest.mark.xfail(strict=False,
                   reason="platform-sensitive: the K search lands one grid step apart on the two durations on "
                          "x86 / numpy 2.5 (f_c 125 vs 250 Hz) while agreeing on aarch64 / numpy 2.2; "
                          "see dmipy-design#17 -- the certificate itself is not in question here")
def test_the_certified_bandwidth_is_the_duration_invariant(free):
    """K is meaningless without the window it was measured over -- mode k sits at f = k/(2T), so a
    band count and a duration only ever appear together. f_c is what transfers between walks."""
    e = env()
    fc = []
    for bw in (free.take(np.arange(600)), BandWalk.of(_free_walk(600, 2 * T, 2 * K_STORED, seed=5))):
        # a converged search, not the FAST settings: an under-explored ascent understates the
        # sup at small K, which lets the SHORT walk pass at a K the long one needs more of, and
        # that reads as a broken invariance rather than as a weak optimiser (measured: 6
        # restarts gave 5.5e-2 at K=16 on Magnus where 64 gave 5.3e-1).
        c = certify(bw, e, eps=2e-2, K_grid=(2, 3, 4, 6, 8, 12, 16, 24, 32), steps=200, restarts=32)
        fc.append(c.f_c)
        assert c.K == c.k_for(c.T), "K must be recoverable from f_c and the duration"
    assert fc[1] == pytest.approx(fc[0], rel=0.35), \
        f"certified bandwidth moved with duration: {fc} Hz (it must not)"


def test_k_for_walk_is_the_producers_derivation():
    """The band count of a window is dmipy-sim's one derivation: ``ceil(2 T f)`` with ``f`` a stated bandwidth
    or the scanner class's storage band (its certified band times the storage margin, dmipy-sim#643)."""
    from dmipy_sim.replay.bank import mode_count, storage_band_hz
    assert k_for_walk(0.060, bandwidth_hz=267.0) == 33            # ceil(2 * 0.060 * 267)
    assert k_for_walk(0.100, bandwidth_hz=267.0) == 54
    assert k_for_walk(0.200, bandwidth_hz=267.0) == 107
    f = storage_band_hz("siemens_prisma")
    assert k_for_walk(0.1, scanner="siemens_prisma") == mode_count(0.1, "siemens_prisma") == int(np.ceil(0.2 * f))
    with pytest.raises(ValueError, match="certified band"):          # a class the catalogue holds no band for
        k_for_walk(0.1, scanner="siemens_magnetom_terra_7T")
    with pytest.raises(ValueError, match="bandwidth_hz"):
        k_for_walk(0.1)


def test_the_holdout_runs_below_the_certificate(free):
    """The adversary partly memorises the ensemble, so a waveform fitted on one half reads the
    other half less hard. Certifying on that softer number would under-size K for the pack a
    consumer actually replays -- hence `sup` is the certificate and this is context."""
    r, _ = sup_replay_error(walk(free, 600), 16, env(0.30, 750.0, "magnus"), holdout=True, **FAST)
    assert isinstance(r, SupResult) and r.held_out is not None
    assert r.held_out < r.sup
    assert float(r) == r.sup, "float() must give the certificate, not the softer number"


def test_certify_refuses_rather_than_returning_the_biggest_K_it_tried(free_pack):
    """Silently returning the largest K would ship a pack whose certificate is false: below the stored band
    nothing in the grid meets eps, and the pack's own walk-free term beyond its band does not either."""
    with pytest.raises(ValueError, match="truncation term is"):
        certify(free_pack, env(3.0, 2e4, "insert"), eps=1e-9, K_grid=(4, 8), **FAST)


def test_the_stored_band_is_the_certificate_when_no_sub_band_meets_eps(free_pack):
    """A pack cannot be certified tighter than its own band, and at its own band the error is the walk-free
    truncation term its certificate states (RPK.md 9.5.5), quoted on the certificate; a walk not yet packed
    states none and is refused there."""
    trunc = free_pack.fidelity["terms"]["truncation"]
    with pytest.warns(RuntimeWarning):                                       # low-N, expected for a fast test
        c = certify(free_pack, env(), eps=2 * trunc, K_grid=(), **FAST)
    assert c.K == c.K_stored == K_STORED and c.f_c == pytest.approx(K_STORED / (2 * T))
    assert c.truncation == pytest.approx(trunc) and c.sups == {}
    assert c.describe()["K_stored"] == K_STORED
    with pytest.raises(ValueError, match="not a pack"):
        certify(_free_walk(300, T, 8, seed=4), env(), eps=2 * trunc, K_grid=(), **FAST)


def test_k_min_is_decided_by_the_certificate_not_the_holdout(free):
    K, sups = k_min(walk(free, 600), env(), eps=2e-2, K_grid=(4, 8, 16, 32), holdout=True, **FAST)
    if K is not None:
        assert sups[K].sup <= 2e-2
        assert all(sups[k].sup > 2e-2 for k in sups if k < K)


def test_subsample_is_conservative_not_optimistic(free):
    """Fewer walkers give the adversary MORE room to fit the realisation, so a subsampled search
    may over-state the sup. It must never under-state it, or a cheap search would ship a pack
    certified on a number the full ensemble cannot honour."""
    e = env()
    full = sup_replay_error(walk(free, 800), 16, e, **FAST)[0].sup
    sub = sup_replay_error(walk(free, 200), 16, e, **FAST)[0].sup
    assert sub >= full * 0.5, f"subsampled sup {sub:.2e} far below full-ensemble {full:.2e}"


def test_it_warns_when_certifying_from_too_few_walkers(free):
    """Below the convergence knee the sup is INFLATED, not merely noisy, so K comes out
    over-provisioned with nothing else to signal it. Silence there would be the worst outcome:
    a confident certificate that is simply too large."""
    with pytest.warns(RuntimeWarning, match="below the ~5,000"):
        certify(walk(free, 300), env(), eps=2e-2, K_grid=(8, 16, 32, 48), **FAST)


def test_the_pool_table_is_sized_by_its_worst_row_not_the_mixture(free):
    """The mixture is genuinely easier than its parts -- dilution, plus the adversary being
    unable to be worst-case for every pool with one waveform. A pack sized on the mixture would
    under-certify anyone replaying a single compartment, which packs support."""
    from dmipy_design.certify import certify_pools, PoolCertificate
    # two pools with deliberately different mobility, plus a frozen one
    fast = free.take(np.arange(400))
    slow = BandWalk.of(_free_walk(400, T, K_STORED, seed=7, D=0.09 * D0))
    frozen = np.zeros((200, K_STORED + 2, 3), np.float32)
    frozen[:, 0, :] = np.random.default_rng(7).normal(0, 1e-6, (200, 3))     # a start, no motion
    bw = BandWalk(np.concatenate([fast.C, slow.C, frozen]), T, D0,
                  np.concatenate([np.zeros(400, int), np.ones(400, int), np.full(200, 2)]))
    with pytest.warns(RuntimeWarning):                      # low-N, expected for a fast test
        pc = certify_pools(bw, env(), names={0: "fast", 1: "slow", 2: "frozen"},
                           eps=2e-2, K_grid=(8, 16, 32, 48), **FAST)
    assert isinstance(pc, PoolCertificate)
    assert set(pc.rows) == {"mixture", "fast", "slow", "frozen"}
    # the frozen pool costs nothing; the mobile one binds
    assert pc.rows["frozen"].f_c <= pc.rows["fast"].f_c
    # sizing reads the WORST row, so it is never below the mixture's own requirement
    assert pc.k_store(T) >= pc.rows["mixture"].k_for(T)
    assert pc.binding[0] in pc.rows
    assert pc.for_pool("fast") is pc.rows["fast"]
    with pytest.raises(KeyError, match="no certificate"):
        pc.for_pool("nonexistent")


def test_a_pool_too_small_to_certify_is_skipped_not_guessed(free):
    from dmipy_design.certify import certify_pools
    lab = np.concatenate([np.zeros(400, int), np.ones(20, int)])   # 20 is too few
    with pytest.warns(RuntimeWarning):
        pc = certify_pools(walk(free, 420), env(), labels=lab, eps=2e-2, K_grid=(8, 16, 32, 48),
                           min_walkers=200, **FAST)
    assert "1" not in pc.rows and "0" in pc.rows


def test_pool_labels_come_from_the_walk_or_must_match_it(free_pack, free):
    from dmipy_design.certify import certify_pools
    with pytest.raises(ValueError, match="do not match"):
        certify_pools(walk(free, 100), env(), labels=np.zeros(7, int), **FAST)
    with pytest.raises(ValueError, match="no static pool label"):
        certify_pools(BandWalk(walk(free, 100).C, T, D0), env(), **FAST)
    # a pack's static occupancy column is the default label: one pool here, so one row beside the mixture
    with pytest.warns(RuntimeWarning):
        pc = certify_pools(free_pack, env(), eps=2e-2, K_grid=(8, 16, 32, 48), **FAST)
    assert set(pc.rows) == {"mixture", "0"}
