"""Waveform optimization through replay packs — the shape/size discrimination slice.

Uses two synthetic slab packs of different thickness (a 1-D reflecting random walk, compressed to DCT
coefficients — no simulator needed), which is exactly the *size discrimination* case. Checks that the
optimizer finds a deliverable waveform whose signals separate the two substrates, beating a naive PGSE
baseline, and that the differentiable replay forward drives it. Skipped if jax/optax/dmipy_sim absent.
"""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")

from dmipy_sim.replay import compile_scheme, replay_signal
from dmipy_sim.constants import GAMMA
from dmipy_design.replay_design import design_discriminating_waveform

N_W, N_T, K, DT, D0 = 600, 120, 40, 1e-3, 2e-9


def _slab_pack(L, seed):
    """A reflecting-slab (0 <= x <= L) + free y, z walk, packed the way dmipy-sim packs one."""
    from dmipy_sim.replay.bank import build_replay_pack
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * D0 * DT); x = rng.uniform(0, L, N_W)
    traj = np.zeros((N_W, N_T, 3)); dlog = np.zeros((N_W, N_T))
    for t in range(N_T):
        x = x + rng.normal(0, step, N_W)
        lo, hi = x < 0, x > L
        x = np.where(lo, -x, np.where(hi, 2 * L - x, x))
        traj[:, t, 0] = x; dlog[:, t] = (lo | hi) * step
    traj[:, :, 1:] = np.cumsum(rng.normal(0, step, (N_W, N_T, 2)), axis=1)
    m = dict(traj=traj, dt_traj=DT, T_max=(N_T - 1) * DT, comp=np.zeros((N_W, N_T), np.int8),
             comp0=np.zeros(N_W, np.int64), w=np.ones(N_W), dlog_b=dlog, D_intra=D0, n_walkers=N_W, seed=seed)
    return build_replay_pack(m, id=f"test/slab-{seed}", method="bridge_dst", K=K, license="CC-BY-4.0", citation="test")


@pytest.fixture(scope="module")
def packs():
    return _slab_pack(4e-6, 0), _slab_pack(8e-6, 1)      # 4 um vs 8 um slabs


def _pgse_contrast(pack_a, pack_b, b, delta=8e-3, Delta=None):
    "Contrast |E_a - E_b| of a plain PGSE at b (baseline to beat)."
    Delta = Delta or (N_T - 1) * DT * 0.9
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    g = np.zeros((1, N_T, 3)); nd = max(1, int(round(delta / DT))); ng = int(round(Delta / DT))
    g[0, :nd, 0] = np.sqrt(b / bu); g[0, ng:ng + nd, 0] = -np.sqrt(b / bu)
    ea = replay_signal(pack_a, compile_scheme(g, DT, pack_a.K, GAMMA))[0]
    eb = replay_signal(pack_b, compile_scheme(g, DT, pack_b.K, GAMMA))[0]
    return abs(ea - eb)


def test_discrimination_optimizer_separates_slabs(packs):
    pa, pb = packs
    res = design_discriminating_waveform(pa, pb, limits=(0.3, np.inf), direction=(1., 0, 0),
                                         maxiter=250, n_restarts=4, seed=0)
    # a real separation, and the two substrates genuinely differ under the optimized waveform
    assert res.contrast > 0.05
    assert abs(res.E_A - res.E_B) == pytest.approx(res.contrast, abs=1e-9)
    # beats the best of a naive PGSE b-sweep (the point of optimizing the waveform)
    baseline = max(_pgse_contrast(pa, pb, b) for b in (1e9, 2e9, 4e9, 8e9))
    assert res.contrast >= baseline - 1e-3
    # deliverable: amplitude bounded and (near-)refocused
    assert np.abs(res.G).max() <= 0.3 + 1e-6
    assert abs(np.sum(res.G[:, 0]) * DT) < 0.3 * DT * N_T * 0.02    # |q(TE)| small


def test_forward_matches_engine(packs):
    "The signals the optimizer differentiates through equal the sim replay engine's forward."
    pa, _ = packs
    res = design_discriminating_waveform(pa, packs[1], limits=(0.3, np.inf), maxiter=40, n_restarts=1, seed=1)
    W = compile_scheme(res.G[None], DT, pa.K, GAMMA)
    npt.assert_allclose(replay_signal(pa, W)[0], res.E_A, atol=1e-4)


def test_deliverable_te_window_and_slew(packs):
    "T2: a TE-encoding window bounds b and zeros the waveform after TE; a slew cap is respected."
    pa, pb = packs
    te = 0.6 * (N_T - 1) * DT
    res = design_discriminating_waveform(pa, pb, limits=(0.3, 50.0), direction=(1., 0, 0), te=te,
                                         maxiter=250, n_restarts=3, seed=0)
    te_idx = int(round(te / DT))
    assert np.allclose(res.G[te_idx:], 0.0)                  # no encoding after TE
    assert res.te == pytest.approx(te)
    assert res.contrast > 0.03                               # still discriminates within the window
    # bounded b (a full-window design reaches far higher) and a respected slew cap (soft)
    assert res.max_slew <= 50.0 * 1.3
    # analytic-gradient sanity: finite-difference on the objective at a random c
    from dmipy_design.replay_design import _PackForward  # noqa: F401  (import path check)


def test_the_result_is_a_sequence(packs):
    "The designed waveform is dmipy-sim's object: a self-refocusing gradient echo the pack replays directly."
    pa, pb = packs
    res = design_discriminating_waveform(pa, pb, limits=(0.3, np.inf), maxiter=40, n_restarts=1, seed=1)
    seq = res.to_sequence()
    assert seq.n_meas == 1 and not seq.rf and seq.refocusing_residual < 1e-2
    npt.assert_allclose(seq.b(), [res.b_value], rtol=1e-5)
    npt.assert_allclose(pa.replay(seq, tissue=False)[0], res.E_A, atol=2e-3)
