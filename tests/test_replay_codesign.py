"""Joint gradient + B1 co-design (T4): the pipeline returns a discriminating gradient (replayed through
the packs) and a substrate-robust refocusing pulse that beats a hard pulse over the transmit ensemble,
and the delivered contrast = ideal contrast x refocusing efficiency. Synthetic slab packs; skipped if
dmipy_sim absent."""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")
from scipy.fft import dct
from dmipy_sim.replay import ReplayPack
from dmipy_design.replay_codesign import codesign_waveform_and_b1

N_W, N_T, K, DT, D0 = 500, 120, 40, 1e-3, 2e-9


def _slab_pack(L, seed):
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * D0 * DT); x = rng.uniform(0, L, N_W)
    traj = np.zeros((N_W, N_T, 3))
    for t in range(N_T):
        x = x + rng.normal(0, step, N_W); x = np.mod(x, 2 * L); x = np.where(x > L, 2 * L - x, x)
        traj[:, t, 0] = x
    traj[:, :, 1:] = np.cumsum(rng.normal(0, step, (N_W, N_T, 2)), axis=1)
    traj -= traj.mean(1, keepdims=True)
    C = dct(traj, type=2, norm="ortho", axis=1)[:, :K, :]
    return ReplayPack({"dct_coeffs": C.astype(np.float32), "spin_weights": np.ones(N_W, np.float32)},
                      {"n_t": N_T, "dt": DT, "walk_params": {"n_t": N_T, "dt_traj": DT}})


@pytest.fixture(scope="module")
def packs():
    return _slab_pack(4e-6, 0), _slab_pack(8e-6, 1)


def test_codesign_pipeline(packs):
    pa, pb = packs
    res = codesign_waveform_and_b1(
        pa, pb, direction=(1., 0, 0), G_max=0.3, te=0.6 * (N_T - 1) * DT, slew_max=80.0,
        rf_duration=5e-3, B1_max=19e-6,
        grad_kwargs=dict(n_basis=16, n_restarts=2, maxiter=200, seed=0),
        rf_kwargs=dict(n_mu=8, n_b1=5, n_off_resonance=5, refine=False))
    # gradient half: a real discriminating, deliverable waveform
    assert res.contrast_ideal > 0.03
    assert res.gradient.te == pytest.approx(0.6 * (N_T - 1) * DT)
    # B1 half: the designed refocusing pulse beats a hard pulse over the transmit ensemble
    assert res.refocus_efficiency > res.refocus_efficiency_hard
    assert 0.0 < res.refocus_efficiency <= 1.0 + 1e-9
    # delivered contrast = ideal contrast x refocusing efficiency (the factorization)
    npt.assert_allclose(res.contrast_delivered, res.contrast_ideal * res.refocus_efficiency, atol=1e-9)
    assert res.contrast_delivered <= res.contrast_ideal + 1e-9
