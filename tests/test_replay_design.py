"""Waveform optimization through replay packs — the shape/size discrimination slice.

Two slab packs of different thickness (dmipy-sim's producer, packed by its bank; the ``slab_packs`` fixture),
which is exactly the *size discrimination* case. Checks that the optimizer finds a deliverable waveform whose
signals separate the two substrates, beating a naive PGSE baseline, and that the differentiable replay forward
is the engine's own. Skipped if dmipy_sim is absent.
"""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")

from dmipy_sim.replay import compile_scheme, replay_signal
from dmipy_sim.constants import GAMMA
from dmipy_design.replay_design import design_discriminating_waveform, DESIGN_RASTER

from conftest import SLAB_T as T, SLAB_K as K

DT = 1e-4          # a coarse design raster, for the tests that do not exercise the default


@pytest.fixture(scope="module")
def packs(slab_packs):
    return slab_packs[4e-6], slab_packs[8e-6]      # 4 um vs 8 um slabs


def _pgse_contrast(pack_a, pack_b, b, delta=20e-3, Delta=None):
    "Contrast |E_a - E_b| of a plain square PGSE at b on the raster DT (baseline to beat)."
    Delta = Delta or 0.7 * T
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    n_t = int(round(T / DT))
    g = np.zeros((1, n_t, 3)); nd = max(1, int(round(delta / DT))); ng = int(round(Delta / DT))
    g[0, :nd, 0] = np.sqrt(b / bu); g[0, ng:ng + nd, 0] = -np.sqrt(b / bu)
    ea = replay_signal(pack_a, compile_scheme(g, DT, pack_a.K, GAMMA, T=T))[0]
    eb = replay_signal(pack_b, compile_scheme(g, DT, pack_b.K, GAMMA, T=T))[0]
    return abs(ea - eb)


def test_discrimination_optimizer_separates_slabs(packs):
    pa, pb = packs
    res = design_discriminating_waveform(pa, pb, limits=(0.3, np.inf), direction=(1., 0, 0), dt=DT,
                                         maxiter=250, n_restarts=4, seed=0)
    # a real separation, and the two substrates genuinely differ under the optimized waveform
    assert res.contrast > 0.05
    assert abs(res.E_A - res.E_B) == pytest.approx(res.contrast, abs=1e-9)
    # beats the best of a naive PGSE b-sweep (the point of optimizing the waveform)
    baseline = max(_pgse_contrast(pa, pb, b) for b in (1e9, 2e9, 4e9, 8e9))
    assert res.contrast >= baseline - 1e-3
    # deliverable: amplitude bounded and (near-)refocused, on the raster asked for
    assert res.dt == DT and res.G.shape == (int(round(T / DT)), 3)
    assert np.abs(res.G).max() <= 0.3 + 1e-6
    assert abs(np.sum(res.G[:, 0]) * DT) < 0.3 * T * 0.02             # |q(TE)| small


def test_forward_matches_engine(packs):
    "The signals the optimizer differentiates through equal the sim replay engine's forward, on the default raster."
    pa, pb = packs
    res = design_discriminating_waveform(pa, pb, limits=(0.3, np.inf), maxiter=40, n_restarts=1, seed=1)
    assert res.dt == DESIGN_RASTER                                    # the limits state no raster
    W = compile_scheme(res.G[None], res.dt, pa.K, GAMMA, T=T)
    npt.assert_allclose(replay_signal(pa, W)[0], res.E_A, atol=1e-6)
    W = compile_scheme(res.G[None], res.dt, pb.K, GAMMA, T=T)
    npt.assert_allclose(replay_signal(pb, W)[0], res.E_B, atol=1e-6)


def test_the_design_basis_is_the_packs_band(packs):
    "A design cannot use more sines than the packs store, and the packs must share the window."
    pa, pb = packs
    with pytest.raises(ValueError, match="band"):
        design_discriminating_waveform(pa, pb, limits=(0.3, np.inf), n_basis=K + 1, maxiter=5, n_restarts=1)


def test_deliverable_te_window_and_slew(packs):
    "T2: a TE-encoding window bounds b and zeros the waveform after TE; a slew cap is respected."
    pa, pb = packs
    te = 0.6 * T
    res = design_discriminating_waveform(pa, pb, limits=(0.3, 50.0), direction=(1., 0, 0), te=te, dt=DT,
                                         maxiter=250, n_restarts=3, seed=0)
    te_idx = int(round(te / DT))
    assert np.allclose(res.G[te_idx:], 0.0)                  # no encoding after TE
    assert res.te == pytest.approx(te)
    assert res.contrast > 0.03                               # still discriminates within the window
    # bounded b (a full-window design reaches far higher) and a respected slew cap (soft)
    assert res.max_slew <= 50.0 * 1.3


def test_the_result_is_a_sequence(packs):
    "The designed waveform is dmipy-sim's object: a self-refocusing gradient echo the pack replays directly."
    pa, pb = packs
    res = design_discriminating_waveform(pa, pb, limits=(0.3, np.inf), dt=DT, maxiter=40, n_restarts=1, seed=1)
    seq = res.to_sequence()
    assert seq.n_meas == 1 and not seq.rf and seq.refocusing_residual < 1e-2
    npt.assert_allclose(seq.b(), [res.b_value], rtol=1e-5)
    npt.assert_allclose(pa.replay(seq, tissue=None)[0], res.E_A, atol=2e-3)
