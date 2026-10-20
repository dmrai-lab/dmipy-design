"""Joint gradient + B1 co-design (T4): the pipeline returns a discriminating gradient (replayed through
the packs) and a substrate-robust refocusing pulse that beats a hard pulse over the transmit ensemble,
and the delivered contrast = ideal contrast x refocusing efficiency. Slab packs from dmipy-sim's producer
(the ``slab_packs`` fixture); skipped if dmipy_sim absent."""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")
from dmipy_design.replay_codesign import codesign_waveform_and_b1

from conftest import SLAB_T as T


@pytest.fixture(scope="module")
def packs(slab_packs):
    return slab_packs[4e-6], slab_packs[8e-6]


def test_codesign_pipeline(packs):
    pa, pb = packs
    res = codesign_waveform_and_b1(
        pa, pb, limits=(0.3, 80.0), direction=(1., 0, 0), te=0.6 * T,
        rf_duration=5e-3, B1_max=19e-6,
        grad_kwargs=dict(n_basis=16, n_restarts=2, maxiter=200, seed=0, dt=1e-4),
        rf_kwargs=dict(n_mu=8, n_b1=5, n_off_resonance=5, refine=False))
    # gradient half: a real discriminating, deliverable waveform
    assert res.contrast_ideal > 0.03
    assert res.gradient.te == pytest.approx(0.6 * T)
    # B1 half: the designed refocusing pulse beats a hard pulse over the transmit ensemble
    assert res.refocus_efficiency > res.refocus_efficiency_hard
    assert 0.0 < res.refocus_efficiency <= 1.0 + 1e-9
    # delivered contrast = ideal contrast x refocusing efficiency (the factorization)
    npt.assert_allclose(res.contrast_delivered, res.contrast_ideal * res.refocus_efficiency, atol=1e-9)
    assert res.contrast_delivered <= res.contrast_ideal + 1e-9
