"""Cross-package interoperability: a dmipy-design waveform is eaten correctly by dmipy-sim
and fitted correctly by dmipy-fit — the three engines speak the same language.

Requires the ``[interop]`` extra (dmipy-sim + dmipy-fit); skipped otherwise.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("dmipy_sim")
pytest.importorskip("dmipy_fit")

import dmipy_sim as ds
from dmipy_design import design_waveform_now
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.signal_models.gaussian_models import G1Ball


def _design():
    return design_waveform_now(1.0, G_max=0.08, slew_rate_max=200.0, TE=0.09,
                               n_t=100, n_restarts=3, seed=0)


def _multi_b_scheme(d, fracs):
    """A multi-b acquisition scheme from one design, scaling amplitude (b ∝ |G|²). The
    effective (sign-folded) gradient with echo_idx at the end is the convention both engines
    share — matching NowDesign.to_sim_waveform."""
    Geff = np.asarray(d.effective_G())
    G = np.stack([Geff * np.sqrt(f) for f in fracs]).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AcquisitionScheme.from_btensor_waveform(
            G, d.dt, echo_idx=Geff.shape[0] - 1, allow_offcenter_180=True)


def test_design_b_matches_sim():
    """The b-value the designer reports is the b dmipy-sim computes for the same waveform."""
    d = _design()
    b_sim = float(np.asarray(ds.calc_b(d.to_sim_waveform())).ravel()[0])
    assert abs(b_sim - d.b_value) / d.b_value < 1e-3        # same language, no drift


def test_sim_reproduces_stejskal_tanner_on_designed_waveform():
    """dmipy-sim eats the designed pulse and returns exp(-b·D) for free diffusion."""
    d = _design()
    D = 1.7e-9
    S = float(np.asarray(ds.simulate(50_000, D, d.to_sim_waveform(), ds.FreeDiffusion(),
                                     seed=0, require_gpu=False)).ravel()[0])
    assert abs(S - np.exp(-d.b_value * D)) < 0.02          # MC noise floor


def test_design_sim_fit_recovers_diffusivity():
    """Full loop: one design → one shared scheme → dmipy-sim simulates → dmipy-fit recovers D."""
    d = _design()
    scheme = _multi_b_scheme(d, np.array([0.0, 0.3, 0.6, 1.0]))
    D_true = 1.7e-9
    sig = np.asarray(ds.simulate(60_000, D_true, scheme, ds.FreeDiffusion(),
                                 seed=0, require_gpu=False)).ravel()
    fit = MultiCompartmentModel([G1Ball()]).fit(scheme, sig[None, :], solver="brute2fine")
    D_fit = float(np.asarray(fit.fitted_parameters["G1Ball_1_lambda_iso"]).ravel()[0])
    assert abs(D_fit - D_true) / D_true < 0.05             # recovered within 5%
