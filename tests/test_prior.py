"""Tests for prior sampling.

Test IDs
--------
T3-PRIOR-1  sample_prior returns (n_samples, n_params) array
T3-PRIOR-2  all samples are within parameter ranges (after scale)
"""

import jax
jax.config.update("jax_enable_x64", True)


def test_sample_prior_shape():
    """sample_prior returns (n_samples, n_params) array."""
    from dmipy_fit.signal_models.gaussian_models import G1Ball
    from dmipy_fit.core.modeling_framework import MultiCompartmentModel
    from dmipy_design.prior import sample_prior
    model = MultiCompartmentModel([G1Ball()])
    samples = sample_prior(model, n_samples=50, seed=0)
    assert samples.shape[0] == 50
    assert samples.shape[1] == len(model.parameter_ranges)


def test_sample_prior_within_range():
    """All samples are within parameter ranges (after scale)."""
    from dmipy_fit.signal_models.gaussian_models import G1Ball
    from dmipy_fit.core.modeling_framework import MultiCompartmentModel
    from dmipy_design.prior import sample_prior
    model = MultiCompartmentModel([G1Ball()])
    samples = sample_prior(model, n_samples=200, seed=1)
    # Samples are scaled: sample = raw / scale, so raw = sample * scale
    scales = list(model.parameter_scales.values())
    raw = samples * scales
    for i, (name, (lo, hi)) in enumerate(model.parameter_ranges.items()):
        assert (raw[:, i] >= lo - 1e-8).all(), f"{name}: sample below lower bound"
        assert (raw[:, i] <= hi + 1e-8).all(), f"{name}: sample above upper bound"
