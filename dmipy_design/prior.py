"""
Parameter prior sampling for FIM averaging.

The FIM is averaged over a prior distribution p(theta) so that the optimised
protocol is robust across the expected range of tissue parameters.
"""

import numpy as np


def sample_prior(model, n_samples: int = 512, seed: int = 0) -> np.ndarray:
    """Draw uniform samples from dmipy-core model parameter ranges.

    Parameters
    ----------
    model : MultiCompartmentModel
        A dmipy-core model exposing ``parameter_ranges`` and
        ``parameter_scales``.
    n_samples : int
        Number of prior samples.
    seed : int

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
        Scaled parameter samples in [0, 1] (dmipy-core convention).
    """
    rng = np.random.default_rng(seed)
    lo = np.array([v[0] for v in model.parameter_ranges.values()], dtype=np.float64)
    hi = np.array([v[1] for v in model.parameter_ranges.values()], dtype=np.float64)
    raw = rng.uniform(lo, hi, size=(n_samples, len(lo)))
    scales = np.array(list(model.parameter_scales.values()), dtype=np.float64)
    return (raw / scales).astype(np.float32)
