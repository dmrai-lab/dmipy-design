"""
Greedy sequential optimal experiment design.

Given n measurements already chosen, find the (n+1)-th measurement that
maximally increases det(FIM).  Efficient for small additions and online
protocol adaptation.
"""

import jax.numpy as jnp


def greedy_add_measurement(
    current_scheme,
    candidate_pool: list,
    forward_fn,
    prior_samples: jnp.ndarray,
    sigma: float,
) -> int:
    """Find the candidate measurement that most increases log det(FIM).

    Parameters
    ----------
    current_scheme : JaxScheme or None
        Currently selected measurements.  None = start from scratch.
    candidate_pool : list of JaxScheme
        Pool of candidate single-measurement schemes to evaluate.
    forward_fn : callable
    prior_samples : jnp.ndarray, shape (M, P)
    sigma : float

    Returns
    -------
    best_idx : int
        Index into ``candidate_pool`` of the best next measurement.
    """
    from ..fim import compute_fim_averaged
    from ..objectives import d_optimal

    if current_scheme is None:
        current_FIM = jnp.zeros((prior_samples.shape[1], prior_samples.shape[1]))
    else:
        current_FIM = compute_fim_averaged(forward_fn, prior_samples, current_scheme, sigma)

    best_idx = 0
    best_gain = -jnp.inf
    for i, candidate in enumerate(candidate_pool):
        candidate_FIM = compute_fim_averaged(forward_fn, prior_samples, candidate, sigma)
        combined_FIM = current_FIM + candidate_FIM
        gain = -d_optimal(combined_FIM)   # higher is better
        if gain > best_gain:
            best_gain = gain
            best_idx = i
    return best_idx
