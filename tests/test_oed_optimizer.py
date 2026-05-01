"""Tests for gradient_oed (jaxopt.LBFGSB) and greedy_add_measurement.

Test IDs
--------
T2-OPT-1  gradient_oed reduces D-optimal objective
T2-OPT-2  gradient_oed u_opt is within hardware bounds
T2-OPT-3  greedy_add_measurement selects high b-value over b=0
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Float64 required for numerical accuracy in FIM differentiation.
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared forward function: G1Ball with normalised theta
# ---------------------------------------------------------------------------

LAMBDA_SCALE = 3e-9  # maps theta[0] ∈ (0, 1] to lambda_iso ∈ (0, 3e-9] m²/s


def _ball_forward_fn(theta, scheme):
    """G1Ball: E_k = exp(-b_k * lambda_iso), lambda_iso = theta[0] * LAMBDA_SCALE."""
    from dmipy_fit.jax.signal_models_jax import g1ball_signal
    return g1ball_signal(scheme.bvalues, theta[0] * LAMBDA_SCALE)


# ---------------------------------------------------------------------------
# T2-OPT-1: gradient_oed reduces D-optimal objective
# ---------------------------------------------------------------------------

def test_gradient_oed_decreases_d_optimal():
    """After optimisation, D-optimal CRLB must be lower than at initialisation."""
    from dmipy_design.jax_scheme_encoder import encode_pgse
    from dmipy_design.fim import compute_fim_averaged
    from dmipy_design.objectives import d_optimal
    from dmipy_design.optimizers.gradient_based import gradient_oed
    from dmipy_design.constraints import HardwareConstraints

    rng = np.random.default_rng(42)
    # theta ~ Uniform[0.33, 1.0] → lambda_iso ~ Uniform[1e-9, 3e-9]
    prior_samples = jnp.array(rng.uniform(0.33, 1.0, (20, 1)))

    # 4 measurements along z
    bvecs = jnp.array(np.tile([0.0, 0.0, 1.0], (4, 1)))
    sigma = 0.05
    constraints = HardwareConstraints()

    # Start well away from the optimum: b=200 s/mm² = 200e6 s/m² (sub-optimal)
    # The D-optimal b is around 600 s/mm², so this gives clear room for improvement.
    u0 = jnp.array([200e6, 0.02, 0.04])

    u_opt, crlb_opt = gradient_oed(
        _ball_forward_fn, prior_samples, bvecs, sigma, u0,
        constraints, objective="D", max_iter=200,
    )

    # Compute initial CRLB for comparison
    scheme0 = encode_pgse(u0, bvecs)
    FIM0 = compute_fim_averaged(_ball_forward_fn, prior_samples, scheme0, sigma)
    crlb_init = float(d_optimal(FIM0))

    assert crlb_opt < crlb_init, (
        f"Optimiser did not improve D-optimal: opt={crlb_opt:.4f} vs init={crlb_init:.4f}"
    )
    assert jnp.isfinite(u_opt).all(), f"u_opt contains non-finite values: {u_opt}"


# ---------------------------------------------------------------------------
# T2-OPT-2: gradient_oed u_opt is within hardware bounds
# ---------------------------------------------------------------------------

def test_gradient_oed_u_opt_within_bounds():
    """Optimised u must stay within the hardware box bounds."""
    from dmipy_design.optimizers.gradient_based import gradient_oed
    from dmipy_design.constraints import HardwareConstraints

    rng = np.random.default_rng(7)
    prior_samples = jnp.array(rng.uniform(0.33, 1.0, (20, 1)))
    bvecs = jnp.array(np.tile([0.0, 0.0, 1.0], (4, 1)))
    sigma = 0.05
    constraints = HardwareConstraints()
    # Start sub-optimally; LBFGSB must stay within the hardware box
    u0 = jnp.array([200e6, 0.02, 0.04])

    u_opt, _ = gradient_oed(
        _ball_forward_fn, prior_samples, bvecs, sigma, u0,
        constraints, objective="D", max_iter=50,
    )

    b_opt = float(u_opt[0])
    delta_opt = float(u_opt[1])
    Delta_opt = float(u_opt[2])

    assert 100e6 <= b_opt <= 10000e6, (
        f"b_opt={b_opt:.3e} s/m² outside [100e6, 10000e6]"
    )
    assert 0.001 <= delta_opt <= 0.060, (
        f"delta_opt={delta_opt:.4f} s outside [0.001, 0.060]"
    )
    assert 0.005 <= Delta_opt <= 0.100, (
        f"Delta_opt={Delta_opt:.4f} s outside [0.005, 0.100]"
    )
    # Physical requirement: diffusion time >= pulse duration
    assert delta_opt < Delta_opt, (
        f"delta_opt={delta_opt:.4f} >= Delta_opt={Delta_opt:.4f} (non-physical)"
    )


# ---------------------------------------------------------------------------
# T2-OPT-3: greedy_add_measurement selects high b-value over b=0
# ---------------------------------------------------------------------------

def test_greedy_selects_high_bvalue_over_zero():
    """For a Ball model, b=1000 s/mm² adds more FIM information than b=0."""
    from dmipy_design.optimizers.greedy_sequential import greedy_add_measurement
    from dmipy_design.jax_scheme_encoder import JaxScheme

    # Fixed theta: lambda_iso = 0.7 * 3e-9 = 2.1e-9 m²/s
    prior_samples = jnp.array(np.full((10, 1), 0.7))
    bvec = jnp.array([[0.0, 0.0, 1.0]])
    sigma = 0.05

    # Candidate 0: b=0 (no diffusion weighting, zero gradient — no information)
    scheme_b0 = JaxScheme(
        bvalues=jnp.array([0.0]),
        bvecs=bvec,
        delta=jnp.array([0.01]),
        Delta=jnp.array([0.03]),
    )
    # Candidate 1: b=1000 s/mm² = 1000e6 s/m²
    scheme_b1000 = JaxScheme(
        bvalues=jnp.array([1000e6]),
        bvecs=bvec,
        delta=jnp.array([0.01]),
        Delta=jnp.array([0.03]),
    )

    best_idx = greedy_add_measurement(
        current_scheme=None,
        candidate_pool=[scheme_b0, scheme_b1000],
        forward_fn=_ball_forward_fn,
        prior_samples=prior_samples,
        sigma=sigma,
    )

    assert best_idx == 1, (
        f"Expected b=1000 s/mm² (idx=1) to be selected as more informative, "
        f"but got idx={best_idx}"
    )
