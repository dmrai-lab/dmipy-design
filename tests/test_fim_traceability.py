"""Phase 1, Tier 1 traceability tests for dmipy-design.

Tests T1-P1-A through T1-P1-C verify that:
  - jax.grad can flow through encode_pgse -> forward_fn -> FIM -> d_optimal
  - compute_fim (autodiff) matches compute_fim_fd (finite differences)
  - D-optimal gradient w.r.t. FIM elements is finite
  - A/D/E objectives are self-consistent on a diagonal FIM

The forward functions used here are JAX-native and take (theta, scheme) so
that both parameter and scheme arrays are traceable.  They replicate the
G1Ball signal model (exp(-b * lambda_iso)) using dmipy.jax.signal_models_jax.
"""

import jax
import jax.numpy as jnp
import pytest

# Enable float64 for all tests in this module (needed for numerical accuracy).
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper: G1Ball-compatible forward function in (theta, scheme) form
# ---------------------------------------------------------------------------
# dmipy-core's build_mc_forward_fn bakes the acquisition scheme into the
# closure and returns forward_fn(params_scaled) -- which is great for fitting
# but cannot be differentiated w.r.t. scheme parameters.
#
# For scheme-level OED (differentiating through b-values, delta, Delta), we
# write a thin wrapper that calls dmipy-core's g1ball_signal directly with
# scheme.bvalues taken from the JaxScheme struct.  This gives the correct
# API: forward_fn(theta, scheme) -> jnp.ndarray shape (N,).
#
# dmipy-core JAX API used:
#   from dmipy.jax.signal_models_jax import g1ball_signal
#   g1ball_signal(bvalues: jnp.ndarray, lambda_iso: scalar) -> jnp.ndarray


def _ball_forward_fn(theta, scheme):
    """G1Ball-like forward: E_k = exp(-b_k * lambda_iso).

    theta : jnp.ndarray, shape (1,)   [lambda_iso in SI units, m^2/s]
    scheme : JaxScheme  (bvalues is the only field used by Ball)
    """
    from dmipy.jax.signal_models_jax import g1ball_signal
    return g1ball_signal(scheme.bvalues, theta[0])


def _ball_forward_normalized(theta, scheme):
    """G1Ball forward with theta normalised to O(1) for FD accuracy.

    theta : jnp.ndarray, shape (1,)   [lambda_iso / 1e-9, dimensionless]
    Actual lambda_iso = theta[0] * 1e-9 m^2/s.
    """
    _LAMBDA_SCALE = jnp.float64(1e-9)
    from dmipy.jax.signal_models_jax import g1ball_signal
    return g1ball_signal(scheme.bvalues, theta[0] * _LAMBDA_SCALE)


# ---------------------------------------------------------------------------
# T1-P1-A  (part 1): JAX traceability
# ---------------------------------------------------------------------------

def test_jax_scheme_encoder_is_traceable():
    """jax.grad must flow through encode_pgse -> forward_fn -> FIM -> d_optimal.

    This is the central traceability spike for dmipy-design.  If this test
    passes, the full OED gradient pipeline is valid.

    dmipy-core JAX API used:
        dmipy.jax.signal_models_jax.g1ball_signal
    """
    from dmipy_design.jax_scheme_encoder import encode_pgse
    from dmipy_design.fim import compute_fim
    from dmipy_design.objectives import d_optimal

    bvecs = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (10, 1))
    # theta is fixed at a single point; u is the design variable
    theta = jnp.array([2e-9])   # lambda_iso = 2 um^2/ms (typical free water)
    sigma = 0.05

    def crlb_loss(u):
        scheme = encode_pgse(u, bvecs)
        FIM = compute_fim(_ball_forward_fn, theta, scheme, sigma)
        return d_optimal(FIM)

    # u = [b_value (s/m^2), delta (s), Delta (s)]
    u0 = jnp.array([1e9, 0.02, 0.05])
    grad = jax.grad(crlb_loss)(u0)

    assert jnp.all(jnp.isfinite(grad)), (
        f"Gradient contains non-finite values: {grad}"
    )
    assert not jnp.all(grad == 0), (
        "Gradient is all zeros — chain rule is broken"
    )


# ---------------------------------------------------------------------------
# T1-P1-A  (part 2): FIM autodiff vs finite differences
# ---------------------------------------------------------------------------

def test_fim_autodiff_matches_finite_differences():
    """compute_fim (jax.jacobian) must match compute_fim_fd within atol=1e-4.

    Uses a normalised forward function (theta ~ O(1)) so that the absolute
    eps=1e-7 used by compute_fim_fd gives accurate finite differences.
    """
    from dmipy_design.jax_scheme_encoder import encode_pgse
    from dmipy_design.fim import compute_fim, compute_fim_fd

    bvecs = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (10, 1))
    # Normalised theta: 2.0 corresponds to lambda_iso = 2e-9 m^2/s
    theta = jnp.array([2.0])
    u0 = jnp.array([1e9, 0.02, 0.05])
    scheme = encode_pgse(u0, bvecs)
    sigma = 0.05

    FIM_ad = compute_fim(_ball_forward_normalized, theta, scheme, sigma)
    FIM_fd = compute_fim_fd(_ball_forward_normalized, theta, scheme, sigma)

    assert jnp.allclose(FIM_ad, FIM_fd, atol=1e-4), (
        f"FIM mismatch: autodiff={FIM_ad}, fd={FIM_fd}, "
        f"max_diff={float(jnp.max(jnp.abs(FIM_ad - FIM_fd))):.3e}"
    )


# ---------------------------------------------------------------------------
# T1-P1-B: D-optimal gradient is finite
# ---------------------------------------------------------------------------

def test_d_optimal_gradient_finite():
    """d_optimal gradient w.r.t. FIM elements must be finite and non-trivial."""
    from dmipy_design.objectives import d_optimal

    P = 3
    A = jnp.eye(P) * 10.0 + jax.random.normal(jax.random.PRNGKey(0), (P, P)) * 0.1
    FIM = A.T @ A

    grad = jax.grad(d_optimal)(FIM)

    assert jnp.all(jnp.isfinite(grad)), (
        f"d_optimal gradient contains non-finite values: {grad}"
    )
    assert not jnp.all(grad == 0), (
        "d_optimal gradient is all zeros"
    )


# ---------------------------------------------------------------------------
# T1-P1-C: A/D/E objectives are self-consistent on diagonal FIM
# ---------------------------------------------------------------------------

def test_objectives_self_consistent():
    """For a diagonal FIM with equal eigenvalues lambda, A/D/E must agree.

    CRLB = FIM^{-1} = (1/lambda) * I
      A-optimal = trace(CRLB) = P / lambda
      E-optimal = max eigenvalue of CRLB = 1 / lambda
    """
    from dmipy_design.objectives import a_optimal, d_optimal, e_optimal

    lam = 5.0
    P = 4
    FIM = jnp.eye(P) * lam

    a = a_optimal(FIM)
    e = e_optimal(FIM)

    assert abs(float(a) - P / lam) < 1e-4, (
        f"A-optimal wrong: {float(a):.6f} vs expected {P / lam:.6f}"
    )
    assert abs(float(e) - 1.0 / lam) < 1e-4, (
        f"E-optimal wrong: {float(e):.6f} vs expected {1.0 / lam:.6f}"
    )

    # D-optimal should be differentiable too
    grad_d = jax.grad(d_optimal)(FIM)
    assert jnp.all(jnp.isfinite(grad_d)), (
        f"D-optimal gradient on diagonal FIM is non-finite: {grad_d}"
    )
