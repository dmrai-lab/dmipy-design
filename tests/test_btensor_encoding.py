"""Phase 2 tests: B-tensor encoding — STE and OGSE scheme support.

Tests:
  1. test_ste_isotropy        — STE scheme signal (ball model) is identical
                                regardless of bvec direction.
  2. test_ste_fim_trace       — compute_fim_averaged returns a valid (non-NaN,
                                positive semi-definite) FIM for STE scheme.
  3. test_ogse_effective_time — OGSE scheme at f=50 Hz has t_eff ≈ 5 ms;
                                verify it is stored correctly.
  4. test_encode_ste_shape    — encoded STE scheme has correct b_tensor shape
                                (N, 3, 3) with isotropic diagonal.
  5. test_encode_lte_alias    — encode_lte returns same result as encode_pgse.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ball_forward_fn(theta, scheme):
    """G1Ball-like forward: E_k = exp(-b_k * lambda_iso).

    theta : jnp.ndarray, shape (1,)   [lambda_iso in SI units, m^2/s]
    scheme : JaxScheme  (bvalues is the only field used by Ball)
    """
    from dmipy.jax.signal_models_jax import g1ball_signal
    return g1ball_signal(scheme.bvalues, theta[0])


# ---------------------------------------------------------------------------
# 1. STE isotropy: signal is direction-independent for ball model
# ---------------------------------------------------------------------------

def test_ste_isotropy():
    """STE scheme signal (ball model) is identical regardless of bvec direction.

    For STE with a Ball (isotropic diffusion) model, the signal is
    E = exp(-b * d) regardless of the gradient direction because the
    b-tensor is isotropic.  The forward model only uses bvalues, so two
    STE schemes with different bvecs but the same b-value must give the
    same signal.
    """
    from dmipy_design.jax_scheme_encoder import encode_ste
    from dmipy_design.fim import _scheme_to_jaxscheme

    N = 20
    u = jnp.array([1e9, 0.02, 0.05])

    # Two sets of bvecs pointing in completely different directions
    bvecs_x = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (N, 1))
    bvecs_z = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (N, 1))

    scheme_x = encode_ste(u, bvecs_x)
    scheme_z = encode_ste(u, bvecs_z)

    theta = jnp.array([2e-9])   # lambda_iso = 2 µm²/ms

    jax_scheme_x = _scheme_to_jaxscheme(scheme_x)
    jax_scheme_z = _scheme_to_jaxscheme(scheme_z)

    E_x = _ball_forward_fn(theta, jax_scheme_x)
    E_z = _ball_forward_fn(theta, jax_scheme_z)

    assert jnp.allclose(E_x, E_z, atol=1e-12), (
        f"STE signal should be direction-independent for ball model, "
        f"but got max diff {float(jnp.max(jnp.abs(E_x - E_z))):.3e}"
    )


# ---------------------------------------------------------------------------
# 2. STE FIM: compute_fim_averaged returns valid PSD FIM
# ---------------------------------------------------------------------------

def test_ste_fim_trace():
    """compute_fim_averaged returns valid (non-NaN, PSD) FIM for STE scheme."""
    from dmipy_design.jax_scheme_encoder import encode_ste
    from dmipy_design.fim import compute_fim_averaged

    N = 20
    u = jnp.array([1e9, 0.02, 0.05])
    bvecs = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (N, 1))
    scheme = encode_ste(u, bvecs)

    # Single-parameter prior: lambda_iso ~ Uniform[1e-9, 3e-9]
    rng = jax.random.PRNGKey(0)
    prior_samples = jax.random.uniform(rng, (30, 1), minval=1e-9, maxval=3e-9)

    sigma = 0.05
    FIM = compute_fim_averaged(_ball_forward_fn, prior_samples, scheme, sigma)

    # 1. No NaNs or Infs
    assert jnp.all(jnp.isfinite(FIM)), (
        f"FIM contains non-finite values: {FIM}"
    )

    # 2. Positive semi-definite: all eigenvalues >= 0
    eigvals = jnp.linalg.eigvalsh(FIM)
    assert jnp.all(eigvals >= -1e-10), (
        f"FIM has negative eigenvalues: {eigvals}"
    )

    # 3. Non-trivial: FIM trace > 0
    assert float(jnp.trace(FIM)) > 0.0, (
        f"FIM trace is non-positive: {float(jnp.trace(FIM))}"
    )


# ---------------------------------------------------------------------------
# 3. OGSE effective time
# ---------------------------------------------------------------------------

def test_ogse_effective_time():
    """OGSE scheme at f=50 Hz has t_eff = 1/(4*50) = 5 ms."""
    from dmipy_design.jax_scheme_encoder import encode_ogse

    N = 15
    freq = 50.0   # Hz
    u_ogse = jnp.array([1e9, freq, 4.0])   # [b, freq, n_cycles]
    bvecs = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (N, 1))

    scheme = encode_ogse(u_ogse, bvecs)

    expected_t_eff = 1.0 / (4.0 * freq)   # = 0.005 s = 5 ms

    assert abs(float(scheme["t_eff"]) - expected_t_eff) < 1e-12, (
        f"OGSE t_eff wrong: got {float(scheme['t_eff']):.6f} s, "
        f"expected {expected_t_eff:.6f} s"
    )
    assert scheme["encoding"] == "OGSE"
    assert float(scheme["frequency"]) == freq
    assert float(scheme["n_cycles"]) == 4.0


# ---------------------------------------------------------------------------
# 4. STE b_tensor shape and isotropy
# ---------------------------------------------------------------------------

def test_encode_ste_shape():
    """Encoded STE scheme has b_tensor shape (N, 3, 3) with isotropic diagonal."""
    from dmipy_design.jax_scheme_encoder import encode_ste

    N = 25
    b_val = 2e9   # s/m²
    u = jnp.array([b_val, 0.025, 0.060])
    bvecs = jax.random.normal(jax.random.PRNGKey(1), (N, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)

    scheme = encode_ste(u, bvecs)

    # Shape check
    assert scheme["b_tensors"].shape == (N, 3, 3), (
        f"b_tensors shape wrong: {scheme['b_tensors'].shape}"
    )

    # All b-tensors should equal (b/3) * I
    expected_diag = b_val / 3.0
    B = scheme["b_tensors"]

    # Diagonal elements should equal b/3
    for i in range(3):
        diag_vals = B[:, i, i]
        assert jnp.allclose(diag_vals, expected_diag, atol=1e-6), (
            f"b_tensor diagonal[{i},{i}] wrong: "
            f"got {float(diag_vals[0]):.4e}, expected {expected_diag:.4e}"
        )

    # Off-diagonal elements should be zero
    for i in range(3):
        for j in range(3):
            if i != j:
                off_diag = B[:, i, j]
                assert jnp.allclose(off_diag, 0.0, atol=1e-10), (
                    f"b_tensor off-diagonal[{i},{j}] non-zero: {float(off_diag[0]):.4e}"
                )

    # b_values should equal b_val
    assert jnp.allclose(scheme["b_values"], b_val, atol=1e-6), (
        "b_values in STE scheme do not match the input b-value"
    )
    assert scheme["encoding"] == "STE"


# ---------------------------------------------------------------------------
# 5. LTE alias
# ---------------------------------------------------------------------------

def test_encode_lte_alias():
    """encode_lte returns a JaxScheme equivalent to encode_pgse."""
    from dmipy_design.jax_scheme_encoder import encode_lte, encode_pgse, JaxScheme

    N = 10
    u = jnp.array([1e9, 0.02, 0.05])
    bvecs = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (N, 1))

    scheme_pgse = encode_pgse(u, bvecs)
    scheme_lte = encode_lte(u, bvecs)

    assert isinstance(scheme_lte, JaxScheme), (
        "encode_lte should return a JaxScheme"
    )
    assert jnp.allclose(scheme_pgse.bvalues, scheme_lte.bvalues), (
        "LTE and PGSE bvalues differ"
    )
    assert jnp.allclose(scheme_pgse.delta, scheme_lte.delta), (
        "LTE and PGSE delta differ"
    )
    assert jnp.allclose(scheme_pgse.Delta, scheme_lte.Delta), (
        "LTE and PGSE Delta differ"
    )
