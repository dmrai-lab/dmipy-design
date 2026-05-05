"""
Integration tests for the two-term pricing objective (FIM + MC bias).

Coverage:
  1. solve_pricing with mc_bias_weight=0.0 gives same result as without bank
  2. solve_pricing with mc_bias_weight=0.1 and synthetic bank completes without error
  3. Returned atom has valid (finite, positive) rc value
  4. column_generation_oed accepts substrate_bank without error
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

GAMMA = 267513000.0  # rad/(s·T)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_prior():
    """Ball-only prior (P=1) for fast pricing tests."""
    rng = np.random.default_rng(42)
    M = 8
    D = rng.uniform(1e-9, 3e-9, size=(M, 1))
    return jnp.array(D, dtype=jnp.float64)


@pytest.fixture(scope="module")
def ball_forward():
    """Simple isotropic Ball forward function: E = exp(-b * D)."""
    def _forward(theta, scheme):
        b = scheme.bvalues.astype(jnp.float64)
        D = theta[0].astype(jnp.float64)
        return jnp.exp(-b * D)
    return _forward


@pytest.fixture(scope="module")
def tiny_bank():
    """Very small synthetic bank for fast integration testing."""
    from dmipy_design.substrate_bank_synthetic import make_synthetic_bank
    # 200 walkers, 21 time steps — tiny but enough to exercise the code path
    return make_synthetic_bank(n_walkers=200, n_t=21, dt=1e-4, seed=0)


# ---------------------------------------------------------------------------
# Test 1: mc_bias_weight=0 gives same result as no bank
# ---------------------------------------------------------------------------

def test_pricing_mc_weight_zero_matches_no_bank(small_prior, ball_forward, tiny_bank):
    """mc_bias_weight=0 must be backward-compatible with no bank."""
    from dmipy_design.optimizers.pricing_problem import (
        solve_pricing, CONNECTOM_3T
    )

    P = small_prior.shape[1]
    F_inv = np.eye(P, dtype=np.float64)

    # Without bank
    rc_no_bank, params_no_bank, _ = solve_pricing(
        ball_forward, small_prior, 0.05, F_inv,
        'pgse', n_restarts=4, rng_seed=0, lbfgs_maxiter=20,
        hardware=CONNECTOM_3T,
        substrate_bank=None, mc_bias_weight=0.0,
    )

    # With bank but mc_bias_weight=0 (bank is ignored)
    rc_zero_weight, params_zero_weight, _ = solve_pricing(
        ball_forward, small_prior, 0.05, F_inv,
        'pgse', n_restarts=4, rng_seed=0, lbfgs_maxiter=20,
        hardware=CONNECTOM_3T,
        substrate_bank=tiny_bank, mc_bias_weight=0.0,
    )

    # Results should be identical (same optimizer, same seed, bank ignored)
    assert abs(rc_no_bank - rc_zero_weight) < 1e-4, (
        f"mc_bias_weight=0 should give same rc as no bank: "
        f"{rc_no_bank:.6f} vs {rc_zero_weight:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: mc_bias_weight>0 with synthetic bank completes without error
# ---------------------------------------------------------------------------

def test_pricing_mc_weight_nonzero_runs(small_prior, ball_forward, tiny_bank):
    """solve_pricing with mc_bias_weight=0.1 must complete without error."""
    from dmipy_design.optimizers.pricing_problem import (
        solve_pricing, CONNECTOM_3T
    )

    P = small_prior.shape[1]
    F_inv = np.eye(P, dtype=np.float64)

    rc, params, scheme = solve_pricing(
        ball_forward, small_prior, 0.05, F_inv,
        'pgse', n_restarts=4, rng_seed=1, lbfgs_maxiter=20,
        hardware=CONNECTOM_3T,
        substrate_bank=tiny_bank, mc_bias_weight=0.1, fim_weight=1.0,
    )

    assert np.isfinite(rc), f"rc must be finite, got {rc}"
    assert 'type' in params
    assert params['type'] == 'pgse'
    assert 'b' in params
    assert np.isfinite(params['b']), f"b must be finite, got {params['b']}"


# ---------------------------------------------------------------------------
# Test 3: Returned atom has valid rc value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wtype", ["pgse", "ogse"])
def test_pricing_returns_valid_atom(small_prior, ball_forward, tiny_bank, wtype):
    """Best atom rc must be positive and finite for both PGSE and OGSE."""
    from dmipy_design.optimizers.pricing_problem import (
        solve_pricing, CONNECTOM_3T
    )

    P = small_prior.shape[1]
    F_inv = np.eye(P, dtype=np.float64)

    rc, params, scheme = solve_pricing(
        ball_forward, small_prior, 0.05, F_inv,
        wtype, n_restarts=4, rng_seed=2, lbfgs_maxiter=20,
        hardware=CONNECTOM_3T,
        substrate_bank=tiny_bank, mc_bias_weight=0.05, fim_weight=1.0,
    )

    assert np.isfinite(rc), f"rc must be finite for {wtype}, got {rc}"
    assert rc > 0, f"rc should be positive for {wtype}, got {rc}"
    assert params['type'] == wtype


# ---------------------------------------------------------------------------
# Test 4: column_generation_oed accepts substrate_bank
# ---------------------------------------------------------------------------

def test_column_generation_accepts_bank(small_prior, ball_forward, tiny_bank):
    """column_generation_oed with substrate_bank runs for 2 iterations."""
    from dmipy_design.optimizers.column_generation import (
        column_generation_oed, Atom
    )
    from dmipy_design.jax_scheme_encoder import encode_pgse_shell
    from dmipy_design.optimizers.pricing_problem import BVECS_30, CONNECTOM_3T

    bvecs = jnp.array(BVECS_30, dtype=jnp.float64)
    initial_scheme = encode_pgse_shell(1e9, 0.02, 0.04, bvecs)
    initial_atom   = Atom(type='pgse',
                          params={'b': 1e9, 'delta': 0.02, 'Delta': 0.04},
                          scheme=initial_scheme)

    result = column_generation_oed(
        forward_fn=ball_forward,
        prior_samples=small_prior,
        sigma=0.05,
        waveform_types=['pgse'],
        initial_atoms=[initial_atom],
        max_iter=2,
        n_pricing_restarts=4,
        lbfgs_maxiter=20,
        verbose=False,
        hardware=CONNECTOM_3T,
        substrate_bank=tiny_bank,
        mc_bias_weight=0.1,
        fim_weight=1.0,
    )

    assert len(result.atoms) >= 1
    assert len(result.history) >= 1
    assert np.isfinite(result.final_obj)
