"""
Tests for SubstrateBank, SubstrateEntry, apply_waveform_jax, and
SubstrateBank.compute_bias_jax.

Coverage:
  1. SubstrateEntry creation and field access
  2. apply_waveform_jax returns correct shape
  3. apply_waveform_jax: free diffusion at b~0 (near-zero G) gives signal ≈ 1.0
  4. apply_waveform_jax is differentiable (jax.grad does not crash)
  5. SubstrateBank.compute_bias_jax returns a scalar
  6. walker_weights: non-uniform weights give different result from uniform
  7. Synthetic bank creation via make_synthetic_bank
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

# Enable float64 for FIM consistency
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_bank():
    from dmipy_design.substrate_bank_synthetic import make_synthetic_bank
    return make_synthetic_bank(n_walkers=500, n_t=51, dt=1e-4, seed=7)


@pytest.fixture(scope="module")
def simple_entry(synthetic_bank):
    return synthetic_bank.entries[0]


@pytest.fixture(scope="module")
def small_G():
    """Low-amplitude PGSE G array — signal ≈ 1."""
    bvecs = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=jnp.float32)
    n_t = 51
    G = jnp.zeros((3, n_t, 3), dtype=jnp.float32)   # near-zero amplitude
    return G


@pytest.fixture(scope="module")
def nonzero_G():
    """Moderate-amplitude G array for differentiability tests."""
    bvecs = jnp.array([[1., 0., 0.], [0., 1., 0.]], dtype=jnp.float32)
    n_t = 51
    dt = 1e-4
    delta = 10
    t = np.arange(n_t, dtype=np.float32)
    lobe1 = np.where((t >= 0) & (t < delta), 1.0, 0.0)
    lobe2 = np.where((t >= 20) & (t < 20 + delta), -1.0, 0.0)
    shape = (lobe1 + lobe2).astype(np.float32)
    G_amp = 0.05  # T/m
    G = G_amp * jnp.einsum('mx,t->mtx',
                            jnp.array([[1., 0., 0.], [0., 1., 0.]], dtype=jnp.float32),
                            jnp.array(shape))
    return G


# ---------------------------------------------------------------------------
# Test 1: SubstrateEntry creation
# ---------------------------------------------------------------------------

def test_substrate_entry_creation(simple_entry):
    from dmipy_design.substrate_bank import SubstrateEntry
    assert isinstance(simple_entry, SubstrateEntry)
    assert simple_entry.substrate_id == "synthetic_free"
    assert simple_entry.trajectories.shape[0] == 500   # n_walkers
    assert simple_entry.trajectories.shape[2] == 3     # xyz
    assert simple_entry.dt_traj == 1e-4
    assert float(simple_entry.biological_weight) == 1.0
    assert simple_entry.walker_weights is None
    assert simple_entry.theta_nominal.shape == (1,)


# ---------------------------------------------------------------------------
# Test 2: apply_waveform_jax returns correct shape
# ---------------------------------------------------------------------------

def test_apply_waveform_jax_shape(simple_entry, nonzero_G):
    from dmipy_sim.trajectories import apply_waveform_jax

    n_meas = nonzero_G.shape[0]
    signals = apply_waveform_jax(
        nonzero_G, simple_entry.trajectories, simple_entry.dt_traj
    )
    assert signals.shape == (n_meas,), f"Expected ({n_meas},), got {signals.shape}"


# ---------------------------------------------------------------------------
# Test 3: At b~0 (near-zero G), signal ≈ 1.0
# ---------------------------------------------------------------------------

def test_apply_waveform_jax_b_zero(simple_entry, small_G):
    from dmipy_sim.trajectories import apply_waveform_jax

    signals = apply_waveform_jax(
        small_G, simple_entry.trajectories, simple_entry.dt_traj
    )
    # All signals should be ≈ 1 (cosine of zero phase)
    np.testing.assert_allclose(np.array(signals), 1.0, atol=1e-3,
                               err_msg="Signal at b≈0 should be ≈ 1.0")


# ---------------------------------------------------------------------------
# Test 4: apply_waveform_jax is differentiable w.r.t. G
# ---------------------------------------------------------------------------

def test_apply_waveform_jax_differentiable(simple_entry):
    from dmipy_sim.trajectories import apply_waveform_jax

    traj = simple_entry.trajectories
    dt   = simple_entry.dt_traj
    n_t  = traj.shape[1]
    G0   = jnp.zeros((1, n_t, 3), dtype=jnp.float32)

    def objective(G):
        signals = apply_waveform_jax(G, traj, dt)
        return jnp.sum(signals)

    # This must not raise
    grad = jax.grad(objective)(G0)
    assert grad.shape == G0.shape, f"Gradient shape mismatch: {grad.shape}"
    # At G=0 the gradient of cos is -sin(0)=0, so gradient is 0 — just verify no NaN
    assert not jnp.any(jnp.isnan(grad)), "NaN in gradient at G=0"


# ---------------------------------------------------------------------------
# Test 5: SubstrateBank.compute_bias_jax returns scalar
# ---------------------------------------------------------------------------

def test_compute_bias_jax_scalar(synthetic_bank):
    from dmipy_design.waveform_builders import build_pgse_G
    from dmipy_design.jax_scheme_encoder import encode_pgse_shell

    n_dirs = 6
    bvecs = jnp.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 1., 0.],
                       [1., 0., 1.],
                       [0., 1., 1.]], dtype=jnp.float32)
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)

    dt   = 1e-4
    G_amp = 0.05
    delta, Delta = 5e-4 * 10, 5e-4 * 25   # 5e-3 s, 12.5e-3 s (tiny for speed)
    G = build_pgse_G(G_amp, delta, Delta, bvecs, dt)

    b_val = 267513000.0**2 * G_amp**2 * delta**2 * (Delta - delta / 3.0)
    scheme = encode_pgse_shell(b_val, delta, Delta, bvecs.astype(jnp.float64))

    def simple_forward(theta, scheme):
        # Free diffusion: E = exp(-b * D)
        b = scheme.bvalues.astype(jnp.float32)
        D = theta[0]
        return jnp.exp(-b * D)

    bias = synthetic_bank.compute_bias_jax(G, dt, simple_forward, scheme, sigma=0.05)
    assert bias.shape == (), f"Expected scalar, got shape {bias.shape}"
    assert not jnp.isnan(bias), "Bias is NaN"
    assert float(bias) >= 0.0, f"Bias should be non-negative, got {float(bias)}"


# ---------------------------------------------------------------------------
# Test 6: walker_weights non-uniform gives different result from uniform
# ---------------------------------------------------------------------------

def test_walker_weights_affect_result(simple_entry):
    from dmipy_sim.trajectories import apply_waveform_jax

    traj = simple_entry.trajectories
    dt   = simple_entry.dt_traj
    n_t  = traj.shape[1]

    G_amp = 0.10   # T/m — moderate attenuation
    n_meas = 2
    bvecs_np = np.zeros((n_meas, 3), dtype=np.float32)
    bvecs_np[:, 0] = 1.0
    # Build PGSE shape
    delta_idx = max(1, n_t // 5)
    Delta_idx  = max(delta_idx + 1, n_t // 2)
    shape = np.zeros(n_t, dtype=np.float32)
    shape[:delta_idx] = 1.0
    shape[Delta_idx:Delta_idx + delta_idx] = -1.0
    G = G_amp * jnp.einsum('mx,t->mtx', jnp.array(bvecs_np), jnp.array(shape))

    n_walkers = traj.shape[0]
    # Uniform weights
    sig_uniform = apply_waveform_jax(G, traj, dt, walker_weights=None)
    # Strongly skewed weights (concentrate on first 10% of walkers)
    w = np.zeros(n_walkers, dtype=np.float32)
    w[:n_walkers // 10] = 1.0
    w[n_walkers // 10:] = 0.001
    sig_weighted = apply_waveform_jax(G, traj, dt,
                                      walker_weights=jnp.array(w))

    # They should differ (different walkers emphasised)
    diff = float(jnp.max(jnp.abs(sig_uniform - sig_weighted)))
    assert diff > 1e-4, (
        f"Uniform and strongly-skewed walker weights produced identical signals "
        f"(diff={diff:.2e}); expected difference."
    )


# ---------------------------------------------------------------------------
# Test 7: make_synthetic_bank produces bank with correct structure
# ---------------------------------------------------------------------------

def test_make_synthetic_bank():
    from dmipy_design.substrate_bank_synthetic import make_synthetic_bank
    from dmipy_design.substrate_bank import SubstrateBank, SubstrateEntry

    bank = make_synthetic_bank(n_walkers=100, n_t=21, dt=1e-4, seed=0)
    assert isinstance(bank, SubstrateBank)
    assert len(bank.entries) == 1
    entry = bank.entries[0]
    assert isinstance(entry, SubstrateEntry)
    assert entry.trajectories.shape == (100, 21, 3)
    assert entry.dt_traj == 1e-4
    assert float(entry.biological_weight) == 1.0
    assert entry.walker_weights is None
    assert entry.meta["type"] == "synthetic_brownian"
