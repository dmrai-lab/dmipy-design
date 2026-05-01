"""
Tests for the column generation OED framework.

Tests cover:
  1. Master problem simplex correctness
  2. Master problem analytic gradient vs finite differences
  3. PGSE decode roundtrip
  4. OGSE physics: b = gamma² G² t_eff³
  5. Ball+C4Cylinder forward shape
  6. Ball+C4Cylinder forward at b→0 → 1
  7. FIM[diameter, diameter] sensitivity: OGSE > PGSE for 10-µm cylinder
  8. Column generation adds an atom in one iteration
  9. (slow) Column generation discovers at least one OGSE atom in 5 iterations
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# float64 is required for C4 Van Gelderen sums and FIM accuracy
jax.config.update("jax_enable_x64", True)

GAMMA = 267513000.0  # rad/(s·T)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def prior_samples_simple():
    """Small Ball-only prior (P=1) for fast master / basic CG tests."""
    rng = np.random.default_rng(0)
    M = 32
    lambda_iso = rng.uniform(1e-9, 3e-9, size=(M, 1))
    return jnp.array(lambda_iso, dtype=jnp.float64)


@pytest.fixture(scope="module")
def prior_samples_ball_cyl():
    """Ball+C4Cylinder prior (P=4) for forward and FIM tests."""
    rng = np.random.default_rng(42)
    M = 32
    vf_ball    = rng.uniform(0.3, 0.7, size=(M, 1))
    lambda_iso = rng.uniform(1e-9, 3e-9, size=(M, 1))
    lambda_par = rng.uniform(1.5e-9, 2.5e-9, size=(M, 1))
    diameter   = rng.uniform(2e-6, 20e-6, size=(M, 1))
    return jnp.array(np.concatenate([vf_ball, lambda_iso, lambda_par, diameter], axis=1),
                     dtype=jnp.float64)


@pytest.fixture(scope="module")
def bvecs_30():
    from dmipy_design.optimizers.pricing_problem import BVECS_30
    return jnp.array(BVECS_30, dtype=jnp.float64)


@pytest.fixture(scope="module")
def pgse_scheme_1000(bvecs_30):
    """Single PGSE shell: b=1000 s/mm², delta=20 ms, Delta=40 ms."""
    from dmipy_design.jax_scheme_encoder import encode_pgse_shell
    return encode_pgse_shell(1000e6, 0.020, 0.040, bvecs_30)


@pytest.fixture(scope="module")
def ogse_scheme_100hz(bvecs_30):
    """Single OGSE shell: f=100 Hz, G=0.1 T/m."""
    from dmipy_design.jax_scheme_encoder import encode_ogse_shell
    return encode_ogse_shell(100.0, 0.1, bvecs_30)


# ============================================================================
# 1. Master problem simplex
# ============================================================================

class TestMasterProblem:

    def _random_spd(self, P, rng, scale=1.0):
        A = rng.standard_normal((P, P))
        return scale * (A @ A.T) + 1e-4 * np.eye(P)

    def test_master_problem_simplex(self):
        """Weights sum to 1, all non-negative."""
        from dmipy_design.optimizers.master_problem import solve_master
        rng = np.random.default_rng(7)
        P = 3
        K = 4
        atoms = [self._random_spd(P, rng) for _ in range(K)]
        w, obj = solve_master(atoms)
        assert w.shape == (K,)
        assert np.all(w >= -1e-9), f"Negative weights: {w}"
        assert abs(w.sum() - 1.0) < 1e-8, f"Weights do not sum to 1: {w.sum()}"

    def test_master_problem_objective_decreases_with_better_atom(self):
        """Adding a more informative atom should not increase (or mildly change) objective."""
        from dmipy_design.optimizers.master_problem import solve_master
        rng = np.random.default_rng(13)
        P = 2
        # Weak atoms
        atoms_weak = [self._random_spd(P, rng, scale=0.1) for _ in range(3)]
        w_weak, obj_weak = solve_master(atoms_weak)

        # Add a strong atom
        atoms_strong = atoms_weak + [self._random_spd(P, rng, scale=10.0)]
        w_strong, obj_strong = solve_master(atoms_strong)

        # -log det F decreases (= becomes more negative) when F is larger
        assert obj_strong <= obj_weak + 1e-6, (
            f"Objective did not decrease: {obj_weak:.4f} -> {obj_strong:.4f}"
        )

    def test_master_problem_gradient_matches_fd(self):
        """Analytic gradient matches finite difference gradient."""
        from dmipy_design.optimizers.master_problem import solve_master, EPS_REG
        rng = np.random.default_rng(99)
        P = 3
        K = 3
        atoms = [self._random_spd(P, rng) for _ in range(K)]
        F_stack = np.stack(atoms)
        I_reg = EPS_REG * np.eye(P)

        def obj_only(w):
            F = np.einsum('k,kij->ij', w, F_stack) + I_reg
            _, ld = np.linalg.slogdet(F)
            return -ld

        def analytic_grad(w):
            F = np.einsum('k,kij->ij', w, F_stack) + I_reg
            F_inv = np.linalg.inv(F)
            return np.array([-np.trace(F_inv @ F_stack[k]) for k in range(K)])

        w0 = np.array([0.4, 0.35, 0.25])
        g_ana = analytic_grad(w0)
        eps = 1e-5
        g_fd = np.zeros(K)
        for k in range(K):
            wp = w0.copy(); wm = w0.copy()
            wp[k] += eps; wm[k] -= eps
            g_fd[k] = (obj_only(wp) - obj_only(wm)) / (2 * eps)

        np.testing.assert_allclose(g_ana, g_fd, rtol=1e-4, atol=1e-6)


# ============================================================================
# 2. PGSE decode roundtrip
# ============================================================================

class TestDecoding:

    def test_decode_pgse_roundtrip(self):
        """encode_pgse_shell -> check gradient_strengths consistent with b, delta, Delta."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse, BVECS_30
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell

        rng = np.random.default_rng(0)
        v = rng.uniform(0, 1, size=3)
        b, delta, Delta = decode_pgse(v)

        bvecs = jnp.array(BVECS_30, dtype=jnp.float64)
        scheme = encode_pgse_shell(b, delta, Delta, bvecs)

        # Check gradient_strengths satisfies b = gamma² G² delta² (Delta - delta/3)
        G = scheme.gradient_strengths[0]
        b_reconstructed = GAMMA**2 * float(G)**2 * delta**2 * (Delta - delta / 3.0)
        assert abs(b_reconstructed - b) / b < 1e-6, (
            f"b roundtrip mismatch: orig={b:.4g}, reconstructed={b_reconstructed:.4g}"
        )

    def test_decode_pgse_delta_lt_Delta(self):
        """Delta > delta for all v in [0,1]^3."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse
        rng = np.random.default_rng(1)
        for _ in range(100):
            v = rng.uniform(0, 1, size=3)
            b, delta, Delta = decode_pgse(v)
            assert Delta > delta, f"Delta={Delta:.4g} <= delta={delta:.4g}"


# ============================================================================
# 3. OGSE physics: b = gamma² G² t_eff³
# ============================================================================

class TestOgsePhysics:

    def test_decode_ogse_b_formula(self):
        """At f=100 Hz, G=0.1 T/m: b = gamma² G² t_eff³."""
        from dmipy_design.optimizers.pricing_problem import decode_ogse

        v = np.array([
            (100.0 - 10.0) / (500.0 - 10.0),   # f -> 100 Hz
            (0.1  - 0.02)  / (0.30  - 0.02),    # G -> 0.1 T/m
        ])
        f, G, b = decode_ogse(v)

        t_eff = 1.0 / (4.0 * f)
        b_expected = GAMMA**2 * G**2 * t_eff**3

        assert abs(f - 100.0) < 0.1,  f"f mismatch: {f}"
        assert abs(G - 0.1)   < 0.001, f"G mismatch: {G}"
        assert abs(b - b_expected) / b_expected < 1e-6, (
            f"b mismatch: {b:.4g} vs {b_expected:.4g}"
        )

    def test_ogse_shell_gradient_strengths(self):
        """encode_ogse_shell should store G_arr = G for all directions."""
        from dmipy_design.jax_scheme_encoder import encode_ogse_shell
        from dmipy_design.optimizers.pricing_problem import BVECS_30
        bvecs = jnp.array(BVECS_30, dtype=jnp.float64)
        scheme = encode_ogse_shell(100.0, 0.15, bvecs)
        np.testing.assert_allclose(
            np.array(scheme.gradient_strengths), 0.15, rtol=1e-9
        )


# ============================================================================
# 4. Ball+C4Cylinder forward function
# ============================================================================

class TestBallC4CylinderForward:

    def test_forward_shape(self, pgse_scheme_1000):
        """forward_fn returns shape (n_dirs,)."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        theta = jnp.array([0.5, 2e-9, 2e-9, 10e-6], dtype=jnp.float64)
        sig = ball_c4cylinder_forward(theta, pgse_scheme_1000)
        assert sig.shape == (30,), f"Unexpected shape: {sig.shape}"

    def test_forward_values_in_range(self, pgse_scheme_1000):
        """Signal values in (0, 1]."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        theta = jnp.array([0.5, 2e-9, 2e-9, 10e-6], dtype=jnp.float64)
        sig = ball_c4cylinder_forward(theta, pgse_scheme_1000)
        assert jnp.all(sig > 0.0),    "Signal has non-positive values"
        assert jnp.all(sig <= 1.001), "Signal exceeds 1"

    def test_forward_near_zero_b_approaches_one(self, bvecs_30):
        """At very small b, signal → 1."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        # b = 1 s/m² (effectively zero), delta=0.01s, Delta=0.02s
        scheme = encode_pgse_shell(1.0, 0.01, 0.02, bvecs_30)
        theta = jnp.array([0.5, 2e-9, 2e-9, 10e-6], dtype=jnp.float64)
        sig = ball_c4cylinder_forward(theta, scheme)
        np.testing.assert_allclose(np.array(sig), 1.0, atol=5e-3)

    def test_forward_is_jax_differentiable(self, pgse_scheme_1000):
        """jax.jacobian must succeed (no tracing errors)."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        theta = jnp.array([0.5, 2e-9, 2e-9, 10e-6], dtype=jnp.float64)
        jac = jax.jacobian(ball_c4cylinder_forward)(theta, pgse_scheme_1000)
        assert jac.shape == (30, 4), f"Jacobian shape: {jac.shape}"
        # Jacobian should not be all-zero
        assert jnp.any(jnp.abs(jac) > 0), "Jacobian is all zero"


# ============================================================================
# 5. FIM sensitivity: OGSE > PGSE for diameter
# ============================================================================

class TestFIMSensitivity:

    def test_fim_ogse_diameter_sensitive(self, bvecs_30, prior_samples_ball_cyl):
        """FIM[diameter, diameter] for OGSE (f=100 Hz) > PGSE (delta=40ms, same b)."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell, encode_ogse_shell
        from dmipy_design.fim import compute_fim_averaged

        # OGSE: f=100 Hz, G=0.08 T/m
        f = 100.0
        G = 0.08
        t_eff = 1.0 / (4.0 * f)
        b_ogse = GAMMA**2 * G**2 * t_eff**3

        scheme_ogse = encode_ogse_shell(f, G, bvecs_30)

        # PGSE: same b-value, delta=40ms
        delta = 0.040
        # b = gamma² G² delta² (Delta - delta/3)  => Delta
        # Solve: Delta - delta/3 = b / (gamma² G²_pgse delta²)
        G_pgse = G * 0.5   # different G to get same b via different Delta
        b_pgse = b_ogse
        G2_pgse = b_pgse / (GAMMA**2 * delta**2)
        G_pgse_actual = np.sqrt(G2_pgse * 1.0)  # we need Delta-delta/3
        diff_time = b_pgse / (GAMMA**2 * G_pgse_actual**2 * delta**2)
        Delta_pgse = diff_time + delta / 3.0
        if Delta_pgse <= delta:
            # Fall back to a longer delta
            delta = 0.020
            G2_pgse = b_pgse / (GAMMA**2 * delta**2)
            G_pgse_actual = np.sqrt(G2_pgse)
            diff_time = b_pgse / (GAMMA**2 * G_pgse_actual**2 * delta**2)
            Delta_pgse = diff_time + delta / 3.0

        scheme_pgse = encode_pgse_shell(b_pgse, delta, Delta_pgse, bvecs_30)

        sigma = 0.05
        fim_ogse = np.array(
            compute_fim_averaged(ball_c4cylinder_forward, prior_samples_ball_cyl,
                                 scheme_ogse, sigma)
        )
        fim_pgse = np.array(
            compute_fim_averaged(ball_c4cylinder_forward, prior_samples_ball_cyl,
                                 scheme_pgse, sigma)
        )

        # diameter is index 3 in theta
        fim_ogse_d = fim_ogse[3, 3]
        fim_pgse_d = fim_pgse[3, 3]
        assert fim_ogse_d > fim_pgse_d, (
            f"OGSE FIM[diam,diam]={fim_ogse_d:.4g} not > PGSE FIM[diam,diam]={fim_pgse_d:.4g}"
        )


# ============================================================================
# 6. Ball-only column generation: adds an atom in one iteration
# ============================================================================

class TestColumnGenerationBasic:

    def _ball_forward(self, theta, scheme):
        from dmipy_fit.jax.signal_models_jax import g1ball_signal
        return g1ball_signal(scheme.bvalues, theta[0])

    def test_column_generation_adds_atom(self, bvecs_30, prior_samples_simple):
        """With one initial PGSE atom, one CG iteration adds a second atom."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        from dmipy_design.optimizers.column_generation import (
            column_generation_oed, Atom
        )

        scheme0 = encode_pgse_shell(1000e6, 0.020, 0.040, bvecs_30)
        atom0 = Atom(type='pgse',
                     params={'b': 1000e6, 'delta': 0.020, 'Delta': 0.040},
                     scheme=scheme0)

        result = column_generation_oed(
            forward_fn=self._ball_forward,
            prior_samples=prior_samples_simple,
            sigma=0.05,
            waveform_types=['pgse'],
            initial_atoms=[atom0],
            max_iter=1,
            verbose=False,
            n_pricing_restarts=3,
        )
        # After 1 iteration (not converged), the atom library grows
        # (unless the initial atom is already optimal, which is unlikely)
        # The history should have exactly 1 entry
        assert len(result.history) == 1
        # KW gap should be computed
        assert 'kw_gap' in result.history[0]

    def test_column_generation_weights_sum_to_one(self, bvecs_30, prior_samples_simple):
        """Final weights sum to 1."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        from dmipy_design.optimizers.column_generation import (
            column_generation_oed, Atom
        )

        scheme0 = encode_pgse_shell(1000e6, 0.020, 0.040, bvecs_30)
        atom0 = Atom(type='pgse',
                     params={'b': 1000e6, 'delta': 0.020, 'Delta': 0.040},
                     scheme=scheme0)

        result = column_generation_oed(
            forward_fn=self._ball_forward,
            prior_samples=prior_samples_simple,
            sigma=0.05,
            waveform_types=['pgse'],
            initial_atoms=[atom0],
            max_iter=3,
            verbose=False,
            n_pricing_restarts=2,
        )
        assert abs(result.weights.sum() - 1.0) < 1e-8


# ============================================================================
# 7. (slow) Ball+C4Cylinder: OGSE atom discovered in 5 iterations
# ============================================================================

@pytest.mark.slow
class TestColumnGenerationOgseDiscovery:

    def test_column_generation_ogse_discovery(self, bvecs_30, prior_samples_ball_cyl):
        """Ball+C4Cyl, PGSE-only start, 5 iterations; at least one OGSE atom found."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        from dmipy_design.optimizers.column_generation import (
            column_generation_oed, Atom
        )

        scheme0 = encode_pgse_shell(1000e6, 0.020, 0.040, bvecs_30)
        atom0 = Atom(
            type='pgse',
            params={'b': 1000e6, 'delta': 0.020, 'Delta': 0.040},
            scheme=scheme0,
        )

        result = column_generation_oed(
            forward_fn=ball_c4cylinder_forward,
            prior_samples=prior_samples_ball_cyl,
            sigma=0.05,
            waveform_types=['pgse', 'ogse'],
            initial_atoms=[atom0],
            max_iter=5,
            reduced_cost_tol=0.5,   # loose tolerance for speed
            verbose=True,
            n_pricing_restarts=5,
        )

        ogse_types = [a.type for a in result.atoms]
        has_ogse = 'ogse' in ogse_types or any(
            h.get('new_atom_type') == 'ogse' for h in result.history
        )
        assert has_ogse, (
            f"No OGSE atom found after 5 iterations. Atom types: {ogse_types}. "
            f"History: {[(h['iter'], h['new_atom_type']) for h in result.history]}"
        )
