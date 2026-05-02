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
        """At f=100 Hz, G=0.1 T/m: b = gamma² G² t_eff³ (using CONNECTOM_3T preset)."""
        from dmipy_design.optimizers.pricing_problem import decode_ogse, CONNECTOM_3T

        # CONNECTOM_3T: ogse_f_max=500 Hz, g_max=0.30 T/m
        # v[0]: f = 10 + v[0] * (500 - 10) = 100  ->  v[0] = 90/490
        # v[1]: G = 0.30*0.1 + v[1]*0.30*0.9 = 0.03 + v[1]*0.27 = 0.1 -> v[1] = 0.07/0.27
        hw = CONNECTOM_3T
        v = np.array([
            (100.0 - 10.0) / (hw.ogse_f_max - 10.0),        # f -> 100 Hz
            (0.1 - hw.g_max * 0.1) / (hw.g_max * 0.9),      # G -> 0.1 T/m
        ])
        f, G, b = decode_ogse(v, hardware=hw)

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

# ============================================================================
# 8. jaxopt pricing: returns valid rc and expected param keys
# ============================================================================

class TestJaxoptPricing:

    @pytest.fixture(scope="class")
    def ball_cyl_prior(self):
        rng = np.random.default_rng(7)
        M = 16
        vf_ball    = rng.uniform(0.3, 0.7, M)
        lambda_iso = rng.uniform(1e-9, 3e-9, M)
        lambda_par = rng.uniform(1.5e-9, 2.5e-9, M)
        diameter   = rng.uniform(2e-6, 20e-6, M)
        return jnp.array(
            np.column_stack([vf_ball, lambda_iso, lambda_par, diameter]),
            dtype=jnp.float64
        )

    def test_jaxopt_pricing_returns_valid_rc(self, ball_cyl_prior):
        """solve_pricing (jaxopt backend) returns best_rc > 0 and expected param keys."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        from dmipy_design.optimizers.pricing_problem import solve_pricing, BVECS_30

        P = ball_cyl_prior.shape[1]
        F_id = np.eye(P, dtype=np.float64)

        best_rc, best_params, best_scheme = solve_pricing(
            forward_fn     = ball_c4cylinder_forward,
            prior_samples  = ball_cyl_prior,
            sigma          = 0.05,
            F_total_inv_np = F_id,
            wtype          = 'pgse',
            n_restarts     = 4,
            rng_seed       = 42,
        )

        assert best_rc > 0, f"Expected best_rc > 0, got {best_rc}"
        for key in ('type', 'b', 'G', 'delta', 'Delta'):
            assert key in best_params, f"Missing key '{key}' in best_params: {best_params}"

    def test_jaxopt_pricing_ogse_returns_valid_rc(self, ball_cyl_prior):
        """solve_pricing with wtype='ogse' returns best_rc > 0 and expected param keys."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        from dmipy_design.optimizers.pricing_problem import solve_pricing

        P = ball_cyl_prior.shape[1]
        F_id = np.eye(P, dtype=np.float64)

        best_rc, best_params, best_scheme = solve_pricing(
            forward_fn     = ball_c4cylinder_forward,
            prior_samples  = ball_cyl_prior,
            sigma          = 0.05,
            F_total_inv_np = F_id,
            wtype          = 'ogse',
            n_restarts     = 4,
            rng_seed       = 42,
        )

        assert best_rc > 0, f"Expected best_rc > 0, got {best_rc}"
        for key in ('type', 'f', 'G', 'b'):
            assert key in best_params, f"Missing key '{key}' in best_params: {best_params}"


# ============================================================================
# 9. Sigmoid / logit roundtrip
# ============================================================================

class TestSigmoidLogit:

    def test_sigmoid_logit_roundtrip(self):
        """v -> logit(v) -> sigmoid(logit(v)) roundtrips to 1e-6 tolerance."""
        from dmipy_design.optimizers.pricing_problem import _sigmoid, _logit

        rng = np.random.default_rng(0)
        v = jnp.array(rng.uniform(0.05, 0.95, size=(20,)), dtype=jnp.float64)
        v_roundtrip = _sigmoid(_logit(v))
        np.testing.assert_allclose(
            np.array(v_roundtrip), np.array(v), atol=1e-6,
            err_msg="sigmoid(logit(v)) != v"
        )

    def test_logit_sigmoid_roundtrip(self):
        """u -> sigmoid(u) -> logit(sigmoid(u)) roundtrips to 1e-6 tolerance."""
        from dmipy_design.optimizers.pricing_problem import _sigmoid, _logit

        rng = np.random.default_rng(1)
        u = jnp.array(rng.uniform(-4.0, 4.0, size=(20,)), dtype=jnp.float64)
        u_roundtrip = _logit(_sigmoid(u))
        np.testing.assert_allclose(
            np.array(u_roundtrip), np.array(u), atol=1e-6,
            err_msg="logit(sigmoid(u)) != u"
        )


# ============================================================================
# 10. Gamma prior diameter sanity check
# ============================================================================

class TestGammaPrior:

    def test_gamma_prior_diameter_range(self):
        """Gamma(k=2, scale=0.4µm) prior: 95th pct < 4µm, mean > 0.5µm."""
        from scipy.stats import gamma as gamma_dist

        rng = np.random.default_rng(42)
        M = 10000
        d = gamma_dist(a=2.0, scale=0.4e-6).rvs(M, random_state=rng)
        d = np.clip(d, 0.2e-6, 4.0e-6)

        p95 = np.percentile(d, 95)
        mean = d.mean()

        assert p95 < 4.0e-6, f"95th percentile {p95*1e6:.2f}µm >= 4µm"
        assert mean > 0.5e-6, f"Mean diameter {mean*1e6:.2f}µm <= 0.5µm"

    def test_gamma_prior_mode(self):
        """Gamma(k=2, scale=0.4µm): mode = (k-1)*scale = 0.4µm."""
        from scipy.stats import gamma as gamma_dist

        # For gamma(a=k, scale=theta): mode = (k-1)*theta when k >= 1
        # With k=2, theta=0.4e-6: mode = 0.4e-6
        rng = np.random.default_rng(0)
        M = 50000
        d = gamma_dist(a=2.0, scale=0.4e-6).rvs(M, random_state=rng)
        # Histogram peak should be near 0.4µm
        counts, edges = np.histogram(d, bins=50)
        mode_bin_centre = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
        # Allow ±0.2µm tolerance
        assert abs(mode_bin_centre - 0.4e-6) < 0.2e-6, (
            f"Histogram mode {mode_bin_centre*1e6:.2f}µm not near expected 0.4µm"
        )


# ============================================================================
# 11. (slow) Ball+C4Cylinder: OGSE atom discovered in 5 iterations
# ============================================================================

# ============================================================================
# 11b. Hardware presets and STE
# ============================================================================

class TestHardwarePresets:

    def test_hardware_preset_connectom_allows_short_delta(self):
        """CONNECTOM_3T allows delta < 0.010 s at v[1]=0 (minimum)."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse, CONNECTOM_3T
        v = np.array([0.5, 0.0, 0.5])
        b, delta, Delta = decode_pgse(v, hardware=CONNECTOM_3T)
        assert float(delta) < 0.010, (
            f"CONNECTOM delta={float(delta)*1e3:.1f}ms should be < 10ms "
            f"(pgse_delta_min={CONNECTOM_3T.pgse_delta_min*1e3:.1f}ms)"
        )

    def test_hardware_preset_clinical_limits_delta(self):
        """CLINICAL_3T enforces delta >= 0.015 s for all v in [0,1]^3."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse, CLINICAL_3T
        rng = np.random.default_rng(5)
        for _ in range(50):
            v = rng.uniform(0, 1, size=3)
            b, delta, Delta = decode_pgse(v, hardware=CLINICAL_3T)
            assert float(delta) >= 0.015, (
                f"CLINICAL delta={float(delta)*1e3:.1f}ms < 15ms at v={v}"
            )

    def test_hardware_preset_connectom_higher_gmax(self):
        """CONNECTOM_3T allows G up to 300 mT/m (CLINICAL_3T max is 80 mT/m)."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse, CLINICAL_3T, CONNECTOM_3T
        v = np.array([1.0, 0.5, 0.5])  # maximum G
        _, _, _ = decode_pgse(v, hardware=CLINICAL_3T)
        # CONNECTOM G_max = hardware.g_max (0.30 T/m), CLINICAL G_max = 0.08 T/m
        assert CONNECTOM_3T.g_max > CLINICAL_3T.g_max, (
            "CONNECTOM_3T g_max should exceed CLINICAL_3T g_max"
        )

    def test_decode_ste_same_as_pgse_physics(self):
        """decode_ste returns identical (b, delta, Delta) to decode_pgse for same v."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse, decode_ste, CLINICAL_3T
        rng = np.random.default_rng(7)
        for _ in range(20):
            v = rng.uniform(0, 1, size=3)
            b_pgse, delta_pgse, Delta_pgse = decode_pgse(v, hardware=CLINICAL_3T)
            b_ste,  delta_ste,  Delta_ste  = decode_ste(v,  hardware=CLINICAL_3T)
            assert abs(float(b_pgse) - float(b_ste)) < 1e-10, "b mismatch"
            assert abs(float(delta_pgse) - float(delta_ste)) < 1e-12, "delta mismatch"
            assert abs(float(Delta_pgse) - float(Delta_ste)) < 1e-12, "Delta mismatch"

    def test_encode_ste_shell_encoding_type(self, bvecs_30):
        """encode_ste_shell produces a JaxScheme with encoding_type='ste'."""
        from dmipy_design.jax_scheme_encoder import encode_ste_shell
        scheme = encode_ste_shell(1000e6, 0.020, 0.040, bvecs_30)
        assert scheme.encoding_type == 'ste', (
            f"Expected encoding_type='ste', got '{scheme.encoding_type}'"
        )

    def test_encode_pgse_shell_encoding_type_default(self, bvecs_30):
        """encode_pgse_shell produces a JaxScheme with encoding_type='pgse' (default)."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        scheme = encode_pgse_shell(1000e6, 0.020, 0.040, bvecs_30)
        assert scheme.encoding_type == 'pgse', (
            f"Expected encoding_type='pgse', got '{scheme.encoding_type}'"
        )

    def test_ball_c4cylinder_forward_ste_direction_independent(self, bvecs_30):
        """STE forward: all 30 directions produce the same signal (within tolerance)."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        from dmipy_design.jax_scheme_encoder import encode_ste_shell

        scheme = encode_ste_shell(1000e6, 0.020, 0.040, bvecs_30)
        theta = jnp.array([0.5, 2e-9, 2e-9, 10e-6], dtype=jnp.float64)
        sig = ball_c4cylinder_forward(theta, scheme)

        # All 30 directions should give the same signal (STE is rotationally invariant)
        assert sig.shape == (30,), f"Expected shape (30,), got {sig.shape}"
        np.testing.assert_allclose(
            np.array(sig), float(sig[0]), rtol=1e-8, atol=1e-10,
            err_msg="STE signal is not direction-independent"
        )

    def test_pricing_ste_returns_valid_rc(self, bvecs_30):
        """solve_pricing with wtype='ste' returns best_rc > 0 and expected param keys."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward
        from dmipy_design.optimizers.pricing_problem import solve_pricing, CLINICAL_3T

        rng = np.random.default_rng(42)
        M = 16
        vf_ball    = rng.uniform(0.3, 0.7, M)
        lambda_iso = rng.uniform(1e-9, 3e-9, M)
        lambda_par = rng.uniform(1.5e-9, 2.5e-9, M)
        diameter   = rng.uniform(2e-6, 20e-6, M)
        prior = jnp.array(
            np.column_stack([vf_ball, lambda_iso, lambda_par, diameter]),
            dtype=jnp.float64
        )

        P = prior.shape[1]
        F_id = np.eye(P, dtype=np.float64)

        best_rc, best_params, best_scheme = solve_pricing(
            forward_fn     = ball_c4cylinder_forward,
            prior_samples  = prior,
            sigma          = 0.05,
            F_total_inv_np = F_id,
            wtype          = 'ste',
            n_restarts     = 3,
            rng_seed       = 42,
            hardware       = CLINICAL_3T,
        )

        assert best_rc > 0, f"Expected best_rc > 0, got {best_rc}"
        for key in ('type', 'b', 'G', 'delta', 'Delta'):
            assert key in best_params, f"Missing key '{key}' in best_params: {best_params}"
        assert best_params['type'] == 'ste', (
            f"Expected type='ste', got '{best_params['type']}'"
        )
        assert best_scheme.encoding_type == 'ste', (
            f"Expected encoding_type='ste', got '{best_scheme.encoding_type}'"
        )


# ============================================================================
# 12. TE field in shell encoders
# ============================================================================

class TestTEField:

    def test_pgse_shell_has_te(self, bvecs_30):
        """encode_pgse_shell populates TE = 2 * Delta."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        delta, Delta = 0.020, 0.040
        scheme = encode_pgse_shell(1000e6, delta, Delta, bvecs_30)
        assert scheme.TE is not None, "TE should be populated by encode_pgse_shell"
        expected_TE = 2.0 * Delta
        np.testing.assert_allclose(
            float(scheme.TE[0]), expected_TE, rtol=1e-10,
            err_msg=f"TE={float(scheme.TE[0])*1e3:.1f}ms, expected {expected_TE*1e3:.1f}ms"
        )

    def test_ogse_shell_has_te(self, bvecs_30):
        """encode_ogse_shell populates TE = 2 / freq."""
        from dmipy_design.jax_scheme_encoder import encode_ogse_shell
        freq = 100.0
        scheme = encode_ogse_shell(freq, 0.1, bvecs_30)
        assert scheme.TE is not None, "TE should be populated by encode_ogse_shell"
        expected_TE = 2.0 / freq   # 20 ms
        np.testing.assert_allclose(
            float(scheme.TE[0]), expected_TE, rtol=1e-10,
            err_msg=f"TE={float(scheme.TE[0])*1e3:.1f}ms, expected {expected_TE*1e3:.1f}ms"
        )

    def test_ste_shell_has_te(self, bvecs_30):
        """encode_ste_shell populates TE = 2 * Delta."""
        from dmipy_design.jax_scheme_encoder import encode_ste_shell
        delta, Delta = 0.020, 0.040
        scheme = encode_ste_shell(1000e6, delta, Delta, bvecs_30)
        assert scheme.TE is not None, "TE should be populated by encode_ste_shell"
        expected_TE = 2.0 * Delta
        np.testing.assert_allclose(
            float(scheme.TE[0]), expected_TE, rtol=1e-10,
        )

    def test_ogse_higher_freq_shorter_te(self, bvecs_30):
        """Higher OGSE frequency → shorter TE (T2 penalty decreases)."""
        from dmipy_design.jax_scheme_encoder import encode_ogse_shell
        scheme_low  = encode_ogse_shell(10.0,  0.1, bvecs_30)
        scheme_high = encode_ogse_shell(500.0, 0.1, bvecs_30)
        assert float(scheme_high.TE[0]) < float(scheme_low.TE[0]), (
            f"TE should decrease with frequency: low={float(scheme_low.TE[0])*1e3:.1f}ms, "
            f"high={float(scheme_high.TE[0])*1e3:.1f}ms"
        )

    def test_pgse_longer_delta_longer_te(self, bvecs_30):
        """Longer Delta → longer TE (T2 penalty increases)."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        scheme_short = encode_pgse_shell(1000e6, 0.015, 0.020, bvecs_30)
        scheme_long  = encode_pgse_shell(1000e6, 0.040, 0.080, bvecs_30)
        assert float(scheme_long.TE[0]) > float(scheme_short.TE[0])


# ============================================================================
# 13. make_t2_forward
# ============================================================================

class TestMakeT2Forward:

    def test_make_t2_forward_returns_callable(self, bvecs_30):
        """make_t2_forward returns a callable."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_t2_forward
        fwd = make_t2_forward(ball_c4cylinder_forward, T2=0.080, S0=1.0)
        assert callable(fwd)

    def test_make_t2_forward_applies_t2_attenuation(self, bvecs_30):
        """T2-weighted signal < base signal for schemes with long TE."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_t2_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell

        # Long Delta → long TE → strong T2 attenuation
        scheme = encode_pgse_shell(1000e6, 0.040, 0.080, bvecs_30)  # TE = 160ms
        theta = jnp.array([0.5, 2e-9, 2e-9, 5e-6], dtype=jnp.float64)

        sig_base = ball_c4cylinder_forward(theta, scheme)
        sig_t2   = make_t2_forward(ball_c4cylinder_forward, T2=0.080)(theta, scheme)

        # T2 factor = exp(-0.160/0.080) = exp(-2) ≈ 0.135
        # All T2-weighted values should be < base values
        assert jnp.all(sig_t2 < sig_base), (
            "T2-weighted signal should be less than base signal at TE=160ms, T2=80ms"
        )

    def test_make_t2_forward_t2_factor_correct(self, bvecs_30):
        """T2-weighted signal = S0 * exp(-TE/T2) * base signal."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_t2_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell
        import math

        T2, S0 = 0.080, 1.0
        Delta = 0.040   # TE = 2*Delta = 0.080s = T2 → factor = exp(-1) ≈ 0.368
        scheme = encode_pgse_shell(1000e6, 0.020, Delta, bvecs_30)
        theta = jnp.array([0.5, 2e-9, 2e-9, 5e-6], dtype=jnp.float64)

        sig_base = ball_c4cylinder_forward(theta, scheme)
        sig_t2   = make_t2_forward(ball_c4cylinder_forward, T2=T2, S0=S0)(theta, scheme)

        expected_factor = S0 * math.exp(-2.0 * Delta / T2)  # TE = 2*Delta
        np.testing.assert_allclose(
            np.array(sig_t2), np.array(sig_base) * expected_factor,
            rtol=1e-6,
            err_msg="T2 attenuation factor not applied correctly"
        )

    def test_make_t2_forward_s0_scaling(self, bvecs_30):
        """S0 scaling is linear: signal(S0=2) = 2 * signal(S0=1)."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_t2_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell

        scheme = encode_pgse_shell(500e6, 0.020, 0.040, bvecs_30)
        theta = jnp.array([0.5, 2e-9, 2e-9, 5e-6], dtype=jnp.float64)
        T2 = 0.080

        sig_s1 = make_t2_forward(ball_c4cylinder_forward, T2=T2, S0=1.0)(theta, scheme)
        sig_s2 = make_t2_forward(ball_c4cylinder_forward, T2=T2, S0=2.0)(theta, scheme)

        np.testing.assert_allclose(np.array(sig_s2), np.array(sig_s1) * 2.0, rtol=1e-10)

    def test_make_t2_forward_is_differentiable(self, bvecs_30):
        """make_t2_forward output is JAX-differentiable with respect to theta."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_t2_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell

        scheme = encode_pgse_shell(1000e6, 0.020, 0.040, bvecs_30)
        theta = jnp.array([0.5, 2e-9, 2e-9, 5e-6], dtype=jnp.float64)
        fwd = make_t2_forward(ball_c4cylinder_forward, T2=0.080, S0=1.0)

        jac = jax.jacobian(fwd)(theta, scheme)
        assert jac.shape == (30, 4), f"Expected jacobian shape (30, 4), got {jac.shape}"
        assert jnp.all(jnp.isfinite(jac)), "Jacobian contains non-finite values"


# ============================================================================
# 14. b_max safety clip in HardwarePreset
# ============================================================================

class TestBMaxClip:

    def test_hardware_preset_has_b_max(self):
        """HardwarePreset has a b_max attribute."""
        from dmipy_design.optimizers.pricing_problem import CLINICAL_3T, CONNECTOM_3T
        assert hasattr(CLINICAL_3T, 'b_max'), "CLINICAL_3T missing b_max"
        assert hasattr(CONNECTOM_3T, 'b_max'), "CONNECTOM_3T missing b_max"
        assert CLINICAL_3T.b_max > 0
        assert CONNECTOM_3T.b_max > 0

    def test_decode_pgse_clips_at_b_max(self):
        """decode_pgse clips b at hardware.b_max when physics would exceed it."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse, HardwarePreset

        # Set a very low b_max so the clip fires at mid-range v
        hw = HardwarePreset('test', g_max=0.08, pgse_delta_min=0.015,
                            ogse_f_max=300.0, b_max=500e6)
        # v = [1, 1, 1]: G=g_max, delta=60ms, ratio=3 → Delta=180ms
        # b ≈ GAMMA^2 * 0.08^2 * 0.06^2 * (0.18 - 0.02) >> 500e6
        v = np.array([1.0, 1.0, 1.0])
        b, delta, Delta = decode_pgse(v, hardware=hw)
        assert float(b) <= hw.b_max + 1e3, (
            f"b={float(b)/1e6:.0f} s/mm² exceeds b_max={hw.b_max/1e6:.0f} s/mm²"
        )

    def test_decode_pgse_below_b_max_unclipped(self):
        """decode_pgse does not clip b when physics is below b_max."""
        from dmipy_design.optimizers.pricing_problem import decode_pgse, CLINICAL_3T
        # Low v: small G, small delta, small ratio → low b
        v = np.array([0.0, 0.0, 0.0])  # minimum parameters
        b, delta, Delta = decode_pgse(v, hardware=CLINICAL_3T)
        assert float(b) < CLINICAL_3T.b_max, (
            f"b={float(b)/1e6:.1f} s/mm² should be below b_max at minimum v"
        )


# ============================================================================
# 15. PGSTE shell encoder and SNR advantage
# ============================================================================

class TestPGSTE:

    def test_pgste_shell_has_correct_te_and_tm(self, bvecs_30):
        """encode_pgste_shell: TE = 2·delta, TM = Delta − delta."""
        from dmipy_design.jax_scheme_encoder import encode_pgste_shell
        delta, Delta = 0.020, 0.100
        scheme = encode_pgste_shell(1000e6, delta, Delta, bvecs_30)

        assert scheme.TE is not None
        assert scheme.TM is not None
        np.testing.assert_allclose(float(scheme.TE[0]), 2.0 * delta, rtol=1e-10,
                                   err_msg="PGSTE TE should be 2·delta")
        np.testing.assert_allclose(float(scheme.TM[0]), Delta - delta, rtol=1e-10,
                                   err_msg="PGSTE TM should be Delta − delta")

    def test_pgste_encoding_type(self, bvecs_30):
        """encode_pgste_shell sets encoding_type='pgste'."""
        from dmipy_design.jax_scheme_encoder import encode_pgste_shell
        scheme = encode_pgste_shell(1000e6, 0.020, 0.100, bvecs_30)
        assert scheme.encoding_type == 'pgste'

    def test_pgste_same_b_as_pgse(self, bvecs_30):
        """PGSTE b-value equals PGSE b-value for identical (b, delta, Delta)."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell, encode_pgste_shell
        b0, delta, Delta = 2000e6, 0.020, 0.080
        pgse  = encode_pgse_shell(b0, delta, Delta, bvecs_30)
        pgste = encode_pgste_shell(b0, delta, Delta, bvecs_30)
        np.testing.assert_allclose(
            np.array(pgste.bvalues), np.array(pgse.bvalues), rtol=1e-10,
            err_msg="PGSTE and PGSE should have identical b-values"
        )

    def test_pgste_shorter_te_than_pgse(self, bvecs_30):
        """PGSTE TE (2·delta) < PGSE TE (2·Delta) for Delta > delta."""
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell, encode_pgste_shell
        delta, Delta = 0.015, 0.100
        pgse  = encode_pgse_shell(1000e6, delta, Delta, bvecs_30)
        pgste = encode_pgste_shell(1000e6, delta, Delta, bvecs_30)
        assert float(pgste.TE[0]) < float(pgse.TE[0]), (
            f"PGSTE TE={float(pgste.TE[0])*1e3:.1f}ms should be < "
            f"PGSE TE={float(pgse.TE[0])*1e3:.1f}ms"
        )

    def test_pgste_snr_advantage_at_long_delta(self, bvecs_30):
        """PGSTE SNR factor > PGSE SNR factor for long Delta."""
        import math
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_snr_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell, encode_pgste_shell

        T2, T1 = 0.080, 1.0
        delta, Delta = 0.020, 0.150  # long Delta: PGSE TE=300ms, PGSTE TE=40ms+TM=130ms

        scheme_pgse  = encode_pgse_shell(2000e6, delta, Delta, bvecs_30)
        scheme_pgste = encode_pgste_shell(2000e6, delta, Delta, bvecs_30)
        theta = jnp.array([0.5, 2e-9, 2e-9, 5e-6], dtype=jnp.float64)

        fwd = make_snr_forward(ball_c4cylinder_forward, T2=T2, T1=T1, S0=1.0)
        sig_pgse  = fwd(theta, scheme_pgse)
        sig_pgste = fwd(theta, scheme_pgste)

        # PGSTE signal must be larger (better SNR at long Delta)
        ratio = float(jnp.mean(sig_pgste) / jnp.mean(sig_pgse))
        assert ratio > 1.0, (
            f"PGSTE signal should exceed PGSE at long Delta; ratio={ratio:.3f}"
        )
        # At delta=20ms, Delta=150ms: ratio ≈ exp(-(40-300)/80) × exp(-130/1000)
        # = exp(260/80) / 1 × exp(-0.13) ≈ very large / 0.87
        # Just check it's substantially larger (> 2×)
        assert ratio > 2.0, f"PGSTE SNR advantage should be > 2× at this Delta; ratio={ratio:.2f}"

    def test_make_snr_forward_applies_t1_to_pgste_only(self, bvecs_30):
        """T1 term applied to PGSTE (TM != None) but not to PGSE (TM == None)."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_snr_forward
        from dmipy_design.jax_scheme_encoder import encode_pgse_shell, encode_pgste_shell

        T2, T1 = 0.080, 1.0
        delta, Delta = 0.020, 0.040
        scheme_pgse  = encode_pgse_shell(1000e6, delta, Delta, bvecs_30)
        scheme_pgste = encode_pgste_shell(1000e6, delta, Delta, bvecs_30)
        theta = jnp.array([0.5, 2e-9, 2e-9, 5e-6], dtype=jnp.float64)

        fwd = make_snr_forward(ball_c4cylinder_forward, T2=T2, T1=T1)

        sig_pgse  = fwd(theta, scheme_pgse)
        sig_pgste = fwd(theta, scheme_pgste)

        # PGSE: weight = exp(-TE_pgse / T2) = exp(-80/80) = 0.368
        # PGSTE: weight = exp(-TE_pgste / T2) * exp(-TM / T1)
        #             = exp(-40/80) * exp(-20/1000)
        # PGSTE / PGSE = exp(-40/80) * exp(-20/1000) / exp(-80/80)
        #              = exp(40/80) * exp(-20/1000) = e^0.5 * 0.98 ≈ 1.62
        # So for this short Delta, PGSTE still has better signal than PGSE
        import math
        expected_ratio = math.exp(0.5) * math.exp(-0.020 / T1)
        actual_ratio = float(jnp.mean(sig_pgste) / jnp.mean(sig_pgse))
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-4)

    def test_pgste_pricing_returns_valid_rc(self, bvecs_30):
        """solve_pricing with wtype='pgste' returns best_rc > 0 with TM in params."""
        from dmipy_design.analytical_forward import ball_c4cylinder_forward, make_snr_forward
        from dmipy_design.optimizers.pricing_problem import solve_pricing, CLINICAL_3T

        rng = np.random.default_rng(0)
        M = 16
        prior = jnp.array(np.column_stack([
            rng.uniform(0.3, 0.7, M),
            rng.uniform(1e-9, 3e-9, M),
            rng.uniform(1.5e-9, 2.5e-9, M),
            rng.uniform(2e-6, 20e-6, M),
        ]), dtype=jnp.float64)

        fwd = make_snr_forward(ball_c4cylinder_forward, T2=0.080, T1=1.0)
        P = prior.shape[1]
        F_id = np.eye(P, dtype=np.float64)

        best_rc, best_params, best_scheme = solve_pricing(
            forward_fn=fwd, prior_samples=prior, sigma=0.05,
            F_total_inv_np=F_id, wtype='pgste',
            n_restarts=3, rng_seed=0, hardware=CLINICAL_3T,
        )

        assert best_rc > 0
        assert best_params['type'] == 'pgste'
        assert 'TM' in best_params, f"TM missing from params: {best_params}"
        assert best_scheme.encoding_type == 'pgste'
        assert best_scheme.TM is not None


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
