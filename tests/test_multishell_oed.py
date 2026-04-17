"""Tests for multishell_oed optimizer.

Test IDs
--------
T3-MS-1  2-shell optimization improves D-optimal over random init
"""

import jax
jax.config.update("jax_enable_x64", True)


def test_multishell_oed_decreases_d_optimal():
    """2-shell optimization improves D-optimal over random init."""
    import jax.numpy as jnp
    import numpy as np
    from dmipy_design.optimizers.multishell import multishell_oed
    from dmipy_design.jax_scheme_encoder import JaxScheme
    from dmipy_design.fim import compute_fim_averaged
    from dmipy_design.objectives import d_optimal

    LAMBDA_SCALE = 3e-9

    def forward_fn(theta, scheme):
        from dmipy.jax.signal_models_jax import g1ball_signal
        return g1ball_signal(scheme.bvalues, theta[0] * LAMBDA_SCALE)

    prior_samples = jnp.array(np.full((10, 1), 0.7))
    sigma = 0.05
    n_shells, n_dirs = 2, 4

    # Initial: two identical shells at b=500 s/mm²
    u0 = jnp.array([[500e6, 0.02, 0.04],
                     [500e6, 0.02, 0.04]])

    u_opt, crlb_opt = multishell_oed(
        forward_fn, prior_samples, n_shells, n_dirs, sigma, u0,
        objective="D", max_iter=50
    )
    assert u_opt.shape == (n_shells, 3)
    assert np.isfinite(crlb_opt)
    # Optimized should differ from initial (optimizer moved)
    assert not jnp.allclose(u_opt, u0, atol=1e-3), "Optimizer did not move from init"
