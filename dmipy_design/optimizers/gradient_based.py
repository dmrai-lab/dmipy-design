"""
Gradient-based optimal experiment design via JAX autodiff + jax.scipy.optimize.
"""

import jax
import jax.numpy as jnp
import jax.scipy.optimize  # noqa: F401 – ensures submodule is importable
import numpy as np
import scipy.optimize


def gradient_oed(
    forward_fn,
    prior_samples: jnp.ndarray,
    bvecs: jnp.ndarray,
    sigma: float,
    u0: jnp.ndarray,
    constraints,
    objective: str = "D",
    max_iter: int = 200,
    learning_rate: float = 1e-2,
) -> tuple[jnp.ndarray, float]:
    """Optimise a PGSE protocol vector via jax.scipy.optimize.minimize (L-BFGS-B) on the CRLB.

    Parameters
    ----------
    forward_fn : callable (theta, scheme) -> jnp.ndarray shape (N,)
        JAX-differentiable forward model.
    prior_samples : jnp.ndarray, shape (M, P)
        Prior parameter samples for FIM averaging.
    bvecs : jnp.ndarray, shape (N, 3)
        Fixed gradient directions.
    sigma : float
        Noise standard deviation.
    u0 : jnp.ndarray, shape (3,)
        Initial protocol vector [b_value, delta, Delta].
    constraints : HardwareConstraints
        Used to determine feasible box bounds.
    objective : {'D', 'A', 'E'}
        CRLB objective to minimise.
    max_iter : int
        Maximum LBFGSB iterations.
    learning_rate : float
        Retained for backwards compatibility; not used by LBFGSB (which uses
        its own line search).

    Returns
    -------
    u_opt : jnp.ndarray, shape (3,)
        Optimised protocol vector within hardware bounds.
    crlb_opt : float
        Objective value at optimum.

    Notes
    -----
    Uses jax.scipy.optimize.minimize (L-BFGS-B) with explicit box constraints:
        b    ∈ [100e6, 10000e6]  s/m²
        delta ∈ [0.001, 0.060]   s
        Delta ∈ [0.005, 0.100]   s

    The ``learning_rate`` parameter is accepted but unused; L-BFGS-B performs
    its own Zoom/Armijo line search.
    """
    from ..fim import compute_fim_averaged
    from ..objectives import a_optimal, d_optimal, e_optimal
    from ..jax_scheme_encoder import encode_pgse

    obj_fn = {"D": d_optimal, "A": a_optimal, "E": e_optimal}[objective]

    # Hardware-feasible bounds:
    # u = [b_value (s/m²), delta (s), Delta (s)]
    lower_phys = jnp.array([100e6,  0.001, 0.005])
    upper_phys = jnp.array([10000e6, 0.060, 0.100])

    # Variable scaling: map u to v ∈ [0, 1]^3 so that LBFGSB sees O(1) gradients.
    #   v = (u - lower_phys) / (upper_phys - lower_phys)
    #   u = lower_phys + v * (upper_phys - lower_phys)
    scale = upper_phys - lower_phys

    def loss_scaled(v):
        u = lower_phys + v * scale
        scheme = encode_pgse(u, bvecs)
        FIM = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
        return obj_fn(FIM)

    # Bounds in normalised space are [0, 1]^3
    lower_v = jnp.zeros(3)
    upper_v = jnp.ones(3)
    v0 = (u0 - lower_phys) / scale
    v0 = jnp.clip(v0, lower_v, upper_v)

    loss_and_grad = jax.value_and_grad(loss_scaled)

    def loss_and_grad_np(v_np):
        v = jnp.array(v_np)
        val, grad = loss_and_grad(v)
        return float(val), np.array(grad)

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    result = scipy.optimize.minimize(
        loss_and_grad_np, np.array(v0), method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'gtol': 1e-6},
    )
    v_opt = jnp.array(result.x)
    u_opt = lower_phys + v_opt * scale
    return u_opt, float(loss_scaled(v_opt))
