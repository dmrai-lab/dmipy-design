"""
Multi-shell optimal experiment design.

Jointly optimises n_shells PGSE shell vectors u_k = [b_k, delta_k, Delta_k]
to minimise the CRLB objective averaged over the prior.
"""
import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize


def multishell_oed(
    forward_fn,
    prior_samples: jnp.ndarray,
    n_shells: int,
    n_dirs_per_shell: int,
    sigma: float,
    u0: jnp.ndarray,           # shape (n_shells, 3) — initial shell vectors
    objective: str = "D",
    max_iter: int = 300,
) -> tuple[jnp.ndarray, float]:
    """Optimise n_shells PGSE shell vectors jointly.

    Parameters
    ----------
    forward_fn : callable (theta, scheme) -> jnp.ndarray shape (N,)
        where N = n_shells * n_dirs_per_shell
    prior_samples : jnp.ndarray, shape (M, P)
    n_shells : int
    n_dirs_per_shell : int
        Number of directions per shell (fixed, isotropically distributed).
    sigma : float
    u0 : jnp.ndarray, shape (n_shells, 3)
        Initial [b_value, delta, Delta] for each shell.
    objective : {'D', 'A', 'E'}
    max_iter : int

    Returns
    -------
    u_opt : jnp.ndarray, shape (n_shells, 3)
    crlb_opt : float
    """
    from ..fim import compute_fim_averaged
    from ..objectives import a_optimal, d_optimal, e_optimal
    from ..jax_scheme_encoder import JaxScheme

    obj_fn = {"D": d_optimal, "A": a_optimal, "E": e_optimal}[objective]

    # Fixed isotropic directions for all shells (same bvecs reused)
    # Simple: use a fixed set of unit vectors (golden-angle or random)
    rng = np.random.default_rng(42)
    bvecs_np = rng.standard_normal((n_dirs_per_shell, 3))
    bvecs_np /= np.linalg.norm(bvecs_np, axis=1, keepdims=True)
    bvecs_fixed = jnp.array(np.tile(bvecs_np, (n_shells, 1)))  # (N_total, 3)

    # Physical bounds per shell
    lower_phys = jnp.array([100e6, 0.001, 0.005])
    upper_phys = jnp.array([10000e6, 0.060, 0.100])
    scale = upper_phys - lower_phys

    # Flatten n_shells shell vectors into a single vector for the optimizer
    # v shape: (n_shells * 3,) in [0,1]
    def u_to_v(u):
        return ((u - lower_phys[None, :]) / scale[None, :]).reshape(-1)

    def v_to_u(v):
        return (lower_phys[None, :] + v.reshape(n_shells, 3) * scale[None, :])

    def build_scheme(u):
        # u: (n_shells, 3)
        bvalues = jnp.repeat(u[:, 0], n_dirs_per_shell)  # (N_total,)
        deltas = jnp.repeat(u[:, 1], n_dirs_per_shell)
        Deltas = jnp.repeat(u[:, 2], n_dirs_per_shell)
        return JaxScheme(bvalues=bvalues, bvecs=bvecs_fixed, delta=deltas, Delta=Deltas)

    def loss(v):
        u = v_to_u(v)
        scheme = build_scheme(u)
        FIM = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
        return obj_fn(FIM)

    v0 = u_to_v(u0)
    v0 = jnp.clip(v0, 0.0, 1.0)
    bounds = [(0.0, 1.0)] * (n_shells * 3)

    loss_and_grad = jax.value_and_grad(loss)

    def loss_and_grad_np(v_np):
        v = jnp.array(v_np)
        val, grad = loss_and_grad(v)
        return float(val), np.array(grad)

    result = scipy.optimize.minimize(
        loss_and_grad_np, np.array(v0), method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'gtol': 1e-5},
    )
    v_opt = jnp.array(result.x)
    u_opt = v_to_u(v_opt)
    return u_opt, float(loss(v_opt))
