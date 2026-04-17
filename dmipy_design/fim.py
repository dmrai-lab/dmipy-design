"""
Fisher Information Matrix computation for diffusion MRI acquisition design.

The FIM for Gaussian noise is:
    F_ij(theta, scheme) = (1/sigma²) * sum_k  [dE_k/dtheta_i] * [dE_k/dtheta_j]

where E_k is the signal at measurement k and theta are tissue parameters.
Optimal design minimises scalar objectives of inv(F) (the CRLB matrix).

Multi-encoding support (Phase 2)
----------------------------------
``compute_fim`` and ``compute_fim_averaged`` accept both ``JaxScheme`` objects
(PGSE / LTE) and the dict-based scheme representations returned by
``encode_ste()`` and ``encode_ogse()``.  Dispatch is performed by inspecting
the ``encoding`` key of dict schemes:

    - No dict / ``JaxScheme``  → PGSE path (original behaviour)
    - ``encoding == "STE"``    → forward_fn receives a JaxScheme with
                                 ``bvalues`` drawn from the isotropic b-tensor
                                 trace and ``bvecs`` set to isotropic proxies.
    - ``encoding == "OGSE"``   → forward_fn receives a JaxScheme-like object
                                 where ``delta`` and ``Delta`` are derived from
                                 ``t_eff``.

The forward_fn signature is always ``(theta, scheme) -> jnp.ndarray shape (N,)``.
For STE/OGSE the scheme argument passed to forward_fn is a ``JaxScheme``
constructed from the dict fields so that existing forward functions remain
compatible.
"""

import jax
import jax.numpy as jnp
from .jax_scheme_encoder import JaxScheme


def _scheme_to_jaxscheme(scheme) -> JaxScheme:
    """Convert a dict-based scheme (STE / OGSE) to a ``JaxScheme`` for forward models.

    Parameters
    ----------
    scheme : JaxScheme or dict
        If a ``JaxScheme``, returned unchanged.
        If a dict with ``encoding == "STE"``, builds a JaxScheme with
        ``bvalues`` from the isotropic trace and ``bvecs`` from the dict.
        If a dict with ``encoding == "OGSE"``, builds a JaxScheme where
        ``delta = t_eff`` and ``Delta = t_eff`` (proxy for t_eff-based models).

    Returns
    -------
    JaxScheme
    """
    if isinstance(scheme, JaxScheme):
        return scheme

    if not isinstance(scheme, dict):
        raise TypeError(
            f"scheme must be a JaxScheme or dict, got {type(scheme)}"
        )

    encoding = scheme.get("encoding", "PGSE")

    if encoding == "STE":
        # STE: isotropic b-tensor — signal depends only on b-value (trace).
        # Pass b_values directly; bvecs are formally irrelevant but kept for shape.
        return JaxScheme(
            bvalues=scheme["b_values"],
            bvecs=scheme["bvecs"],
            delta=scheme["delta"],
            Delta=scheme["Delta"],
        )

    if encoding == "OGSE":
        # OGSE: effective diffusion time t_eff replaces Delta - delta/3.
        # Encode as a JaxScheme where delta = Delta = t_eff so that any
        # model that uses Delta - delta/3 as effective time will receive t_eff.
        N = scheme["b_values"].shape[0]
        t_eff = scheme["t_eff"]
        return JaxScheme(
            bvalues=scheme["b_values"],
            bvecs=scheme["bvecs"],
            delta=jnp.broadcast_to(t_eff, (N,)),
            Delta=jnp.broadcast_to(t_eff + t_eff / 3.0, (N,)),
        )

    # Fallback: try to construct a JaxScheme from known keys
    return JaxScheme(
        bvalues=scheme["b_values"],
        bvecs=scheme["bvecs"],
        delta=scheme.get("delta", jnp.zeros(scheme["b_values"].shape[0])),
        Delta=scheme.get("Delta", jnp.zeros(scheme["b_values"].shape[0])),
    )


def compute_fim(
    forward_fn,
    theta: jnp.ndarray,
    scheme,
    sigma: float,
) -> jnp.ndarray:
    """Compute the Fisher Information Matrix at a single parameter point.

    Parameters
    ----------
    forward_fn : callable (theta, scheme) -> jnp.ndarray shape (N,)
        JAX-differentiable signal model (from ``dmipy.jax.multicompartment_jax``
        ``build_mc_forward_fn``).
    theta : jnp.ndarray, shape (P,)
        Tissue parameters (scaled to [0, 1] by dmipy-core convention).
    scheme : JaxScheme or dict
        JAX-traceable acquisition scheme.  Dict-based schemes (returned by
        ``encode_ste`` / ``encode_ogse``) are automatically converted to
        a ``JaxScheme`` before being passed to ``forward_fn``.
    sigma : float
        Noise standard deviation (assumed uniform).

    Returns
    -------
    FIM : jnp.ndarray, shape (P, P)
        Fisher information matrix.

    Notes
    -----
    Uses ``jax.jacobian`` for exact autodiff.  For models without a JAX
    forward path, use ``compute_fim_fd`` (finite differences) instead.
    """
    jax_scheme = _scheme_to_jaxscheme(scheme)
    jac = jax.jacobian(forward_fn)(theta, jax_scheme)   # (N, P)
    return jac.T @ jac / (sigma ** 2)


def compute_fim_averaged(
    forward_fn,
    prior_samples: jnp.ndarray,
    scheme,
    sigma: float,
) -> jnp.ndarray:
    """FIM averaged over a prior distribution of tissue parameters.

    Parameters
    ----------
    forward_fn : callable (theta, scheme) -> jnp.ndarray shape (N,)
    prior_samples : jnp.ndarray, shape (M, P)
        M samples from the parameter prior.
    scheme : JaxScheme or dict
        Dict-based schemes (STE / OGSE) are automatically converted to a
        ``JaxScheme`` before being passed to ``forward_fn``.
    sigma : float

    Returns
    -------
    FIM_avg : jnp.ndarray, shape (P, P)

    Notes
    -----
    Implements: F_avg(scheme) = (1/M) * sum_m F(theta_m, scheme)
    Vmapped over the prior samples for efficiency.
    """
    # Convert once (outside vmap) so the conversion is not traced repeatedly.
    jax_scheme = _scheme_to_jaxscheme(scheme)
    fim_per_sample = jax.vmap(
        lambda theta: compute_fim(forward_fn, theta, jax_scheme, sigma)
    )(prior_samples)
    return fim_per_sample.mean(axis=0)


def compute_fim_fd(
    forward_fn,
    theta: jnp.ndarray,
    scheme,
    sigma: float,
    eps: float = 1e-7,
) -> jnp.ndarray:
    """FIM via finite differences (fallback for non-differentiable models).

    Parameters
    ----------
    forward_fn : callable (theta, scheme) -> ndarray shape (N,)
        May be a numpy function (will be called as-is).
    theta : array, shape (P,)
    scheme : JaxScheme or dict
        Dict-based schemes (STE / OGSE) are converted to ``JaxScheme`` first.
    sigma : float
    eps : float   perturbation step

    Returns
    -------
    FIM : jnp.ndarray, shape (P, P)
    """
    import numpy as np
    jax_scheme = _scheme_to_jaxscheme(scheme)
    theta_np = np.array(theta)
    E0 = np.array(forward_fn(theta_np, jax_scheme))
    P = len(theta_np)
    N = len(E0)
    jac = np.zeros((N, P))
    for i in range(P):
        t_plus = theta_np.copy()
        t_plus[i] += eps
        E_plus = np.array(forward_fn(t_plus, jax_scheme))
        jac[:, i] = (E_plus - E0) / eps
    FIM = jac.T @ jac / (sigma ** 2)
    return jnp.array(FIM)
