"""
CRLB scalar objective functions for optimal experiment design.

All objectives take a FIM and return a scalar to minimise.
They are JAX-compatible and can be differentiated through.
"""

import jax.numpy as jnp


_EPS = jnp.float32(1e-10)


def a_optimal(FIM: jnp.ndarray) -> jnp.ndarray:
    """A-optimality: trace of CRLB = sum of parameter variances."""
    CRLB = jnp.linalg.inv(FIM + _EPS * jnp.eye(FIM.shape[0]))
    return jnp.trace(CRLB)


def d_optimal(FIM: jnp.ndarray) -> jnp.ndarray:
    """D-optimality: negative log-determinant of FIM.

    Minimising this maximises the volume of information (minimises the
    volume of the confidence ellipsoid).
    """
    sign, logdet = jnp.linalg.slogdet(FIM + _EPS * jnp.eye(FIM.shape[0]))
    return -logdet


def e_optimal(FIM: jnp.ndarray) -> jnp.ndarray:
    """E-optimality: largest eigenvalue of CRLB = worst-case parameter variance."""
    CRLB = jnp.linalg.inv(FIM + _EPS * jnp.eye(FIM.shape[0]))
    return jnp.max(jnp.linalg.eigvalsh(CRLB))


def parameter_selective_crlb(
    FIM: jnp.ndarray,
    param_indices: jnp.ndarray,
) -> jnp.ndarray:
    """A-optimality restricted to a subset of parameters.

    Useful when only a subset of parameters (e.g., axon diameter) is of
    interest and the rest are nuisance parameters.

    Parameters
    ----------
    FIM : (P, P) array
    param_indices : 1-D int array
        Indices of the parameters of interest.

    Returns
    -------
    scalar: trace of the sub-CRLB for the selected parameters.
    """
    CRLB = jnp.linalg.inv(FIM + _EPS * jnp.eye(FIM.shape[0]))
    sub = CRLB[jnp.ix_(param_indices, param_indices)]
    return jnp.trace(sub)
