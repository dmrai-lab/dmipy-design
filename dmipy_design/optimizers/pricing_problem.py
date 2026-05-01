"""
Pricing problem for column generation OED.

For each waveform type, finds the shell parameters (b, timing, frequency)
that maximise the reduced cost:

    rc(u) = trace(F_total^{-1} @ F_new(u))

where F_new(u) is the FIM of a new 30-direction shell at parameters u.
The KW threshold is always P (number of parameters), regardless of n_dirs, because
sum_k w_k trace(F^{-1} F_k) = trace(F^{-1} F_total) = P by construction.

Uses JAX autodiff: jax.value_and_grad through compute_fim_averaged.

Parameterization
----------------
PGSE: v in [0,1]^3 -> (b, delta, Delta)
  v[0] -> b     in [50e6, 10000e6] s/m²
  v[1] -> delta in [0.005, 0.060] s
  v[2] -> ratio = Delta/delta in [1.1, 3.0]  (guarantees Delta > delta)

OGSE: v in [0,1]^2 -> (f, G), b derived
  v[0] -> f in [10, 500] Hz
  v[1] -> G in [0.02, 0.30] T/m
  b = gamma² G² t_eff³,  t_eff = 1/(4f)

float64 throughout (required for C4 Van Gelderen sums).
"""

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize

from ..fim import compute_fim_averaged
from ..jax_scheme_encoder import encode_pgse_shell, encode_ogse_shell

GAMMA = 267513000.0  # rad/(s·T)
N_DIRS = 30

# ---------------------------------------------------------------------------
# Fixed isotropic 30-direction hemisphere
# ---------------------------------------------------------------------------

def _get_isotropic_dirs(n: int = 30) -> np.ndarray:
    """Return n approximately isotropic unit vectors on the hemisphere."""
    try:
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric362')
        verts = sphere.vertices
        # Select hemisphere (z >= 0)
        hemi = verts[verts[:, 2] >= 0]
        # Downsample to n directions
        idx = np.round(np.linspace(0, len(hemi) - 1, n)).astype(int)
        dirs = hemi[idx]
        # Normalise
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs.astype(np.float64)
    except Exception:
        # Fallback: fibonacci sphere on hemisphere
        golden = (1 + 5**0.5) / 2
        i = np.arange(n)
        theta = np.arccos(1.0 - (i + 0.5) / n)
        phi = 2 * np.pi * i / golden
        dirs = np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=1)
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs.astype(np.float64)


BVECS_30 = _get_isotropic_dirs(N_DIRS)

# ---------------------------------------------------------------------------
# Atom parameter bounds (physical units)
# ---------------------------------------------------------------------------
PGSE_B_RANGE          = (50e6,   10000e6)   # s/m²
PGSE_DELTA_RANGE      = (0.005,  0.060)     # s
PGSE_DELTA_RATIO_RANGE = (1.1,   3.0)       # Delta / delta

OGSE_F_RANGE = (10.0,  500.0)   # Hz
OGSE_G_RANGE = (0.02,  0.30)    # T/m (≤ 300 mT/m Connectom limit)


# ---------------------------------------------------------------------------
# Normalised decode helpers
# ---------------------------------------------------------------------------

def decode_pgse(v):
    """v in [0,1]^3 -> (b, delta, Delta) in physical units.

    JAX-traceable: v may be a jnp array (traced) or np array (concrete).
    Returns JAX scalars when v is traced, plain floats when v is concrete numpy.
    """
    b     = PGSE_B_RANGE[0]     + v[0] * (PGSE_B_RANGE[1]     - PGSE_B_RANGE[0])
    delta = PGSE_DELTA_RANGE[0] + v[1] * (PGSE_DELTA_RANGE[1] - PGSE_DELTA_RANGE[0])
    ratio = PGSE_DELTA_RATIO_RANGE[0] + v[2] * (
        PGSE_DELTA_RATIO_RANGE[1] - PGSE_DELTA_RATIO_RANGE[0]
    )
    Delta = delta * ratio
    return b, delta, Delta


def decode_ogse(v):
    """v in [0,1]^2 -> (f, G, b) with b derived from physics.

    JAX-traceable: v may be a jnp array (traced) or np array (concrete).
    """
    f = OGSE_F_RANGE[0] + v[0] * (OGSE_F_RANGE[1] - OGSE_F_RANGE[0])
    G = OGSE_G_RANGE[0] + v[1] * (OGSE_G_RANGE[1] - OGSE_G_RANGE[0])
    t_eff = 1.0 / (4.0 * f)
    b = GAMMA**2 * G**2 * t_eff**3
    return f, G, b


# ---------------------------------------------------------------------------
# Reduced cost (rc) objective builders
# ---------------------------------------------------------------------------

def build_rc_objective(
    forward_fn,
    prior_samples,
    sigma: float,
    F_total_inv_np: np.ndarray,
    wtype: str,
    bvecs: np.ndarray = BVECS_30,
):
    """Build a numpy-compatible (value, grad) function for the pricing problem.

    The returned function maps v_norm (in [0,1]^n_p) to (-rc, -grad_v_norm),
    suitable for scipy L-BFGS-B minimisation.

    Parameters
    ----------
    forward_fn : callable (theta, JaxScheme) -> jnp.ndarray
        JAX-differentiable forward model.
    prior_samples : jnp.ndarray, shape (M, P)
        Prior parameter samples (float64).
    sigma : float
        Noise standard deviation.
    F_total_inv_np : np.ndarray, shape (P, P)
        Inverse of current total FIM (numpy, float64).
    wtype : str
        'pgse' or 'ogse'.
    bvecs : np.ndarray, shape (n_dirs, 3)
        Fixed gradient directions for the candidate shell.

    Returns
    -------
    neg_rc_and_grad_np : callable(v_np) -> (float, np.ndarray)
    n_params_atom : int
        Dimensionality of the normalised parameter vector.
    """
    F_inv = jnp.array(F_total_inv_np, dtype=jnp.float64)
    n_dirs = len(bvecs)
    bvecs_jax = jnp.array(bvecs, dtype=jnp.float64)

    if wtype == 'pgse':
        n_params_atom = 3

        def rc_fn(v):
            b, delta, Delta = decode_pgse(v)
            scheme = encode_pgse_shell(b, delta, Delta, bvecs_jax)
            F_new = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
            return jnp.trace(F_inv @ F_new)

    elif wtype == 'ogse':
        n_params_atom = 2

        def rc_fn(v):
            f, G, _b = decode_ogse(v)
            scheme = encode_ogse_shell(f, G, bvecs_jax)
            F_new = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
            return jnp.trace(F_inv @ F_new)

    else:
        raise ValueError(f"Unknown waveform type: '{wtype}'. Choose 'pgse' or 'ogse'.")

    rc_and_grad = jax.value_and_grad(rc_fn)

    def neg_rc_and_grad_np(v_np: np.ndarray):
        v = jnp.array(v_np, dtype=jnp.float64)
        val, grad = rc_and_grad(v)
        return -float(val), -np.array(grad, dtype=np.float64)

    return neg_rc_and_grad_np, n_params_atom


def solve_pricing(
    forward_fn,
    prior_samples,
    sigma: float,
    F_total_inv_np: np.ndarray,
    wtype: str,
    n_restarts: int = 8,
    rng_seed: int = 0,
    bvecs: np.ndarray = BVECS_30,
):
    """Solve the pricing problem for a given waveform type.

    Finds the shell parameters (in normalised [0,1]^n_p space) that maximise
    the reduced cost rc(u) = trace(F_total^{-1} F_new(u)) / n_dirs.

    Parameters
    ----------
    forward_fn : callable (theta, JaxScheme) -> jnp.ndarray
    prior_samples : jnp.ndarray, shape (M, P)
    sigma : float
    F_total_inv_np : np.ndarray, shape (P, P)
    wtype : str   'pgse' or 'ogse'
    n_restarts : int
        Number of random restarts for L-BFGS-B.
    rng_seed : int
        Random seed for reproducibility.
    bvecs : np.ndarray, shape (n_dirs, 3)

    Returns
    -------
    best_rc : float
        Maximum reduced cost found.
    best_params : dict
        Decoded physical parameters of the best atom.
    best_scheme : JaxScheme
        The shell JaxScheme for the best atom.
    """
    rng = np.random.default_rng(rng_seed)
    neg_rc_and_grad, n_p = build_rc_objective(
        forward_fn, prior_samples, sigma, F_total_inv_np, wtype, bvecs
    )
    bounds = [(0.0, 1.0)] * n_p

    best_rc = -np.inf
    best_v  = None

    for _ in range(n_restarts):
        v0 = rng.uniform(0.0, 1.0, size=n_p)
        result = scipy.optimize.minimize(
            neg_rc_and_grad,
            v0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6},
        )
        rc = -result.fun
        if rc > best_rc:
            best_rc = rc
            best_v  = result.x.copy()

    best_v = np.clip(best_v, 0.0, 1.0)
    bvecs_jax = jnp.array(bvecs, dtype=jnp.float64)

    if wtype == 'pgse':
        b, delta, Delta = decode_pgse(best_v)
        best_params = {
            'type': 'pgse',
            'b': b,
            'delta': delta,
            'Delta': Delta,
        }
        best_scheme = encode_pgse_shell(b, delta, Delta, bvecs_jax)

    elif wtype == 'ogse':
        f, G, b = decode_ogse(best_v)
        best_params = {
            'type': 'ogse',
            'f': f,
            'G': G,
            'b': b,
        }
        best_scheme = encode_ogse_shell(f, G, bvecs_jax)

    else:
        raise ValueError(f"Unknown waveform type: '{wtype}'")

    return best_rc, best_params, best_scheme
