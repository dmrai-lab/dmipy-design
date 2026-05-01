"""
Scheme-aware analytical forward functions for use with column generation OED.

Each function has signature:
    forward_fn(theta: jnp.ndarray, scheme: JaxScheme) -> jnp.ndarray

where theta is the parameter vector (physical units, float64) and scheme is a
JaxScheme with live JAX array fields. The output is the predicted signal vector
of shape (n_measurements,).

These functions are fully JAX-differentiable with respect to both theta and
scheme fields, enabling use in the pricing problem where jax.grad is taken
w.r.t. scheme parameters.

C4 Cylinder (Van Gelderen GPA) physics
---------------------------------------
The signal requires:
  - gradient_strengths  : scheme field (T/m)
  - delta, Delta        : scheme fields (s)
  - bvalues             : scheme field (s/m²)
  - gradient_directions : scheme.bvecs (N, 3)

The cylinder axis is marginalised by computing the signal for each gradient
direction against a fixed cylinder axis [0, 0, 1], then averaging over the
N directions in the shell. This equals the true spherical mean when the
bvecs uniformly sample the sphere.

Roots
-----
The Van Gelderen sum uses 100 zeros of J1'(x) (first derivative of J1 Bessel
function). These are precomputed once using scipy.special.jnp_zeros(1, 100)
and stored as a module-level constant.

float64
-------
Call ``jax.config.update("jax_enable_x64", True)`` before importing this
module (or in any script/test that uses these forward functions) to ensure
C4 Van Gelderen sums converge correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .jax_scheme_encoder import JaxScheme

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GAMMA = 267513000.0  # rad/(s·T)

# ---------------------------------------------------------------------------
# Van Gelderen roots (J1' zeros = zeros of J1 first derivative)
# scipy convention: jnp_zeros(n, nt) returns nt zeros of J_n'(x)
# For C4 cylinder we need zeros of J1'(x), i.e. n=1.
# ---------------------------------------------------------------------------
_N_ROOTS = 100

def _make_roots():
    from scipy.special import jnp_zeros
    return jnp.array(jnp_zeros(1, _N_ROOTS))

C4_CYLINDER_ROOTS = _make_roots()  # shape (100,)

# ---------------------------------------------------------------------------
# Default perpendicular diffusivity for the C4 cylinder model.
# ---------------------------------------------------------------------------
D_PERP_DEFAULT = 1.7e-9  # m²/s (free water)


def _c4cylinder_signal_full(
    bvalues,
    gradient_directions,
    gradient_strengths,
    delta,
    Delta,
    mu_cart,
    lambda_par,
    diameter,
    diffusion_perpendicular,
    gyromagnetic_ratio,
    roots_jax,
):
    """Van Gelderen GPA cylinder signal — vectorized over N measurements.

    Wraps c4cylinder_signal from dmipy_fit with explicit array inputs.
    Returns shape (N,).
    """
    from dmipy_fit.jax.signal_models_jax import c4cylinder_signal
    return c4cylinder_signal(
        bvalues=bvalues,
        gradient_directions=gradient_directions,
        gradient_strengths=gradient_strengths,
        delta=delta,
        Delta=Delta,
        mu_cart=mu_cart,
        lambda_par=lambda_par,
        diameter=diameter,
        diffusion_perpendicular=diffusion_perpendicular,
        gyromagnetic_ratio=gyromagnetic_ratio,
        roots_jax=roots_jax,
    )


def ball_c4cylinder_forward(
    theta: jnp.ndarray,
    scheme: JaxScheme,
    diffusion_perpendicular: float = D_PERP_DEFAULT,
    roots_jax: jnp.ndarray = C4_CYLINDER_ROOTS,
) -> jnp.ndarray:
    """
    Spherical-mean signal for Ball + C4Cylinder (GPA) model.

    Model: S = vf_ball * E_ball + (1 - vf_ball) * E_cylinder_spherical_mean

    theta[0]: vf_ball     — volume fraction of Ball compartment [0, 1]
    theta[1]: lambda_iso  — isotropic diffusivity of Ball (m²/s)
    theta[2]: lambda_par  — parallel diffusivity of Cylinder (m²/s)
    theta[3]: diameter    — cylinder diameter (m)

    The spherical mean of the cylinder is computed by calling c4cylinder_signal
    with a fixed cylinder axis mu = [0, 0, 1] and averaging the per-direction
    signals over all N gradient directions in the shell.  This equals the true
    spherical mean when bvecs uniformly sample the hemisphere.

    Handles encoding types:
    - 'pgse' (default): direction-dependent signal, projects G onto cylinder axis
    - 'ste': direction-independent; uses spherical mean of cylinder signal
      (approximation: exact for Ball, powder-average approx for restricted Cylinder)

    Parameters
    ----------
    theta : jnp.ndarray, shape (4,), float64
        [vf_ball, lambda_iso, lambda_par, diameter]
    scheme : JaxScheme
        Must have: bvalues, bvecs, delta, Delta.
        gradient_strengths is used if present; otherwise derived from b, delta, Delta.
        encoding_type: 'pgse' (default) or 'ste'.
    diffusion_perpendicular : float
        D_perp for Van Gelderen GPA (m²/s). Default: 1.7e-9 m²/s.
    roots_jax : jnp.ndarray, shape (R,)
        Precomputed J1' zeros for Van Gelderen sum.

    Returns
    -------
    signal : jnp.ndarray, shape (n_measurements,)
    """
    from dmipy_fit.jax.signal_models_jax import g1ball_signal

    vf_ball    = theta[0]
    lambda_iso = theta[1]
    lambda_par = theta[2]
    diameter   = theta[3]
    vf_cyl     = 1.0 - vf_ball

    # --- Ball signal: E_ball = exp(-b * lambda_iso) ---
    E_ball = g1ball_signal(scheme.bvalues, lambda_iso)  # (N,)

    # --- gradient_strengths ---
    if scheme.gradient_strengths is not None:
        G_vals = scheme.gradient_strengths
    else:
        G_vals = jnp.sqrt(
            scheme.bvalues / (
                GAMMA**2 * scheme.delta**2 * (scheme.Delta - scheme.delta / 3.0)
            )
        )

    # Detect encoding type; default to 'pgse' for backward compatibility
    encoding = getattr(scheme, 'encoding_type', 'pgse')

    if encoding == 'ste':
        # STE: B = (b/3)*I — signal is the powder average (spherical mean)
        # over all cylinder orientations.  We compute this by averaging the
        # PGSE cylinder signal over 30 isotropic directions with a fixed
        # cylinder axis z-hat.  Because the shell bvecs uniformly cover the
        # hemisphere, this average equals the true spherical mean.
        #
        # For Ball this is exact: E_STE_ball = exp(-b * lambda_iso)
        # (already computed above).
        #
        # For Cylinder this is the powder-average approximation: valid when
        # the fibre orientations are isotropically distributed (tumour cells,
        # WM with unknown orientation).

        # Stack the per-direction cylinder signals then average over directions
        # to obtain one powder-average signal per measurement point.
        # Build a new scheme with single-direction bvecs equal to the stored
        # bvecs one row at a time, then average — or more efficiently,
        # evaluate _c4cylinder_signal_full with fixed axis z-hat (same as the
        # PGSE path) and average over the N direction rows.

        MU = jnp.array([0.0, 0.0, 1.0], dtype=scheme.bvalues.dtype)

        # E_cyl_per_dir has shape (N,): cylinder signal for each gradient
        # direction against the fixed axis z-hat.
        E_cyl_per_dir = _c4cylinder_signal_full(
            bvalues=scheme.bvalues,
            gradient_directions=scheme.bvecs,
            gradient_strengths=G_vals,
            delta=scheme.delta,
            Delta=scheme.Delta,
            mu_cart=MU,
            lambda_par=lambda_par,
            diameter=diameter,
            diffusion_perpendicular=diffusion_perpendicular,
            gyromagnetic_ratio=GAMMA,
            roots_jax=roots_jax,
        )  # shape (N,)

        # All measurements in an STE shell share the same b, delta, Delta so
        # the spherical mean is simply the mean of E_cyl_per_dir.  Broadcast
        # the scalar mean back to shape (N,) so the output matches PGSE shape.
        E_cyl_sm = jnp.mean(E_cyl_per_dir) * jnp.ones_like(E_cyl_per_dir)
        E_cyl = E_cyl_sm

    else:
        # PGSE (default): direction-dependent signal
        # Fix cylinder axis to z-hat; the average over the isotropic shell of
        # gradient directions equals the spherical mean over cylinder orientations.
        MU = jnp.array([0.0, 0.0, 1.0], dtype=scheme.bvalues.dtype)

        # c4cylinder_signal is vectorized over N measurements natively.
        E_cyl_per_dir = _c4cylinder_signal_full(
            bvalues=scheme.bvalues,
            gradient_directions=scheme.bvecs,
            gradient_strengths=G_vals,
            delta=scheme.delta,
            Delta=scheme.Delta,
            mu_cart=MU,
            lambda_par=lambda_par,
            diameter=diameter,
            diffusion_perpendicular=diffusion_perpendicular,
            gyromagnetic_ratio=GAMMA,
            roots_jax=roots_jax,
        )  # shape (N,)

        # The per-measurement cylinder signal already accounts for the angle
        # between the gradient direction and the fixed axis.  We return these
        # per-direction values directly (not averaged) so that the FIM uses
        # the full directional information from the 30-direction shell.
        E_cyl = E_cyl_per_dir

    return vf_ball * E_ball + vf_cyl * E_cyl
