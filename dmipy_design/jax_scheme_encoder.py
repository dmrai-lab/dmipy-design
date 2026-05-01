"""
JAX-traceable acquisition scheme encoder.

This is the critical traceability layer identified in the plan: it represents
a PGSE (or general free-waveform) acquisition as pure JAX arrays so that
``jax.jacobian`` and ``jax.grad`` can differentiate through it.

The standard dmipy-core ``PGSEAcquisitionScheme`` constructor calls numpy
operations that are not JAX-traceable.  ``JaxScheme`` bypasses the constructor
entirely and feeds directly into ``dmipy.jax.jax_compat.scheme_to_jax()``.

Usage
-----
    from dmipy_design.jax_scheme_encoder import encode_pgse
    import jax
    import jax.numpy as jnp

    u = jnp.array([1e9, 0.02, 0.05])   # [b_value (s/m²), delta (s), Delta (s)]
    bvecs = jnp.tile(jnp.array([1., 0., 0.]), (30, 1))
    scheme = encode_pgse(u, bvecs)
    # scheme.bvalues, scheme.delta, scheme.Delta are jnp arrays — fully traceable

B-tensor encoding (Phase 2)
----------------------------
    from dmipy_design.jax_scheme_encoder import encode_ste, encode_ogse

    # Spherical Tensor Encoding (isotropic b-tensor):
    u_ste = jnp.array([1e9, 0.02, 0.05])   # [b_value, delta, Delta]
    scheme_ste = encode_ste(u_ste, bvecs)
    # scheme_ste["b_tensors"] has shape (N, 3, 3) with B = (b/3)*I

    # OGSE:
    u_ogse = jnp.array([1e9, 50.0, 4.0])  # [b_value, frequency (Hz), n_cycles]
    scheme_ogse = encode_ogse(u_ogse, bvecs)
    # scheme_ogse["t_eff"] = 1 / (4 * frequency)
"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class JaxScheme:
    """Minimal JAX-traceable representation of a PGSE acquisition scheme.

    All fields are JAX arrays, making this struct fully differentiable.

    Attributes
    ----------
    bvalues           : jnp.ndarray, shape (N,)       s/m²
    bvecs             : jnp.ndarray, shape (N, 3)     unit vectors
    delta             : jnp.ndarray, shape (N,) or scalar    gradient pulse duration (s)
    Delta             : jnp.ndarray, shape (N,) or scalar    gradient separation (s)
    TE                : jnp.ndarray, shape (N,) or scalar    echo time (s), optional
    gradient_strengths: jnp.ndarray, shape (N,), optional    gradient amplitude (T/m)
    """
    bvalues: jnp.ndarray
    bvecs: jnp.ndarray
    delta: jnp.ndarray
    Delta: jnp.ndarray
    TE: jnp.ndarray | None = None
    gradient_strengths: jnp.ndarray | None = None


def encode_pgse(
    u: jnp.ndarray,
    bvecs: jnp.ndarray,
) -> JaxScheme:
    """Encode a PGSE protocol vector into a JAX-traceable scheme.

    Parameters
    ----------
    u : jnp.ndarray, shape (3,)
        Protocol vector ``[b_value (s/m²), delta (s), Delta (s)]``.
        This is the vector that dmipy-design optimises via ``jax.grad``.
    bvecs : jnp.ndarray, shape (N, 3)
        Gradient directions (unit vectors).  Fixed during optimisation.

    Returns
    -------
    scheme : JaxScheme
        All-JAX scheme struct, traceable through autodiff.

    Notes
    -----
    The relationship between b-value, delta, Delta, and gradient amplitude G is:
        b = gamma² G² delta² (Delta - delta/3)
    This function does NOT verify hardware feasibility; that is enforced
    externally via ``HardwareConstraints``.
    """
    _GAMMA = 267513000.0  # rad/(s·T)
    b = jnp.broadcast_to(u[0], (bvecs.shape[0],))
    delta = jnp.broadcast_to(u[1], (bvecs.shape[0],))
    Delta = jnp.broadcast_to(u[2], (bvecs.shape[0],))
    # b = gamma² G² delta² (Delta - delta/3)  →  G = sqrt(b / (gamma² delta² (Delta - delta/3)))
    gradient_strengths = jnp.sqrt(b / (_GAMMA**2 * delta**2 * (Delta - delta / 3.0)))
    return JaxScheme(bvalues=b, bvecs=bvecs, delta=delta, Delta=Delta,
                     gradient_strengths=gradient_strengths)


def encode_lte(
    u: jnp.ndarray,
    bvecs: jnp.ndarray,
) -> "JaxScheme":
    """Encode a Linear Tensor Encoding (LTE) scheme.

    LTE is equivalent to PGSE — the b-tensor is rank-1 and aligned with
    the gradient direction.  This function is an alias for ``encode_pgse``
    provided for naming symmetry with ``encode_ste`` / ``encode_ogse``.

    Parameters
    ----------
    u : jnp.ndarray, shape (3,)
        Protocol vector ``[b_value (s/m²), delta (s), Delta (s)]``.
    bvecs : jnp.ndarray, shape (N, 3)

    Returns
    -------
    scheme : JaxScheme
    """
    return encode_pgse(u, bvecs)


def encode_ste(
    u: jnp.ndarray,
    bvecs: jnp.ndarray,
) -> dict:
    """Encode a Spherical Tensor Encoding (STE) scheme.

    STE produces an isotropic b-tensor: B = (b/3) * I.
    The signal depends only on the b-value (trace), not the gradient direction.

    Parameters
    ----------
    u : jnp.ndarray, shape (3,)
        Protocol vector ``[b_value (s/m²), delta (s), Delta (s)]``.
    bvecs : jnp.ndarray, shape (N, 3)
        Gradient directions.  Used only to determine the number of measurements;
        STE is rotationally invariant so direction does not affect the signal.

    Returns
    -------
    scheme : dict with keys
        ``b_values``   : jnp.ndarray, shape (N,)   s/m²
        ``b_tensors``  : jnp.ndarray, shape (N, 3, 3)  isotropic: (b/3)*I
        ``bvecs``      : jnp.ndarray, shape (N, 3)  (stored for reference)
        ``delta``      : jnp.ndarray, shape (N,)   s
        ``Delta``      : jnp.ndarray, shape (N,)   s
        ``encoding``   : str  ``"STE"``

    Notes
    -----
    Physics:
        STE b-tensor: B_ij = (b/3) * delta_ij
        Ball signal:  E_STE = exp(-b * d)  (same as PGSE spherical mean)
        Stick signal: E_STE(b) = spherical mean, direction-independent
        FIM partial:  dE/dd = -b/3 * E  (per isotropic axis)
    """
    N = bvecs.shape[0]
    b = jnp.broadcast_to(u[0], (N,))
    delta = jnp.broadcast_to(u[1], (N,))
    Delta = jnp.broadcast_to(u[2], (N,))

    # Isotropic b-tensor: B = (b/3) * I for each measurement
    eye3 = jnp.eye(3)
    b_tensors = (b[:, None, None] / 3.0) * eye3[None, :, :]  # (N, 3, 3)

    return {
        "b_values": b,
        "b_tensors": b_tensors,
        "bvecs": bvecs,
        "delta": delta,
        "Delta": Delta,
        "encoding": "STE",
    }


def encode_ogse(
    u_ogse: jnp.ndarray,
    bvecs: jnp.ndarray,
) -> dict:
    """Encode an Oscillating Gradient Spin Echo (OGSE) scheme.

    OGSE uses oscillating gradients at a fixed frequency.  The effective
    diffusion time is frequency-dependent:
        t_eff = 1 / (4 * frequency)

    This replaces ``Delta - delta/3`` in the Gaussian phase approximation
    for the signal kernel.

    Parameters
    ----------
    u_ogse : jnp.ndarray, shape (3,)
        Protocol vector ``[b_value (s/m²), frequency (Hz), n_cycles (float)]``.
    bvecs : jnp.ndarray, shape (N, 3)
        Gradient directions (unit vectors).

    Returns
    -------
    scheme : dict with keys
        ``b_values``   : jnp.ndarray, shape (N,)   s/m²
        ``bvecs``      : jnp.ndarray, shape (N, 3)
        ``frequency``  : float  Hz
        ``n_cycles``   : float
        ``t_eff``      : float  s  (= 1 / (4 * frequency))
        ``encoding``   : str  ``"OGSE"``

    Notes
    -----
    Physics (Gaussian phase approximation):
        t_eff = 1 / (4 * f)
        The effective diffusion coefficient at frequency f is:
            D_eff(f) ≈ D * [1 - tanh(sqrt(D) * π * f^0.5 * δ) /
                                    (sqrt(D) * π * f^0.5 * δ)]
        For the FIM this is often approximated by using t_eff in place of
        Delta - delta/3 in any t_eff-dependent signal model.
    """
    N = bvecs.shape[0]
    b = jnp.broadcast_to(u_ogse[0], (N,))
    freq = u_ogse[1]        # scalar frequency (Hz)
    n_cycles = u_ogse[2]    # scalar number of oscillation cycles

    t_eff = 1.0 / (4.0 * freq)  # effective diffusion time (s)

    return {
        "b_values": b,
        "bvecs": bvecs,
        "frequency": freq,
        "n_cycles": n_cycles,
        "t_eff": t_eff,
        "encoding": "OGSE",
    }


def encode_multishell_pgse(
    b_values: jnp.ndarray,
    delta_values: jnp.ndarray,
    Delta_values: jnp.ndarray,
    bvecs_per_shell: list[jnp.ndarray],
) -> JaxScheme:
    """Encode a multi-shell PGSE protocol as a JAX-traceable scheme.

    Parameters
    ----------
    b_values : jnp.ndarray, shape (S,)      one b-value per shell
    delta_values : jnp.ndarray, shape (S,)  one delta per shell
    Delta_values : jnp.ndarray, shape (S,)  one Delta per shell
    bvecs_per_shell : list of (Ns, 3) arrays

    Returns
    -------
    scheme : JaxScheme
        Concatenated scheme across all shells.
    """
    _GAMMA = 267513000.0  # rad/(s·T)
    all_b, all_delta, all_Delta, all_bvecs, all_G = [], [], [], [], []
    for s, bvecs in enumerate(bvecs_per_shell):
        n = bvecs.shape[0]
        b_s = jnp.broadcast_to(b_values[s], (n,))
        d_s = jnp.broadcast_to(delta_values[s], (n,))
        D_s = jnp.broadcast_to(Delta_values[s], (n,))
        all_b.append(b_s)
        all_delta.append(d_s)
        all_Delta.append(D_s)
        all_bvecs.append(bvecs)
        all_G.append(jnp.sqrt(b_s / (_GAMMA**2 * d_s**2 * (D_s - d_s / 3.0))))
    return JaxScheme(
        bvalues=jnp.concatenate(all_b),
        bvecs=jnp.concatenate(all_bvecs),
        delta=jnp.concatenate(all_delta),
        Delta=jnp.concatenate(all_Delta),
        gradient_strengths=jnp.concatenate(all_G),
    )


# ---------------------------------------------------------------------------
# Shell-level encoders for column generation OED
# These accept explicit scalar parameters (not u-vectors) and build a
# complete n-direction shell as a JaxScheme.
# ---------------------------------------------------------------------------

_GAMMA = 267513000.0  # rad/(s·T)


def encode_pgse_shell(
    b,
    delta,
    Delta,
    bvecs: jnp.ndarray,
) -> JaxScheme:
    """Build a PGSE shell JaxScheme from explicit scalar parameters.

    Parameters may be plain Python floats, numpy scalars, or JAX traced
    scalars (for use inside jax.grad / jax.value_and_grad).

    Parameters
    ----------
    b : scalar (float or JAX scalar)
        b-value (s/m²).
    delta : scalar
        Gradient pulse duration (s).
    Delta : scalar
        Gradient separation (s).
    bvecs : jnp.ndarray, shape (N, 3)
        Gradient directions for this shell.

    Returns
    -------
    JaxScheme with gradient_strengths populated.
    """
    n = bvecs.shape[0]
    b_arr     = jnp.broadcast_to(jnp.asarray(b,     dtype=jnp.float64), (n,))
    delta_arr = jnp.broadcast_to(jnp.asarray(delta, dtype=jnp.float64), (n,))
    Delta_arr = jnp.broadcast_to(jnp.asarray(Delta, dtype=jnp.float64), (n,))
    G_arr = jnp.sqrt(b_arr / (_GAMMA**2 * delta_arr**2 * (Delta_arr - delta_arr / 3.0)))
    return JaxScheme(
        bvalues=b_arr,
        bvecs=jnp.asarray(bvecs, dtype=jnp.float64),
        delta=delta_arr,
        Delta=Delta_arr,
        gradient_strengths=G_arr,
    )


def encode_ogse_shell(
    freq,
    G,
    bvecs: jnp.ndarray,
) -> JaxScheme:
    """Build an OGSE shell JaxScheme from (f, G) parameterization.

    The b-value is derived:  b = gamma² G² t_eff³,  t_eff = 1 / (4 f).
    delta = t_eff, Delta = t_eff + t_eff/3 so that Delta - delta/3 = t_eff.
    gradient_strengths = G (constant across all directions in the shell).

    Parameters may be plain Python floats, numpy scalars, or JAX traced
    scalars (for use inside jax.grad / jax.value_and_grad).

    Parameters
    ----------
    freq : scalar
        Oscillation frequency (Hz).
    G : scalar
        Gradient strength (T/m).
    bvecs : jnp.ndarray, shape (N, 3)
        Gradient directions for this shell.

    Returns
    -------
    JaxScheme with gradient_strengths = G, delta = t_eff, Delta = t_eff*(4/3).
    """
    n = bvecs.shape[0]
    freq_j = jnp.asarray(freq, dtype=jnp.float64)
    G_j    = jnp.asarray(G,    dtype=jnp.float64)
    t_eff  = 1.0 / (4.0 * freq_j)
    b      = _GAMMA**2 * G_j**2 * t_eff**3
    b_arr     = jnp.broadcast_to(b,     (n,))
    t_arr     = jnp.broadcast_to(t_eff, (n,))
    # Delta - delta/3 = t_eff  =>  Delta = t_eff + t_eff/3 = 4*t_eff/3
    Delta_arr = jnp.broadcast_to(t_eff + t_eff / 3.0, (n,))
    G_arr     = jnp.broadcast_to(G_j,   (n,))
    return JaxScheme(
        bvalues=b_arr,
        bvecs=jnp.asarray(bvecs, dtype=jnp.float64),
        delta=t_arr,
        Delta=Delta_arr,
        gradient_strengths=G_arr,
    )
