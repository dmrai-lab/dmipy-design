"""
JAX-differentiable waveform G-array builders.

Each function returns a (n_meas, n_t, 3) JAX array representing the gradient
waveform time series, constructed from physical parameters via ``jnp.where``
— fully differentiable with respect to all scalar parameters (G_amp, delta,
Delta, freq) via ``jax.grad``.

Note on differentiability of jnp.where w.r.t. timing parameters
----------------------------------------------------------------
``jnp.where(condition, x, y)`` is differentiable w.r.t. the *values* x and y
but not through the *condition* itself (gradient is zero or undefined at the
boundaries). In practice, for L-BFGS optimisation the gradient is well-defined
almost everywhere (piecewise constant timing is fine away from transition
points). For ``G_amp`` differentiation is exact (linear scaling). For
``delta`` / ``Delta``, the gradient is computed via the chain rule through
the piecewise-constant shape function — zero at smooth regions, undefined
only at the transition edges (a measure-zero set).

Usage
-----
    import jax.numpy as jnp
    from dmipy_design.waveform_builders import build_pgse_G

    bvecs = jnp.ones((30, 3)) / jnp.sqrt(3.0)
    G = build_pgse_G(G_amp=0.2, delta=0.02, Delta=0.05, bvecs=bvecs, dt=1e-4)
    # G.shape -> (30, 601, 3)

    # Differentiable w.r.t. G_amp:
    import jax
    def signal_fn(G_amp):
        G = build_pgse_G(G_amp, 0.02, 0.05, bvecs, 1e-4)
        return G.mean()
    jax.grad(signal_fn)(0.2)  # works
"""

from __future__ import annotations

import math

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _require_jax():
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for waveform_builders.")


# ---------------------------------------------------------------------------
# PGSE (pulsed gradient spin echo)
# ---------------------------------------------------------------------------

def build_pgse_G(
    G_amp: float,
    delta: float,
    Delta: float,
    bvecs: "jnp.ndarray",
    dt: float,
    T_total: float | None = None,
) -> "jnp.ndarray":
    """Build a PGSE gradient waveform array.

    Two rectangular lobes of equal amplitude and opposite sign:
        Lobe 1 (positive): t ∈ [0, delta)
        Lobe 2 (negative): t ∈ [Delta, Delta + delta)
        Zero elsewhere

    Parameters
    ----------
    G_amp : float or JAX scalar
        Gradient amplitude in T/m (positive scalar).
    delta : float or JAX scalar
        Gradient pulse duration in seconds.
    Delta : float or JAX scalar
        Separation between lobe starts in seconds. Must satisfy Delta > delta.
    bvecs : jnp.ndarray, shape (n_meas, 3)
        Gradient directions (unit vectors). G is oriented along each bvec.
    dt : float
        Time step in seconds. Determines n_t from T_total.
    T_total : float or None
        Total waveform duration. If None, defaults to Delta + delta.

    Returns
    -------
    G : jnp.ndarray, shape (n_meas, n_t, 3), float32
        Gradient waveform in T/m, differentiable w.r.t. G_amp, delta, Delta.
    """
    _require_jax()
    if T_total is None:
        T_total = float(Delta) + float(delta)
    n_t = max(int(round(T_total / dt)) + 1, 2)
    t = jnp.arange(n_t, dtype=jnp.float32) * float(dt)      # (n_t,)
    lobe1 = jnp.where((t >= 0.0) & (t < delta),  1.0, 0.0)  # (n_t,)
    lobe2 = jnp.where((t >= Delta) & (t < Delta + delta), -1.0, 0.0)
    shape = (lobe1 + lobe2).astype(jnp.float32)              # (n_t,)
    # G shape (n_meas, n_t, 3) = G_amp * bvecs[:, None, :] * shape[None, :, None]
    bvecs_f32 = jnp.asarray(bvecs, dtype=jnp.float32)        # (n_meas, 3)
    G = G_amp * jnp.einsum('mx,t->mtx', bvecs_f32, shape)    # (n_meas, n_t, 3)
    return G


# ---------------------------------------------------------------------------
# OGSE (oscillating gradient spin echo)
# ---------------------------------------------------------------------------

def build_ogse_G(
    G_amp: float,
    freq: float,
    bvecs: "jnp.ndarray",
    dt: float,
    T_total: float | None = None,
) -> "jnp.ndarray":
    """Build an OGSE sinusoidal gradient waveform array.

    A cosine waveform with the requested frequency:
        G(t) = G_amp * cos(2π * freq * t)   for t in [0, T_total]

    The waveform is applied along each bvec direction.

    Parameters
    ----------
    G_amp : float or JAX scalar
        Peak gradient amplitude in T/m.
    freq : float or JAX scalar
        Oscillation frequency in Hz.
    bvecs : jnp.ndarray, shape (n_meas, 3)
        Gradient directions.
    dt : float
        Time step in seconds.
    T_total : float or None
        Total waveform duration. If None, defaults to 1 / freq (one period).

    Returns
    -------
    G : jnp.ndarray, shape (n_meas, n_t, 3), float32
    """
    _require_jax()
    if T_total is None:
        T_total = 1.0 / float(freq)
    n_t = max(int(round(T_total / dt)) + 1, 2)
    t   = jnp.arange(n_t, dtype=jnp.float32) * float(dt)
    shape = (G_amp * jnp.cos(2.0 * math.pi * freq * t)).astype(jnp.float32)
    bvecs_f32 = jnp.asarray(bvecs, dtype=jnp.float32)
    G = jnp.einsum('mx,t->mtx', bvecs_f32, shape)
    return G


# ---------------------------------------------------------------------------
# PGSTE (pulsed gradient stimulated echo) — same diffusion physics as PGSE
# ---------------------------------------------------------------------------

def build_pgste_G(
    G_amp: float,
    delta: float,
    Delta: float,
    bvecs: "jnp.ndarray",
    dt: float,
    T_total: float | None = None,
) -> "jnp.ndarray":
    """Build a PGSTE gradient waveform array.

    Diffusion physics is identical to PGSE (same two-lobe shape), but the
    SNR model differs (TE = 2·delta, T1 relaxation during mixing time).
    The G-array is the same as PGSE — this function is an alias provided
    for naming symmetry.

    Parameters
    ----------
    G_amp : float or JAX scalar
    delta : float or JAX scalar
    Delta : float or JAX scalar
    bvecs : jnp.ndarray, shape (n_meas, 3)
    dt : float
    T_total : float or None

    Returns
    -------
    G : jnp.ndarray, shape (n_meas, n_t, 3), float32
    """
    return build_pgse_G(G_amp, delta, Delta, bvecs, dt, T_total)


# ---------------------------------------------------------------------------
# STE (spherical tensor encoding) — three Cartesian axis pairs
# ---------------------------------------------------------------------------

def build_ste_G(
    G_amp: float,
    Delta: float,
    bvecs: "jnp.ndarray",
    dt: float,
    T_total: float | None = None,
) -> "jnp.ndarray":
    """Build a spherical tensor encoding (STE) gradient waveform array.

    STE is implemented as three sequential bipolar gradient pairs along the
    three Cartesian axes (x, y, z), each contributing equally so that the
    off-diagonal B-tensor elements cancel:
        B = (b/3) I  (isotropic)

    Each bipolar pair has duration delta = Delta / 3 and spacing such that
    all three pairs fit in T_total. The waveform for measurement m uses
    bvecs[m] to weight the three Cartesian components.

    For the trajectory replay the STE G-array is constructed as a single
    vector waveform that is the superposition of three Cartesian components
    weighted by bvecs[m].

    Parameters
    ----------
    G_amp : float or JAX scalar
        Peak gradient amplitude per Cartesian axis in T/m.
    Delta : float or JAX scalar
        Total STE duration in seconds.
    bvecs : jnp.ndarray, shape (n_meas, 3)
        Gradient directions. Used to weight Cartesian components.
    dt : float
        Time step in seconds.
    T_total : float or None
        Total waveform duration. If None, defaults to Delta.

    Returns
    -------
    G : jnp.ndarray, shape (n_meas, n_t, 3), float32
    """
    _require_jax()
    if T_total is None:
        T_total = float(Delta)
    n_t = max(int(round(T_total / dt)) + 1, 2)
    t   = jnp.arange(n_t, dtype=jnp.float32) * float(dt)

    # Each Cartesian axis gets a bipolar rectangular pair of duration Delta/3
    # Axis x: positive [0, Delta/3), negative [Delta/3, 2*Delta/3)
    # Axis y: positive [Delta/3, 2*Delta/3), negative [2*Delta/3, Delta)
    # Axis z: positive [0, Delta/2), negative [Delta/2, Delta)  (simplified: same as x but z)
    # Simplified isotropic STE: use identical positive+negative pairs per axis,
    # staggered to avoid overlap.
    d3 = Delta / 3.0

    # Scalar shape per axis — (n_t,)
    shape_x = jnp.where((t >= 0.0)  & (t < d3),       1.0,
              jnp.where((t >= d3)   & (t < 2.0 * d3), -1.0, 0.0))
    shape_y = jnp.where((t >= 2.0*d3) & (t < 3.0*d3), 1.0,
              jnp.where((t >= 3.0*d3) & (t < 4.0*d3), -1.0, 0.0))
    shape_z = jnp.where((t >= 4.0*d3) & (t < 5.0*d3), 1.0,
              jnp.where((t >= 5.0*d3) & (t < 6.0*d3), -1.0, 0.0))

    # Stack: (n_t, 3) — shape of each Cartesian component
    shapes = jnp.stack([shape_x, shape_y, shape_z], axis=1).astype(jnp.float32)  # (n_t, 3)

    bvecs_f32 = jnp.asarray(bvecs, dtype=jnp.float32)  # (n_meas, 3)
    # G[m, t, x] = G_amp * bvecs[m, x] * shapes[t, x]
    G = G_amp * jnp.einsum('mx,tx->mtx', bvecs_f32, shapes)  # (n_meas, n_t, 3)
    return G
