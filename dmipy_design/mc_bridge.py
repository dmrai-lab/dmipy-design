"""Monte Carlo bridge: wrap dmipy-sim as a forward function for FIM computation.

This module closes the analytical-simulation loop: OED protocols can be
optimised using MC ground truth signals rather than analytical models.

Analytical signal models (GPA, GPD) break down at short diffusion times,
large gradients, or complex geometries.  The MC-bridge lets OED optimise
acquisition protocols using MC ground truth — the same MC signals that
the compendium uses for claim validation.

Usage
-----
Fixed geometry (most OED use cases — geometry parameters are known):

    from dmipy_sim.geometries import Cylinder
    from dmipy_design.mc_bridge import build_mc_forward_fn_fixed
    from dmipy_design.fim import compute_fim_fd
    from dmipy_design.jax_scheme_encoder import encode_pgse
    import jax.numpy as jnp
    import numpy as np

    bvecs = np.tile([1., 0., 0.], (20, 1))
    geom = Cylinder(radius=2e-6, orientation=[0, 0, 1])
    mc_fwd = build_mc_forward_fn_fixed(geom, diffusivity=1.7e-9, n_walkers=5000)
    u = jnp.array([1000e6, 0.013, 0.022])   # [b (s/m²), delta (s), Delta (s)]
    scheme = encode_pgse(u, jnp.array(bvecs))
    theta = jnp.zeros(0)   # no tissue parameters for fixed-geometry forward fn
    FIM = compute_fim_fd(mc_fwd, theta, scheme, sigma=0.02)

Factory geometry (for OED over geometry parameters):

    from dmipy_design.mc_bridge import build_mc_forward_fn

    def geom_factory(params):
        return Cylinder(radius=params['radius'], orientation=[0, 0, 1])

    mc_fwd = build_mc_forward_fn(geom_factory, diffusivity=1.7e-9, n_walkers=5000)
    FIM = compute_fim_fd(mc_fwd, params={'radius': 2e-6}, scheme=scheme, sigma=0.02)

Notes
-----
The MC forward function is stochastic.  For stable finite-difference FIM
estimates, use enough walkers so that MC noise is well below the
finite-difference step size scaled by the signal gradient:

    sigma_MC ≪ eps * |dE/dtheta|

Rule of thumb: n_walkers ≥ 5000 for cylinder at b = 1000 s/mm².
The seed is held fixed across forward calls so that finite differences
cancel MC noise to first order (same walker paths for E0 and E_plus).
"""

import numpy as np


# Gyromagnetic ratio for protons (rad / (s·T))
_GAMMA = 2.675e8


def _scheme_to_sim_waveform(scheme, n_t: int = 500):
    """Convert a JaxScheme or dict to a dmipy-sim PGSE Waveform.

    Extracts b_values, bvecs, delta, Delta from the scheme and constructs
    a PGSE waveform using dmipy-sim's ``pgse()`` and ``set_b()`` functions.

    Only the first measurement's delta/Delta is used (assumes all
    measurements in the scheme share the same pulse timing, which is the
    standard single-shell PGSE case).  For multi-shell schemes, pass a
    per-measurement scheme or split manually.

    Parameters
    ----------
    scheme : JaxScheme or dict
        Scheme from ``encode_pgse`` / ``encode_ste`` / ``encode_ogse``.
        Must contain bvecs, delta, Delta, and either bvalues (JaxScheme)
        or b_values (dict).
    n_t : int
        Number of time points for the waveform grid.

    Returns
    -------
    wf : dmipy_sim.waveforms.Waveform
        PGSE waveform scaled to the target b-values.

    Notes
    -----
    G_magnitude is computed from the b-value, delta, Delta via:
        G = sqrt(b / (gamma² * delta² * (Delta - delta/3)))
    The waveform is subsequently re-scaled via set_b() so the actual
    b-values match the scheme exactly.

    For STE/OGSE schemes the conversion is approximate:
    - STE: delta/Delta extracted from dict and used for PGSE shape.
    - OGSE: delta=t_eff, Delta=t_eff are proxies — resulting waveform is
      a PGSE approximation of the effective diffusion time.
    """
    from dmipy_sim.waveforms import pgse, set_b

    # --- Extract fields from JaxScheme or dict ---
    from dmipy_design.jax_scheme_encoder import JaxScheme

    if isinstance(scheme, JaxScheme):
        b_values = np.array(scheme.bvalues, dtype=np.float64)
        bvecs = np.array(scheme.bvecs, dtype=np.float32)
        delta_arr = np.array(scheme.delta, dtype=np.float64)
        Delta_arr = np.array(scheme.Delta, dtype=np.float64)
    elif isinstance(scheme, dict):
        # dict from encode_ste / encode_ogse / encode_pgse
        b_values = np.array(
            scheme.get("b_values", scheme.get("bvalues")), dtype=np.float64
        )
        bvecs = np.array(scheme["bvecs"], dtype=np.float32)
        delta_arr = np.array(scheme["delta"], dtype=np.float64)
        Delta_arr = np.array(scheme["Delta"], dtype=np.float64)
    else:
        raise TypeError(
            f"scheme must be a JaxScheme or dict, got {type(scheme)}"
        )

    # Use the first measurement's timing for the waveform shape.
    # All measurements in a single-shell PGSE scheme share the same timing.
    delta_val = float(np.ravel(delta_arr)[0])
    Delta_val = float(np.ravel(Delta_arr)[0])

    # Clamp timing to avoid degenerate waveforms
    delta_val = max(delta_val, 1e-5)
    Delta_val = max(Delta_val, delta_val + 1e-5)

    # Compute G_magnitude from first b-value to seed the waveform shape.
    # set_b() will re-scale to the exact per-measurement b-values.
    b_ref = float(np.max(b_values))
    if b_ref <= 0.0:
        b_ref = 1e9  # fallback for b=0 measurements
    t_diff = Delta_val - delta_val / 3.0
    G_ref = np.sqrt(b_ref / (_GAMMA ** 2 * delta_val ** 2 * t_diff))

    wf_single = pgse(
        delta=delta_val,
        DELTA=Delta_val,
        G_magnitude=G_ref,
        bvecs=bvecs[:1],   # single measurement for shape; will tile below
        n_t=n_t,
    )

    # Tile to all measurements and apply per-measurement b-value scaling
    import jax.numpy as jnp
    n_meas = bvecs.shape[0]
    G_tiled = jnp.tile(wf_single.G, (n_meas, 1, 1))  # (n_meas, n_t, 3)

    # Re-scale gradient directions per measurement
    # G_tiled currently repeats bvecs[0] for all measurements; fix to per-measurement bvecs.
    # Rebuild G using per-measurement bvecs at the reference amplitude.
    from dmipy_sim.waveforms import Waveform
    import jax.numpy as jnp

    wf_all = pgse(
        delta=delta_val,
        DELTA=Delta_val,
        G_magnitude=G_ref,
        bvecs=bvecs,
        n_t=n_t,
    )
    # Scale each measurement to its target b-value
    wf = set_b(wf_all, b_values)
    return wf


def build_mc_forward_fn(
    geometry_factory,
    diffusivity: float,
    n_walkers: int = 10000,
    seed: int = 42,
    n_t: int = 500,
):
    """Build a dmipy-sim MC forward function with a factory geometry.

    The returned function maps (params_dict, scheme) → signal array.
    The geometry is re-instantiated per call via ``geometry_factory(params_dict)``,
    making this suitable for OED over geometry parameters (e.g., radius).

    Parameters
    ----------
    geometry_factory : callable(dict) -> geometry
        A geometry *factory* — callable that takes a params dict and returns
        a dmipy_sim geometry object.  For example::

            lambda p: Cylinder(radius=p['radius'], orientation=[0, 0, 1])

    diffusivity : float
        Diffusion coefficient in m²/s.
    n_walkers : int
        Number of Monte Carlo walkers.  More walkers → lower MC noise but
        slower evaluation.
    seed : int
        Fixed PRNG seed.  Held constant across forward calls so that
        finite-difference perturbations cancel MC noise to first order.
    n_t : int
        Number of time points in the waveform grid.

    Returns
    -------
    forward_fn : callable(params_dict, scheme) -> np.ndarray shape (N_meas,)
        ``params_dict`` : dict of geometry parameters (passed to factory).
        ``scheme``      : JaxScheme or dict with b_values/bvecs/delta/Delta.
        Returns float32 signal array, values in [0, 1].

    Notes
    -----
    The function is NOT JAX-differentiable (MC is stochastic and non-traced).
    Use with ``compute_fim_fd`` (finite differences), not ``compute_fim``
    (which requires JAX autodiff).

    Example
    -------
    ::

        from dmipy_sim.geometries import Cylinder
        from dmipy_design.mc_bridge import build_mc_forward_fn
        from dmipy_design.fim import compute_fim_fd
        from dmipy_design.jax_scheme_encoder import encode_pgse
        import jax.numpy as jnp
        import numpy as np

        bvecs = np.tile([1., 0., 0.], (20, 1))
        geom_factory = lambda p: Cylinder(radius=p['radius'], orientation=[0, 0, 1])
        mc_fwd = build_mc_forward_fn(geom_factory, diffusivity=1.7e-9, n_walkers=5000)

        u = jnp.array([1000e6, 0.013, 0.022])
        scheme = encode_pgse(u, jnp.array(bvecs))
        # params is a numpy array for compute_fim_fd compatibility;
        # the factory receives a dict built from it or the raw dict can be
        # passed directly to forward_fn.
        signals = mc_fwd({'radius': 2e-6}, scheme)
    """
    from dmipy_sim.core import simulate

    def forward_fn(params_dict, scheme):
        geom = geometry_factory(params_dict)
        wf = _scheme_to_sim_waveform(scheme, n_t=n_t)
        signals = simulate(
            n_walkers=n_walkers,
            diffusivity=diffusivity,
            waveform=wf,
            geometry=geom,
            seed=seed,
        )
        return np.array(signals, dtype=np.float32)

    return forward_fn


def build_mc_forward_fn_fixed(
    geometry,
    diffusivity: float,
    n_walkers: int = 10000,
    seed: int = 42,
    n_t: int = 500,
):
    """Build a dmipy-sim MC forward function with fixed (pre-instantiated) geometry.

    Faster than ``build_mc_forward_fn`` because the geometry object is
    created once and reused across all forward calls.  Suitable for OED
    over acquisition scheme parameters when the tissue geometry is known.

    The returned ``forward_fn(scheme)`` signature takes only the scheme —
    there are no free tissue parameters.  To use with ``compute_fim_fd``,
    wrap it as ``lambda theta, scheme: forward_fn(scheme)`` and pass an
    empty ``theta = jnp.zeros(0)``, or pass the wrapper directly.

    Parameters
    ----------
    geometry : geometry object
        Already-instantiated dmipy_sim geometry (e.g. ``Cylinder(...)``).
    diffusivity : float
        Diffusion coefficient in m²/s.
    n_walkers : int
        Number of Monte Carlo walkers.
    seed : int
        Fixed PRNG seed (constant across calls for FD consistency).
    n_t : int
        Number of waveform time points.

    Returns
    -------
    forward_fn : callable(scheme) -> np.ndarray shape (N_meas,)
        ``scheme`` : JaxScheme or dict with b_values/bvecs/delta/Delta.
        Returns float32 signal array, values in [0, 1].

    Notes
    -----
    NOT JAX-differentiable.  Use with ``compute_fim_fd``.

    For ``compute_fim_fd`` compatibility the wrapper signature is::

        lambda theta, scheme: forward_fn(scheme)

    so that ``theta`` (unused) can be a zero-length array and the FIM is
    computed over scheme space rather than parameter space.

    Example
    -------
    ::

        from dmipy_sim.geometries import Cylinder
        from dmipy_design.mc_bridge import build_mc_forward_fn_fixed
        from dmipy_design.fim import compute_fim_fd
        from dmipy_design.jax_scheme_encoder import encode_pgse
        import jax.numpy as jnp
        import numpy as np

        bvecs = np.tile([1., 0., 0.], (20, 1))
        geom = Cylinder(radius=2e-6, orientation=[0, 0, 1])
        mc_fwd = build_mc_forward_fn_fixed(geom, diffusivity=1.7e-9, n_walkers=5000)

        u = jnp.array([1000e6, 0.013, 0.022])
        scheme = encode_pgse(u, jnp.array(bvecs))

        # Direct use: signal for a given scheme
        signals = mc_fwd(scheme)

        # With compute_fim_fd (theta is unused, pass zero-length array):
        theta = jnp.zeros(0)
        FIM = compute_fim_fd(
            lambda theta, scheme: mc_fwd(scheme),
            theta, scheme, sigma=0.02
        )
    """
    from dmipy_sim.core import simulate

    def forward_fn(scheme):
        wf = _scheme_to_sim_waveform(scheme, n_t=n_t)
        signals = simulate(
            n_walkers=n_walkers,
            diffusivity=diffusivity,
            waveform=wf,
            geometry=geometry,
            seed=seed,
        )
        return np.array(signals, dtype=np.float32)

    return forward_fn
