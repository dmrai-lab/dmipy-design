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

Optimizer backend
-----------------
solve_pricing uses jaxopt.LBFGS with a sigmoid reparameterization to handle
box constraints [0,1]^n.  All n_restarts are vmapped and JIT-compiled into a
single GPU kernel — no sequential Python loop over restarts.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
from dataclasses import dataclass

from ..fim import compute_fim_averaged
from ..jax_scheme_encoder import encode_pgse_shell, encode_ogse_shell, encode_ste_shell

GAMMA = 267513000.0  # rad/(s·T)
N_DIRS = 30


# ---------------------------------------------------------------------------
# Scanner hardware presets
# ---------------------------------------------------------------------------

@dataclass
class HardwarePreset:
    """Scanner hardware limits that apply to both PGSE and OGSE.

    Attributes
    ----------
    name : str
        Human-readable scanner name.
    g_max : float
        Maximum gradient amplitude (T/m), applied to both PGSE and OGSE.
    pgse_delta_min : float
        Minimum PGSE pulse duration (s); enforced by gradient risetime.
    ogse_f_max : float
        Maximum OGSE frequency (Hz).
    """
    name: str
    g_max: float
    pgse_delta_min: float
    ogse_f_max: float


CLINICAL_3T  = HardwarePreset('clinical_3T',  g_max=0.08, pgse_delta_min=0.015, ogse_f_max=300.0)
CONNECTOM_3T = HardwarePreset('connectom_3T', g_max=0.30, pgse_delta_min=0.005, ogse_f_max=500.0)

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
# Atom parameter bounds (physical units) — legacy module-level constants
# ---------------------------------------------------------------------------
# These are kept for backward compatibility but the preferred interface is
# to pass a HardwarePreset to decode_pgse / decode_ogse.
PGSE_G_RANGE           = (0.01,  0.08)    # T/m  (10–80 mT/m, clinical scanner)
PGSE_DELTA_RANGE       = (0.015, 0.060)   # s    (≥15ms: clinical gradient risetime limit)
PGSE_DELTA_RATIO_RANGE = (1.1,   3.0)     # Delta / delta

OGSE_F_RANGE = (10.0,  500.0)   # Hz
OGSE_G_RANGE = (0.02,  0.30)    # T/m (≤ 300 mT/m Connectom limit)


# ---------------------------------------------------------------------------
# Normalised decode helpers
# ---------------------------------------------------------------------------

def decode_pgse(v, hardware: HardwarePreset = CLINICAL_3T):
    """v in [0,1]^3 -> (b, delta, Delta) with b DERIVED from (G, delta, Delta).

    Parameterization: v = [G_norm, delta_norm, ratio_norm]
      G     = g_max/10 + v[0] * g_max * 0.9            (T/m)
      delta = pgse_delta_min + v[1] * (0.060 - pgse_delta_min)  (s)
      Delta = delta * (1.1 + v[2] * 1.9)               (Delta/delta in [1.1, 3.0])
      b     = gamma^2 * G^2 * delta^2 * (Delta - delta/3)  (derived)

    The hardware parameter is a Python object used at trace time (not a traced
    JAX value) — it acts as a static compile-time constant, making this safe to
    use inside JAX-traced closures.

    Parameters
    ----------
    v : array-like, shape (3,)
        Normalised parameters in [0, 1].
    hardware : HardwarePreset
        Scanner hardware limits.  Default: CLINICAL_3T (80 mT/m).

    Returns
    -------
    (b, delta, Delta) as JAX scalars.

    JAX-traceable: v may be a jnp array (traced) or np array (concrete).
    """
    G     = hardware.g_max * 0.1 + v[0] * hardware.g_max * 0.9
    delta = hardware.pgse_delta_min + v[1] * (0.060 - hardware.pgse_delta_min)
    ratio = 1.1 + v[2] * 1.9    # Delta/delta in [1.1, 3.0]
    Delta = delta * ratio
    b     = GAMMA**2 * G**2 * delta**2 * (Delta - delta / 3.0)
    return b, delta, Delta


def decode_ogse(v, hardware: HardwarePreset = CLINICAL_3T):
    """v in [0,1]^2 -> (f, G, b) with b derived from physics.

    Parameters
    ----------
    v : array-like, shape (2,)
        Normalised parameters in [0, 1].
    hardware : HardwarePreset
        Scanner hardware limits.  Default: CLINICAL_3T (80 mT/m).

    Returns
    -------
    (f, G, b) as JAX scalars.

    JAX-traceable: v may be a jnp array (traced) or np array (concrete).
    """
    f = 10.0 + v[0] * (hardware.ogse_f_max - 10.0)
    G = hardware.g_max * 0.1 + v[1] * hardware.g_max * 0.9
    t_eff = 1.0 / (4.0 * f)
    b = GAMMA**2 * G**2 * t_eff**3
    return f, G, b


def decode_ste(v, hardware: HardwarePreset = CLINICAL_3T):
    """v in [0,1]^3 -> (b, delta, Delta) for STE — identical physics to PGSE.

    STE uses the same (G, delta, Delta) parameterization as PGSE; the
    difference is the encoding type stored in the resulting JaxScheme.

    Parameters
    ----------
    v : array-like, shape (3,)
        Normalised parameters in [0, 1].
    hardware : HardwarePreset
        Scanner hardware limits.  Default: CLINICAL_3T (80 mT/m).

    Returns
    -------
    (b, delta, Delta) as JAX scalars.

    JAX-traceable: v may be a jnp array (traced) or np array (concrete).
    """
    return decode_pgse(v, hardware)


# ---------------------------------------------------------------------------
# Sigmoid / logit reparameterization for box constraints
# ---------------------------------------------------------------------------

def _sigmoid(u):
    """Sigmoid: maps unbounded u -> (0, 1)."""
    return 1.0 / (1.0 + jnp.exp(-u))


def _logit(v):
    """Logit: maps v in (0, 1) -> unbounded space (safe, clips away from 0/1)."""
    v = jnp.clip(v, 1e-6, 1.0 - 1e-6)
    return jnp.log(v / (1.0 - v))


# ---------------------------------------------------------------------------
# Pricing problem solver (jaxopt LBFGS + vmap over restarts)
# ---------------------------------------------------------------------------

def solve_pricing(
    forward_fn,
    prior_samples,
    sigma: float,
    F_total_inv_np: np.ndarray,
    wtype: str,
    n_restarts: int = 8,
    rng_seed: int = 0,
    bvecs: np.ndarray = BVECS_30,
    lbfgs_maxiter: int = 200,
    hardware: HardwarePreset = CLINICAL_3T,
):
    """Solve the pricing problem using jaxopt LBFGS vmapped over restarts.

    All n_restarts run in parallel on GPU as a single JIT-compiled kernel.
    Box constraints [0,1]^n are handled via sigmoid reparameterization:
    optimize in unbounded space u ∈ R^n, with v = sigmoid(u) ∈ (0, 1)^n.

    Parameters
    ----------
    forward_fn : callable (theta, JaxScheme) -> jnp.ndarray
    prior_samples : jnp.ndarray, shape (M, P)
    sigma : float
    F_total_inv_np : np.ndarray, shape (P, P)
    wtype : str   'pgse', 'ogse', or 'ste'
    n_restarts : int
        Number of random restarts; all run in parallel via jax.vmap.
    rng_seed : int
        Random seed for reproducibility.
    bvecs : np.ndarray, shape (n_dirs, 3)
    lbfgs_maxiter : int
        Maximum LBFGS iterations per restart.  All restarts × maxiter are
        unrolled by JAX, so lowering this reduces peak GPU memory.
        Default 200 (good quality); use 50–100 on memory-constrained GPUs.
    hardware : HardwarePreset
        Scanner hardware limits (g_max, delta_min, f_max).  PGSE and OGSE
        both use ``hardware.g_max``, ensuring hardware-consistent bounds.
        Default: CLINICAL_3T (80 mT/m).

    Returns
    -------
    best_rc : float
        Maximum reduced cost found.
    best_params : dict
        Decoded physical parameters of the best atom.
    best_scheme : JaxScheme
        The shell JaxScheme for the best atom.
    """
    F_inv = jnp.array(F_total_inv_np, dtype=jnp.float64)
    bvecs_jax = jnp.array(bvecs, dtype=jnp.float64)
    rng = np.random.default_rng(rng_seed)

    # Build rc_fn in [0,1]^n space for the requested waveform type.
    # The hardware object is captured from the enclosing scope — it is a
    # Python object, not a JAX traced value, so it acts as a static
    # compile-time constant inside the JAX-traced closures below.
    if wtype == 'pgse':
        n_p = 3

        def rc_fn(v):
            b, delta, Delta = decode_pgse(v, hardware)
            scheme = encode_pgse_shell(b, delta, Delta, bvecs_jax)
            F_new = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
            return jnp.trace(F_inv @ F_new)

    elif wtype == 'ogse':
        n_p = 2

        def rc_fn(v):
            f, G, _b = decode_ogse(v, hardware)
            scheme = encode_ogse_shell(f, G, bvecs_jax)
            F_new = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
            return jnp.trace(F_inv @ F_new)

    elif wtype == 'ste':
        n_p = 3

        def rc_fn(v):
            b, delta, Delta = decode_ste(v, hardware)
            scheme = encode_ste_shell(b, delta, Delta, bvecs_jax)
            F_new = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
            return jnp.trace(F_inv @ F_new)

    else:
        raise ValueError(f"Unknown waveform type: '{wtype}'")

    # Sigmoid reparameterization: optimize over unbounded u; v = sigmoid(u)
    def neg_rc_transformed(u):
        v = _sigmoid(u)
        return -rc_fn(v)

    # jaxopt LBFGS over transformed (unbounded) space
    lbfgs = jaxopt.LBFGS(fun=neg_rc_transformed, maxiter=lbfgs_maxiter, tol=1e-6)

    # Sample initial points in [0.05, 0.95]^n, transform to unbounded space
    v0_batch = jnp.array(
        rng.uniform(0.05, 0.95, size=(n_restarts, n_p)), dtype=jnp.float64
    )
    u0_batch = _logit(v0_batch)

    # vmap over restarts — all run in parallel as one JIT kernel
    def run_one(u0):
        result = lbfgs.run(u0)
        v_opt = _sigmoid(result.params)
        rc_val = rc_fn(v_opt)
        return v_opt, rc_val

    run_batch = jax.jit(jax.vmap(run_one))
    v_opts, rc_vals = run_batch(u0_batch)

    # Pick best restart
    best_idx = int(jnp.argmax(rc_vals))
    best_rc  = float(rc_vals[best_idx])
    best_v   = np.array(v_opts[best_idx])

    best_v = np.clip(best_v, 0.0, 1.0)
    bvecs_jax_np = jnp.array(bvecs, dtype=jnp.float64)

    if wtype == 'pgse':
        b, delta, Delta = decode_pgse(best_v, hardware)
        G = float(hardware.g_max * 0.1 + best_v[0] * hardware.g_max * 0.9)
        best_params = {
            'type': 'pgse',
            'b': float(b),
            'G': G,
            'delta': float(delta),
            'Delta': float(Delta),
        }
        best_scheme = encode_pgse_shell(b, delta, Delta, bvecs_jax_np)

    elif wtype == 'ogse':
        f, G, b = decode_ogse(best_v, hardware)
        best_params = {
            'type': 'ogse',
            'f': float(f),
            'G': float(G),
            'b': float(b),
        }
        best_scheme = encode_ogse_shell(f, G, bvecs_jax_np)

    elif wtype == 'ste':
        b, delta, Delta = decode_ste(best_v, hardware)
        G = float(hardware.g_max * 0.1 + best_v[0] * hardware.g_max * 0.9)
        best_params = {
            'type': 'ste',
            'b': float(b),
            'G': G,
            'delta': float(delta),
            'Delta': float(Delta),
        }
        best_scheme = encode_ste_shell(b, delta, Delta, bvecs_jax_np)

    return best_rc, best_params, best_scheme
