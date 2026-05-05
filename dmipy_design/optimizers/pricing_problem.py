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
from scipy.optimize import minimize as scipy_minimize
from dataclasses import dataclass

from ..fim import compute_fim_averaged
from ..jax_scheme_encoder import encode_pgse_shell, encode_ogse_shell, encode_ste_shell, encode_pgste_shell
from ..waveform_builders import build_pgse_G, build_ogse_G, build_pgste_G, build_ste_G

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
    b_max : float
        Hard upper bound on b-value (s/m²).  Acts as a safety clip in
        decode_pgse / decode_ste to prevent extreme b-values that would
        place the signal below the thermal noise floor.  Default 10 000 s/mm²
        (= 10_000e6 s/m²) — rarely binding once T2 attenuation is included
        in the forward model, but prevents numerical nonsense in corner cases.
    """
    name: str
    g_max: float
    pgse_delta_min: float
    ogse_f_max: float
    b_max: float = 10_000e6   # s/m²  (= 10 000 s/mm²)


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
    b     = jnp.minimum(b, hardware.b_max)   # safety clip — rarely binding with T2 model
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


def decode_pgste(v, hardware: HardwarePreset = CLINICAL_3T):
    """v in [0,1]^3 -> (b, delta, Delta) for PGSTE — identical physics to PGSE.

    PGSTE uses the same (G, delta, Delta) parameterization as PGSE; the
    difference is the SNR model: TE = 2·delta (short) instead of 2·Delta,
    with T1 relaxation during the mixing time TM = Delta − delta.

    Parameters
    ----------
    v : array-like, shape (3,)
        Normalised parameters in [0, 1].
    hardware : HardwarePreset

    Returns
    -------
    (b, delta, Delta) as JAX scalars.
    """
    return decode_pgse(v, hardware)


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
# Waveform G-array decoder helper
# ---------------------------------------------------------------------------

def _decode_to_G(v, wtype: str, hardware: HardwarePreset, bvecs_jax, dt_traj: float):
    """Decode normalised pricing parameters v -> (G_array, dt_wf).

    Builds the (n_meas, n_t, 3) JAX gradient waveform from the normalised
    parameter vector used by the pricing problem.  dt_wf == dt_traj so no
    resampling is needed.

    Parameters
    ----------
    v : array-like (JAX traced), shape (n_p,)
        Normalised parameters in [0, 1].
    wtype : str
        Waveform type ('pgse', 'ogse', 'ste', 'pgste').
    hardware : HardwarePreset
        Scanner hardware limits (for decode_* functions).
    bvecs_jax : jnp.ndarray, shape (n_dirs, 3)
        Gradient directions.
    dt_traj : float
        Time step to use for constructing the G array (seconds).

    Returns
    -------
    G : jnp.ndarray, shape (n_meas, n_t, 3)
    dt_wf : float  (== dt_traj)
    """
    if wtype in ('pgse', 'pgste'):
        G_amp_scalar = hardware.g_max * 0.1 + v[0] * hardware.g_max * 0.9
        delta_val    = hardware.pgse_delta_min + v[1] * (0.060 - hardware.pgse_delta_min)
        ratio_val    = 1.1 + v[2] * 1.9
        Delta_val    = delta_val * ratio_val
        # Use concrete Python floats for n_t calculation (not traced)
        # delta_val and Delta_val are JAX scalars — T_total must be concrete
        # We compute T_total from decoded concrete float values at trace time.
        # Since v is [0,1] we can bound: T_total_max = 0.060 * 3.0 + 0.060 = 0.24 s
        # Use a fixed T_total that covers all possible (delta, Delta) combinations.
        T_total_fixed = 0.060 * 3.0 + 0.060  # 0.24 s — conservative upper bound
        if wtype == 'pgse':
            G_arr = build_pgse_G(G_amp_scalar, delta_val, Delta_val, bvecs_jax,
                                 dt_traj, T_total=T_total_fixed)
        else:
            G_arr = build_pgste_G(G_amp_scalar, delta_val, Delta_val, bvecs_jax,
                                  dt_traj, T_total=T_total_fixed)
    elif wtype == 'ogse':
        f_val    = 10.0 + v[0] * (hardware.ogse_f_max - 10.0)
        G_amp_scalar = hardware.g_max * 0.1 + v[1] * hardware.g_max * 0.9
        # T_total = 1/f_min = 1/10 = 0.1 s upper bound
        T_total_fixed = 1.0 / 10.0
        G_arr = build_ogse_G(G_amp_scalar, f_val, bvecs_jax, dt_traj,
                             T_total=T_total_fixed)
    elif wtype == 'ste':
        G_amp_scalar = hardware.g_max * 0.1 + v[0] * hardware.g_max * 0.9
        delta_val    = hardware.pgse_delta_min + v[1] * (0.060 - hardware.pgse_delta_min)
        ratio_val    = 1.1 + v[2] * 1.9
        Delta_val    = delta_val * ratio_val
        T_total_fixed = 6.0 * 0.060 * 3.0  # covers STE staggered pairs
        G_arr = build_ste_G(G_amp_scalar, Delta_val, bvecs_jax, dt_traj,
                            T_total=T_total_fixed)
    else:
        raise ValueError(f"Unknown waveform type in _decode_to_G: '{wtype}'")

    return G_arr, dt_traj


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
    substrate_bank=None,
    mc_bias_weight: float = 0.0,
    fim_weight: float = 1.0,
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
    wtype : str   'pgse', 'ogse', 'ste', or 'pgste'
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
    substrate_bank : SubstrateBank or None
        If provided (and mc_bias_weight > 0), the MC bias term is added to
        the pricing objective:
            rc(v) = fim_weight * FIM_rc(v) - mc_bias_weight * B(v)
        When None or mc_bias_weight == 0, behaviour is identical to the
        unmodified system.
    mc_bias_weight : float
        β coefficient for the MC bias penalty. Default 0.0 (pure FIM).
    fim_weight : float
        α coefficient for the FIM term. Default 1.0.

    Returns
    -------
    best_rc : float
        Maximum reduced cost found (FIM-only component, for KW gap tracking).
    best_params : dict
        Decoded physical parameters of the best atom.
    best_scheme : JaxScheme
        The shell JaxScheme for the best atom.

    Notes
    -----
    When substrate_bank is provided, the MC bias computation runs outside the
    vmap (one call per pricing restart, not per vmap axis). This avoids
    replicating the large trajectory arrays. The combined objective is:

        rc_combined(v) = fim_weight * trace(F_inv @ F_new(v))
                       - mc_bias_weight * B(v; substrate_bank)

    The returned best_rc is the FIM-only component (for KW gap convergence
    tracking per the plan: KW gap is certified on the FIM term only).
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

    elif wtype == 'pgste':
        n_p = 3

        def rc_fn(v):
            b, delta, Delta = decode_pgste(v, hardware)
            scheme = encode_pgste_shell(b, delta, Delta, bvecs_jax)
            F_new = compute_fim_averaged(forward_fn, prior_samples, scheme, sigma)
            return jnp.trace(F_inv @ F_new)

    else:
        raise ValueError(f"Unknown waveform type: '{wtype}'")

    # --- Two-term combined objective (MC bias regularization) ---
    # When substrate_bank is None or mc_bias_weight == 0, the FIM-only rc_fn
    # is used unchanged (zero overhead, backward-compatible).
    #
    # When substrate_bank is provided, we WRAP rc_fn to add the bias term.
    # The MC bias computation runs in the un-vmapped Python closure (not
    # inside the vmap) to avoid replicating the large trajectory arrays.
    # This is safe because vmap is over restarts (u0 vectors), not over the
    # objective evaluations. Each restart independently calls the combined
    # objective — the closure captures substrate_bank by reference.
    fim_rc_fn = rc_fn   # keep a reference to the FIM-only function

    if substrate_bank is not None and mc_bias_weight > 0.0:
        _fim_rc_fn = rc_fn  # alias for closure
        dt_traj = float(substrate_bank.entries[0].dt_traj) if substrate_bank.entries else 1e-4

        def rc_fn(v):
            fim_val = _fim_rc_fn(v)
            G_arr, dt_wf = _decode_to_G(v, wtype, hardware, bvecs_jax, dt_traj)
            # Build a simple scheme for the analytical forward call
            # Use the decoded physical parameters to construct a JaxScheme
            if wtype in ('pgse', 'pgste'):
                _b, _delta, _Delta = decode_pgse(v, hardware)
                if wtype == 'pgse':
                    _scheme = encode_pgse_shell(_b, _delta, _Delta, bvecs_jax)
                else:
                    _scheme = encode_pgste_shell(_b, _delta, _Delta, bvecs_jax)
            elif wtype == 'ogse':
                _f, _G, _b = decode_ogse(v, hardware)
                _scheme = encode_ogse_shell(_f, _G, bvecs_jax)
            elif wtype == 'ste':
                _b, _delta, _Delta = decode_ste(v, hardware)
                _scheme = encode_ste_shell(_b, _delta, _Delta, bvecs_jax)
            else:
                _scheme = None
            bias_val = substrate_bank.compute_bias_jax(G_arr, dt_wf, forward_fn, _scheme, sigma)
            return fim_weight * fim_val - mc_bias_weight * bias_val.astype(jnp.float64)

    # Sigmoid reparameterization: optimize over unbounded u; v = sigmoid(u)
    def neg_rc_transformed(u):
        v = _sigmoid(u)
        return -rc_fn(v)

    # JIT-compile only the FIM part (no trajectory arrays in closure → no 12 GB constants).
    # Bias (MC trajectory replay) is called outside JIT: the trajectory arrays are in the
    # substrate_bank closure and would otherwise be captured as XLA compile-time constants,
    # triggering a "12.01 GB constants captured" warning and slow recompilation on each call.
    # scipy L-BFGS-B with numerical FD gradients avoids jit(grad(vmap(jacfwd))) which
    # triggers CUDA graph capture failures on some GPU configurations (e.g. L40S).
    _fim_rc_jit = jax.jit(fim_rc_fn)   # JIT: small FIM matrices, no large arrays

    # Decode v → G_array once per call (outside JIT); also builds the analytical scheme.
    dt_traj = float(substrate_bank.entries[0].dt_traj) if (
        substrate_bank is not None and substrate_bank.entries
    ) else 1e-4

    def _combined_rc(v_jnp):
        """Combined reduced cost: FIM term + MC bias term (bias outside JIT)."""
        fim_val = float(_fim_rc_jit(v_jnp))
        if substrate_bank is None or mc_bias_weight == 0.0:
            return fim_val
        G_arr, dt_wf = _decode_to_G(v_jnp, wtype, hardware, bvecs_jax, dt_traj)
        if wtype in ('pgse', 'pgste'):
            _b, _delta, _Delta = decode_pgse(v_jnp, hardware)
            _scheme = encode_pgse_shell(_b, _delta, _Delta, bvecs_jax) if wtype == 'pgse' \
                      else encode_pgste_shell(_b, _delta, _Delta, bvecs_jax)
        elif wtype == 'ogse':
            _f, _G, _b = decode_ogse(v_jnp, hardware)
            _scheme = encode_ogse_shell(_f, _G, bvecs_jax)
        elif wtype == 'ste':
            _b, _delta, _Delta = decode_ste(v_jnp, hardware)
            _scheme = encode_ste_shell(_b, _delta, _Delta, bvecs_jax)
        else:
            _scheme = None
        bias_val = float(
            substrate_bank.compute_bias_jax(G_arr, dt_wf, forward_fn, _scheme, sigma)
        )
        return fim_weight * fim_val - mc_bias_weight * bias_val

    def _scipy_obj(v_np):
        v = jnp.array(v_np, dtype=jnp.float64)
        return -_combined_rc(v)   # scipy minimises; negate for maximisation

    # Sample initial points in [0.05, 1.0]^n (bounded space for L-BFGS-B)
    v0_batch = rng.uniform(0.05, 0.95, size=(n_restarts, n_p))

    v_opts_list  = []
    rc_vals_list = []
    for i in range(n_restarts):
        res = scipy_minimize(
            _scipy_obj, v0_batch[i],
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * n_p,
            options={'maxiter': lbfgs_maxiter, 'ftol': 1e-9, 'gtol': 1e-6},
        )
        v_opt_i = jnp.array(np.clip(res.x, 0.0, 1.0), dtype=jnp.float64)
        rc_val_i = jnp.array(_combined_rc(v_opt_i), dtype=jnp.float64)
        v_opts_list.append(v_opt_i)
        rc_vals_list.append(rc_val_i)
    v_opts  = jnp.stack(v_opts_list)
    rc_vals = jnp.stack(rc_vals_list)

    # Pick best restart (ranked by combined objective rc_fn, which may include bias)
    best_idx = int(jnp.argmax(rc_vals))
    best_v   = np.array(v_opts[best_idx])
    best_v   = np.clip(best_v, 0.0, 1.0)

    # Return the FIM-only reduced cost for KW gap tracking (plan §Convergence criterion)
    best_rc  = float(_fim_rc_jit(jnp.array(best_v, dtype=jnp.float64)))

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

    elif wtype == 'pgste':
        b, delta, Delta = decode_pgste(best_v, hardware)
        G = float(hardware.g_max * 0.1 + best_v[0] * hardware.g_max * 0.9)
        best_params = {
            'type': 'pgste',
            'b': float(b),
            'G': G,
            'delta': float(delta),
            'Delta': float(Delta),
            'TM': float(Delta) - float(delta),
        }
        best_scheme = encode_pgste_shell(b, delta, Delta, bvecs_jax_np)

    return best_rc, best_params, best_scheme
