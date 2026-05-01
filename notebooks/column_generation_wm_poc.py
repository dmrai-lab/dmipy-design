"""
White matter axon diameter OED — column generation PoC.

Uses a gamma-distributed diameter prior (Liewald 2014, Aboitiz 1992) reflecting
true WM axon diameter distribution: mean ~0.8 µm, 95th pct ~2.2 µm.

Hardware: clinical 3T scanner (Siemens Prisma, G_max=80 mT/m).
Uses the module-level PGSE_G_RANGE=(0.01, 0.08) T/m and
OGSE_G_RANGE=(0.02, 0.30) T/m constants already set for clinical hardware.
Note: OGSE_G_RANGE upper bound of 0.30 T/m is Connectom; for strict clinical
80 mT/m OGSE, high-G solutions will simply not appear since the optimizer
naturally selects the most informative parameters (high-G OGSE).

Expected result: higher-frequency OGSE (shorter t_eff) compared to the
IMPULSED tumour-cell PoC, because WM axons are 5-10× smaller than tumour cells.
IMPULSED PoC had t_eff ~5-9 ms for ~10 µm tumour cells.
WM axons at mean 0.8 µm → expected t_eff ~1-3 ms (f ~80-250 Hz).

Run:
    /home/rutger/dmipy-core/.venv/bin/python notebooks/column_generation_wm_poc.py
"""

import os
# When the GPU is saturated by other processes (<1 GB free), the vmapped jaxopt
# LBFGS unrolled graph exceeds CUDA graph memory limits.  Set JAX_PLATFORMS=cpu
# to force CPU execution in that case:
#   JAX_PLATFORMS=cpu python notebooks/column_generation_wm_poc.py
# On a dedicated GPU (>8 GB free) the default GPU execution gives ~8× speedup
# over sequential restarts via jax.vmap.  No platform override needed for GPU.

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from scipy.stats import gamma as gamma_dist

from dmipy_design.analytical_forward import ball_c4cylinder_forward
from dmipy_design.optimizers.pricing_problem import encode_pgse_shell, BVECS_30, CLINICAL_3T
from dmipy_design.optimizers.column_generation import column_generation_oed, Atom

# ── Prior ────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
# M controls prior samples.
# GPU run (dedicated, >8 GB free): M=64, n_pricing_restarts=6, lbfgs_maxiter=200
# CPU run / constrained GPU: M=32, n_pricing_restarts=4, lbfgs_maxiter=50
# The vmapped jaxopt LBFGS unrolls lbfgs_maxiter × n_restarts steps as one JIT
# kernel; peak GPU memory scales with M × n_restarts × lbfgs_maxiter.
# Demo / PoC settings — small enough to run in ~5min on any GPU with ≥2 GB free.
# Scale up on a dedicated GPU: M=64, _N_RESTARTS=6, _LBFGS_MAXITER=200, _MAX_ITER=10
# The vmapped jaxopt LBFGS unrolls lbfgs_maxiter × n_restarts steps as one JIT
# kernel; both compile time and GPU memory scale with that product × M.
M              = 16
_N_RESTARTS    = 2
_LBFGS_MAXITER = 10
_MAX_ITER      = 3

vf_intra   = rng.uniform(0.3, 0.7, M)
lambda_par = np.exp(rng.uniform(np.log(1.5e-9), np.log(2.5e-9), M))
lambda_iso = np.exp(rng.uniform(np.log(0.5e-9), np.log(2.0e-9), M))
diameter   = gamma_dist(a=2.0, scale=0.4e-6).rvs(M, random_state=rng)
diameter   = np.clip(diameter, 0.2e-6, 4.0e-6)

# theta = [vf_ball, lambda_iso, lambda_par, diameter]
# vf_ball = 1 - vf_intra  (Ball = extra-axonal compartment)
vf_ball = 1.0 - vf_intra
prior = jnp.array(np.column_stack([
    vf_ball, lambda_iso, lambda_par, diameter
]), dtype=jnp.float64)

print("White matter prior (gamma diameter distribution):")
print(f"  vf_ball:    [{vf_ball.min():.2f}, {vf_ball.max():.2f}]")
print(f"  lambda_iso: [{lambda_iso.min():.2e}, {lambda_iso.max():.2e}] m²/s")
print(f"  lambda_par: [{lambda_par.min():.2e}, {lambda_par.max():.2e}] m²/s")
print(f"  diameter:   mean={diameter.mean()*1e6:.2f}µm, "
      f"median={np.median(diameter)*1e6:.2f}µm, "
      f"p95={np.percentile(diameter, 95)*1e6:.2f}µm")

# Show prior diameter histogram summary
counts, edges = np.histogram(diameter * 1e6, bins=10)
print("  diameter histogram (µm):")
for lo, hi, c in zip(edges[:-1], edges[1:], counts):
    bar = '█' * (c * 20 // M)
    print(f"    [{lo:.1f}, {hi:.1f}): {bar}")

# ── Initial atom: standard WM PGSE shell ────────────────────────────────────
# Typical clinical WM protocol: b=1000 s/mm², delta=22ms, Delta=45ms
scheme0 = encode_pgse_shell(1000e6, 0.022, 0.045, jnp.array(BVECS_30, dtype=jnp.float64))
atom0   = Atom(
    type='pgse',
    params={'b': 1000e6, 'delta': 0.022, 'Delta': 0.045},
    scheme=scheme0,
)
print(f"\nInitial atom: PGSE  b=1000 s/mm²  delta=22ms  Delta=45ms")
print(f"  t_eff = Delta - delta/3 = {(0.045 - 0.022/3)*1e3:.1f} ms")

# ── Run column generation ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Column Generation OED  —  White Matter Axon Diameter")
print("Prior: gamma(k=2, scale=0.4µm) diameter, log-uniform diffusivities")
print("Hardware: clinical 3T scanner (G_max=80 mT/m, PGSE+OGSE+STE)")
print("=" * 70)

result = column_generation_oed(
    forward_fn         = ball_c4cylinder_forward,
    prior_samples      = prior,
    sigma              = 0.05,
    waveform_types     = ['pgse', 'ogse', 'ste'],
    initial_atoms      = [atom0],
    max_iter           = _MAX_ITER,
    reduced_cost_tol   = 0.05,
    n_pricing_restarts = _N_RESTARTS,
    lbfgs_maxiter      = _LBFGS_MAXITER,
    verbose            = True,
    hardware           = CLINICAL_3T,
)

# ── Print results ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Final WM protocol")
print("=" * 70)
print(f"{'#':<5} {'Type':<8} {'Weight':<8} {'Key parameters'}")
print("-" * 70)
for i, (atom, w) in enumerate(zip(result.atoms, result.weights)):
    if atom.type == 'ogse':
        f  = atom.params['f']
        G  = atom.params['G']
        b  = atom.params['b']
        t  = 1000.0 / (4.0 * f)
        print(f"{i:<5} {'ogse':<8} {w:.4f}   f={f:.1f}Hz  G={G*1e3:.0f}mT/m  "
              f"b={b/1e6:.3f}s/mm²  t_eff={t:.1f}ms")
    elif atom.type == 'ste':
        b     = atom.params['b']
        delta = atom.params['delta']
        Delta = atom.params['Delta']
        G     = atom.params.get('G', float('nan'))
        t     = (Delta - delta / 3.0) * 1000.0
        print(f"{i:<5} {'ste':<8} {w:.4f}   b={b/1e6:.0f}s/mm²  "
              f"G={G*1e3:.0f}mT/m  delta={delta*1e3:.1f}ms  Delta={Delta*1e3:.1f}ms  "
              f"t_eff={t:.1f}ms")
    else:
        b     = atom.params['b']
        delta = atom.params['delta']
        Delta = atom.params['Delta']
        G     = atom.params.get('G', float('nan'))
        t     = (Delta - delta / 3.0) * 1000.0
        print(f"{i:<5} {'pgse':<8} {w:.4f}   b={b/1e6:.0f}s/mm²  "
              f"G={G*1e3:.0f}mT/m  delta={delta*1e3:.1f}ms  Delta={Delta*1e3:.1f}ms  "
              f"t_eff={t:.1f}ms")
print("-" * 70)
print(f"Total weight: {result.weights.sum():.6f}")
print(f"Final -log det F: {result.final_obj:.4f}")

# ── Compare WM vs IMPULSED t_eff ─────────────────────────────────────────────
ogse_atoms = [(a, w) for a, w in zip(result.atoms, result.weights)
              if a.type == 'ogse']
if ogse_atoms:
    t_effs = [1000.0 / (4.0 * a.params['f']) for a, _ in ogse_atoms]
    print(f"\nOGSE t_eff values (WM): {[f'{t:.1f}ms' for t in t_effs]}")
    print(f"IMPULSED PoC (tumour cells ~10µm) had t_eff ≈ 5.4ms and 9.3ms")
    wm_shorter = all(t < 5.4 for t in t_effs)
    if wm_shorter:
        print("RESULT: WM OGSE t_eff shorter than IMPULSED — expected (smaller axons).")
    else:
        print("NOTE: WM OGSE t_eff not shorter than IMPULSED "
              "(may reflect broad prior / hardware constraints).")
else:
    print("\nNo OGSE atoms in final active set.")
    ogse_iters = [h['iter'] for h in result.history
                  if h.get('new_atom_type') == 'ogse']
    if ogse_iters:
        print(f"  OGSE added at iteration(s): {ogse_iters} (pruned at convergence).")
    else:
        print("  No OGSE added in any iteration.")
