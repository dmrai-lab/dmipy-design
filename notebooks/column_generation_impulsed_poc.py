"""
IMPULSED Proof-of-Concept: Column Generation OED for Ball + C4Cylinder.

Demonstrates the column generation OED framework finding a D-optimal
multi-shell acquisition protocol for a Ball + C4Cylinder (Van Gelderen GPA)
model, automatically discovering OGSE shells via the pricing problem.

Run:
    /home/rutger/dmipy-core/.venv/bin/python notebooks/column_generation_impulsed_poc.py

Output:
    - Per-iteration KW gap table
    - Final protocol: atom type, key parameters, allocated weight
"""

import numpy as np

# --- float64 must be enabled before any jax imports ---
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
GAMMA = 267513000.0   # rad/(s·T)

# ---------------------------------------------------------------------------
# 2. Prior distribution (Ball + C4Cylinder, P=4 parameters)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2024)
M = 128   # prior samples

vf_ball    = rng.uniform(0.30, 0.70, size=(M, 1))
lambda_iso = rng.uniform(1.0e-9, 3.0e-9, size=(M, 1))   # m²/s
lambda_par = rng.uniform(1.5e-9, 2.5e-9, size=(M, 1))   # m²/s
diameter   = rng.uniform(2.0e-6, 20.0e-6, size=(M, 1))  # m

prior_samples = jnp.array(
    np.concatenate([vf_ball, lambda_iso, lambda_par, diameter], axis=1),
    dtype=jnp.float64,
)
print(f"Prior: M={M} samples, P={prior_samples.shape[1]} parameters")
print(f"  vf_ball:    [{vf_ball.min():.2f}, {vf_ball.max():.2f}]")
print(f"  lambda_iso: [{lambda_iso.min():.2e}, {lambda_iso.max():.2e}] m²/s")
print(f"  lambda_par: [{lambda_par.min():.2e}, {lambda_par.max():.2e}] m²/s")
print(f"  diameter:   [{diameter.min()*1e6:.1f}, {diameter.max()*1e6:.1f}] µm")
print()

# ---------------------------------------------------------------------------
# 3. Gradient directions (30-direction isotropic shell)
# ---------------------------------------------------------------------------
from dmipy_design.optimizers.pricing_problem import BVECS_30, CONNECTOM_3T
bvecs_30 = jnp.array(BVECS_30, dtype=jnp.float64)

# ---------------------------------------------------------------------------
# 4. Initial atom: one PGSE shell at b=1000 s/mm²
# ---------------------------------------------------------------------------
from dmipy_design.jax_scheme_encoder import encode_pgse_shell
from dmipy_design.optimizers.column_generation import Atom

b0      = 1000e6   # s/m²  (= 1000 s/mm² = 1e9 s/m², typical dMRI)
delta0  = 0.020    # 20 ms
Delta0  = 0.040    # 40 ms

scheme0 = encode_pgse_shell(b0, delta0, Delta0, bvecs_30)
atom0   = Atom(
    type='pgse',
    params={'b': b0, 'delta': delta0, 'Delta': Delta0},
    scheme=scheme0,
)
print(f"Initial atom: PGSE  b={b0*1e-6:.0f} s/mm²  delta={delta0*1e3:.0f}ms  Delta={Delta0*1e3:.0f}ms")
print()

# ---------------------------------------------------------------------------
# 5. Forward function
# ---------------------------------------------------------------------------
from dmipy_design.analytical_forward import ball_c4cylinder_forward

# ---------------------------------------------------------------------------
# 6. Column generation OED
# ---------------------------------------------------------------------------
from dmipy_design.optimizers.column_generation import column_generation_oed

sigma      = 0.05   # noise std (SNR ~20 at b=0)
max_iter   = 8
n_restarts = 5      # pricing restarts per waveform type

print("=" * 70)
print("Column Generation OED  —  Ball + C4Cylinder (GPA/IMPULSED)")
print("Waveform types: PGSE + OGSE  |  Hardware: Connectom 3T (300 mT/m)")
print(f"sigma={sigma},  max_iter={max_iter},  n_pricing_restarts={n_restarts}")
print("=" * 70)
print()

result = column_generation_oed(
    forward_fn=ball_c4cylinder_forward,
    prior_samples=prior_samples,
    sigma=sigma,
    waveform_types=['pgse', 'ogse'],
    initial_atoms=[atom0],
    max_iter=max_iter,
    reduced_cost_tol=0.05,
    n_pricing_restarts=n_restarts,
    verbose=True,
    hardware=CONNECTOM_3T,
)

# ---------------------------------------------------------------------------
# 7. Summary table
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Final protocol")
print("=" * 70)
print(f"{'#':<4} {'Type':<6} {'Weight':>8}  Parameters")
print("-" * 70)
for i, (atom, w) in enumerate(zip(result.atoms, result.weights)):
    params_str = "  ".join(
        f"{k}={v:.4g}" for k, v in atom.params.items() if k != 'type'
    )
    # Add derived info
    if atom.type == 'pgse':
        b_mm2 = atom.params['b'] * 1e-6
        extra = f"  [b={b_mm2:.0f} s/mm²]"
    elif atom.type == 'ogse':
        f_hz = atom.params.get('f', atom.params.get('freq', '?'))
        G_tm = atom.params.get('G', '?')
        t_eff = 1.0 / (4.0 * f_hz) if isinstance(f_hz, float) else '?'
        b_mm2 = atom.params.get('b', 0) * 1e-6
        extra = f"  [t_eff={t_eff*1e3:.1f}ms, b={b_mm2:.3f} s/mm²]" if isinstance(t_eff, float) else ''
    else:
        extra = ''
    print(f"{i:<4} {atom.type:<6} {w:>8.4f}  {params_str}{extra}")

print("-" * 70)
print(f"Total weight: {result.weights.sum():.6f}")
print(f"Final -log det F: {result.final_obj:.4f}")

# Count waveform types
from collections import Counter
type_counts = Counter(a.type for a in result.atoms)
print(f"Active atoms: {dict(type_counts)}")
print()

if 'ogse' in type_counts:
    print("SUCCESS: OGSE atom discovered by column generation pricing.")
else:
    print("NOTE: No OGSE atom in final active set (may need more iterations).")
    print("Checking history for OGSE discovery...")
    ogse_iters = [h['iter'] for h in result.history if h.get('new_atom_type') == 'ogse']
    if ogse_iters:
        print(f"  OGSE was added at iteration(s): {ogse_iters}")
        print("  (pruned due to low weight at convergence)")
    else:
        print("  No OGSE added in any iteration.")
