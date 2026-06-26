# dmipy-design

Optimal Experiment Design (OED) for diffusion MRI. Given a signal model and a
noise level, dmipy-design finds the acquisition parameters — b-values, gradient
directions, pulse timing — that minimise the Cramér-Rao lower bound (CRLB) on
parameter estimation error.

---

## Why OED matters

Scanner time is expensive. A typical research dMRI protocol spends 30–60 minutes
acquiring measurements that are informative enough to recover tissue parameters
at clinical SNR. A poorly designed protocol wastes that time acquiring
measurements whose signal lies in a low-sensitivity regime or is redundant with
measurements already taken.

OED provides a principled answer: given your model, your SNR, and your hardware,
what is the smallest set of measurements that achieves a target estimation
precision? In practice, well-designed protocols achieve the same parameter
precision with 20–40% fewer measurements compared to heuristically chosen
protocols (Alexander 2008, ActiveAx). That reduction translates directly to
fewer volumes, shorter scans, or higher SNR per measurement.

The field reference is Alexander (2008) "A general framework for experiment
design in diffusion MRI and its application in measuring direct tissue-microstructure
features in vivo." The ActiveAx protocol for axon diameter mapping was designed
with this framework — but the original implementation was MATLAB-based, not
maintained, and predates GPU computing, JAX autodiff, and B-tensor encoding.
dmipy-design is a modern, open, Python-native replacement.

---

## File map

```
dmipy_design/
  fim.py                    — Fisher Information Matrix: compute_fim, compute_fim_averaged, compute_fim_fd
  objectives.py             — D-optimal, A-optimal, E-optimal CRLB objectives; parameter_selective_crlb
  jax_scheme_encoder.py     — JaxScheme dataclass; encode_pgse, encode_lte, encode_ste, encode_ogse,
                              encode_multishell_pgse
  constraints.py            — HardwareConstraints (G_max, slew_rate_max, TE bounds), TimeConstraints
  acquisition_space.py      — AcquisitionSpace: defines the feasible protocol design space
  prior.py                  — sample_prior: draw uniform samples from dmipy-core model parameter ranges
  protocols.py              — Protocol dataclass: save/load .npz, to_dmipy_scheme()
  mc_bridge.py              — build_mc_forward_fn, build_mc_forward_fn_fixed: MC-backed FIM via
                              finite differences using dmipy-sim
  optimizers/
    gradient_based.py       — gradient_oed: L-BFGS-B + JAX autodiff, single-shell PGSE
    multishell.py           — multishell_oed: joint optimisation over multiple shells
    greedy_sequential.py    — greedy_add_measurement: add one measurement at a time (D-optimal gain)
    now.py                  — design_waveform_now: the DESIGN ORACLE. NumPy+SciPy SQP, max-b gradient
                              waveform at any b-tensor shape (LTE/PTE/STE/OGSE) with machine-precision
                              constraints (slew/amp/M1/M2/shape/Maxwell/spectral/PNS-SAFE/heat)
    timing.py               — SequenceTiming, encoding_spectrum (JAX-free, shared by both solvers)
    waveform_designer.py    — design_waveform: JAX augmented-Lagrangian solver. Differentiable, kept as
                              the simulator-in-the-loop CO-OPT engine (not a competing designer)
    stimulated_echo.py      — design_stimulated_echo: PGSTE timing front-end (routes through NOW)
```

Hardware/safety limits (gradient max/slew/raster, RF peak-B1/raster, IEC SAR/PNS) come from
the cited `dmipy_sim.sequences.scanner_constants` catalogue — the acquisition-side analogue of
dmipy-sim's `biophysical_constants`. The `benchmarks/now_coopt_pipeline.py` pipeline warm-starts a
differentiable-Bloch RF co-optimization from a NOW gradient (RF deliverability constraints applied).

```
tests/
  test_fim_traceability.py
  test_oed_optimizer.py
  test_multishell_oed.py
  test_prior.py
  test_protocol.py
  test_btensor_encoding.py
  test_mc_bridge.py
```

---

## Key concepts

### Fisher Information Matrix (FIM)

The Fisher Information Matrix quantifies how much information a set of
measurements carries about a set of parameters. For Gaussian noise with standard
deviation sigma, the FIM is:

```
F_ij(theta, scheme) = (1 / sigma^2) * sum_k  [dE_k / dtheta_i] * [dE_k / dtheta_j]
```

where `E_k` is the predicted signal at measurement `k` and `theta` are the tissue
parameters. The key insight is that the gradient `dE_k / dtheta_i` can be
computed exactly using JAX autodiff — no finite differences needed.

The FIM is symmetric positive semi-definite. If it is full rank, its inverse is
the CRLB matrix: any unbiased estimator has variance at least as large as the
corresponding diagonal of the CRLB matrix.

### Cramér-Rao Lower Bound (CRLB)

The CRLB gives a lower bound on the variance of any unbiased estimator:

```
Var[theta_hat_i] >= [F^{-1}]_ii
```

OED chooses the acquisition scheme to minimise a scalar summary of `F^{-1}`. The
choice of summary is the optimality criterion (see D/A/E below). The CRLB is
tight for maximum likelihood estimators at high SNR, making it an appropriate
design target for dMRI protocols where acquisition SNR is the dominant noise
source.

### D-optimality

D-optimality minimises `- log det(F)`, which is equivalent to minimising the
volume of the confidence ellipsoid in parameter space. It is invariant to linear
reparameterisation, which makes it appropriate when all parameters are equally
important and the model is not dominated by a single poorly-constrained direction.

```
objective_D = - log det(F)
```

Use D-optimality as the default criterion when you want to summarise overall
information content and do not have strong prior reasons to prioritise one
parameter over others.

### A-optimality

A-optimality minimises the trace of the CRLB matrix, which equals the sum of
individual parameter variances:

```
objective_A = trace(F^{-1}) = sum_i Var[theta_hat_i]
```

A-optimality penalises all parameter variances equally. It is more sensitive
than D-optimality to poorly constrained parameters (a single large variance
dominates the sum). Use A-optimality when you care about the average precision
across all parameters.

### E-optimality

E-optimality minimises the largest eigenvalue of the CRLB, which is the
worst-case variance over all linear combinations of parameters:

```
objective_E = max_eigenvalue(F^{-1})
```

E-optimality is minimax: it protects against the worst-case direction in
parameter space. Use it when you need a guaranteed lower bound on precision
regardless of which parameter combination matters most.

### FIM averaging over a prior

A single-point FIM `F(theta_0, scheme)` is optimal only at `theta_0`. In
practice, tissue parameters vary across the brain and across subjects. dmipy-design
averages the FIM over a prior distribution:

```
F_avg(scheme) = (1/M) * sum_m F(theta_m, scheme),   theta_m ~ p(theta)
```

This makes the optimised protocol robust across the expected range of tissue
parameters. The prior is sampled from the dmipy-core model's `parameter_ranges`
via `sample_prior()`.

---

## Quick start

Optimise a single-shell PGSE protocol for a two-compartment model (cylinder + ball):

```python
import jax.numpy as jnp
import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.jax.multicompartment_jax import build_mc_forward_fn

from dmipy_design.prior import sample_prior
from dmipy_design.constraints import HardwareConstraints
from dmipy_design.optimizers.gradient_based import gradient_oed

# 1. Build the signal model (same model used for fitting)
cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation()
ball = gaussian_models.G1Ball()
model = MultiCompartmentModel(models=[cylinder, ball])

# 2. Build the JAX-differentiable forward function
forward_fn = build_mc_forward_fn(model)

# 3. Sample prior distribution over tissue parameters
prior_samples = jnp.array(sample_prior(model, n_samples=256))

# 4. Define 30 fixed gradient directions
rng = np.random.default_rng(0)
bvecs_np = rng.standard_normal((30, 3))
bvecs_np /= np.linalg.norm(bvecs_np, axis=1, keepdims=True)
bvecs = jnp.array(bvecs_np)

# 5. Hardware constraints (standard 3T Prisma)
hw = HardwareConstraints(G_max=0.08)

# 6. Initial protocol vector: [b_value (s/m²), delta (s), Delta (s)]
u0 = jnp.array([1000e6, 0.020, 0.040])

# 7. Optimise
u_opt, crlb_opt = gradient_oed(
    forward_fn=forward_fn,
    prior_samples=prior_samples,
    bvecs=bvecs,
    sigma=0.02,
    u0=u0,
    constraints=hw,
    objective="D",
    max_iter=200,
)
print(f"Optimal b={u_opt[0]/1e6:.0f} s/mm², delta={u_opt[1]*1e3:.1f} ms, "
      f"Delta={u_opt[2]*1e3:.1f} ms")
print(f"D-optimal CRLB: {crlb_opt:.4f}")
```

For multi-shell optimisation, use `multishell_oed`. To add one measurement at a
time, use `greedy_add_measurement`.

---

## MC-bridge

For most tissue models based on the Gaussian Phase Approximation (GPA), the JAX
forward model is analytically differentiable and `compute_fim` / `compute_fim_averaged`
are the right tools.

The GPA breaks down at short diffusion times (delta, Delta approaching the
structural correlation length), at large gradient amplitudes, or for complex
geometries such as curved fibres, branching, or finite permeability barriers.
In these regimes, an analytical signal model does not exist or is known to be
inaccurate — but the dmipy-sim Monte Carlo engine can simulate the true signal.

The MC-bridge connects dmipy-sim to the FIM machinery via finite differences:

```python
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
theta = jnp.zeros(0)   # no free parameters for fixed geometry

FIM = compute_fim_fd(
    lambda theta, scheme: mc_fwd(scheme),
    theta, scheme, sigma=0.02
)
```

The seed is held fixed across forward calls (`seed=42` by default) so that the
same Monte Carlo walker trajectories are used for the unperturbed signal `E0`
and each perturbed signal `E_plus`. Because the MC noise is a function only of
the seed, it cancels to first order in the finite difference — the gradient
estimate is stable even when individual signal estimates are noisy.

Rule of thumb: `n_walkers >= 5000` for a cylinder at `b = 1000 s/mm²`. Use
`compute_fim_fd` (not `compute_fim`) for the MC-bridge because MC is not
JAX-traceable.

---

## Why not MATLAB / Camino?

**MATLAB ActiveAx (Alexander 2008)**: The original OED tool for dMRI. Closed
source, requires a MATLAB licence, not maintained, no GPU support, no B-tensor
encoding, and uses a different forward model for design than for fitting — the
gradient is approximated by finite differences over the MATLAB signal model.

**Camino**: Open source, Java-based. Has some OED functionality but no JAX/GPU,
no B-tensor, no Monte Carlo validation loop, and the signal model used for
design is not the same as used for fitting.

dmipy-design closes the loop:

1. **Same model for design and fitting.** The identical `forward_fn` from
   `dmipy.jax.multicompartment_jax` is used to compute the FIM and to fit data.
   Consistency is guaranteed by construction.

2. **Exact gradients.** JAX autodiff computes `dE/dtheta` analytically — no
   finite differences over the model parameters.

3. **MC validation.** Any designed protocol can be validated immediately with
   dmipy-sim Monte Carlo using the same physics engine. The MC-bridge brings the
   same validation into the OED loop itself.

4. **B-tensor and OGSE.** `encode_ste`, `encode_ogse`, and the STE/OGSE
   dispatch in `fim.py` extend OED to modern tensor-valued encoding without
   reimplementing the signal model.

5. **Open and reproducible.** Pure Python, Apache 2.0 licence, pip-installable,
   fully tested.

---

## See also

- `docs/faq.md` — answers to common questions about OED, FIM, and the design choices
- `docs/architecture.md` — FIM pipeline diagram and optimizer comparison table
- `docs/oed_theory.md` — CRLB derivation, D/A/E optimality, FIM averaging justification

## Documentation
- [`docs/faq.md`](docs/faq.md) — 20 frequently asked questions
- [`docs/architecture.md`](docs/architecture.md) — pipeline diagram and optimizer comparison
- [`docs/oed_theory.md`](docs/oed_theory.md) — FIM/CRLB derivation and optimality criteria
