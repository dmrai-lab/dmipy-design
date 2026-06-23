# dmipy-design — Agent Rules

## Scope

dmipy-design owns the optimal experiment design layer. Its job is to choose
acquisition parameters (b-values, gradient directions, pulse timing) that
minimise the CRLB on parameter estimation error.

**Belongs here:**
- FIM computation (analytical via JAX autodiff, or finite-difference for MC-backed models)
- CRLB scalar objectives (D/A/E-optimality, parameter-selective)
- Optimisers that operate over acquisition scheme parameters
- JAX-traceable scheme encoders (PGSE, LTE, STE, OGSE, multi-shell)
- MC-bridge: wrapping dmipy-sim as a non-differentiable forward function for FD-FIM
- Hardware and time constraints on acquisition schemes
- Protocol serialisation (`.npz` save/load, conversion to dmipy-core scheme)

**Does NOT belong here (belongs in dmipy-core):**
- Signal models (`MultiCompartmentModel`, compartment models)
- Fitting / estimation (MAP, MLE, Bayesian posterior)
- Microstructure model validation (that lives in benchmarks/ in the orchestration repo)
- Monte Carlo simulation (that lives in dmipy-sim)

The boundary rule: dmipy-design calls `forward_fn(theta, scheme)` but never
defines what that function computes. It is handed a forward function from outside.

---

## Variable normalisation — always normalise to [0, 1] before L-BFGS-B

Protocol parameters have very different physical scales:
- b-value: O(1e8) to O(1e10) s/m²
- delta, Delta: O(1e-3) to O(1e-1) s

If you pass raw physical units to L-BFGS-B, the gradient with respect to the
b-value has magnitude O(1e-9) (because `d_obj / d_b ~ d_obj/d_E * dE/db` and
`dE/db ~ E * (-b_eff) * b_scale ~ 1e-9`). L-BFGS-B uses a gradient norm
termination criterion (`gtol`). With a gradient of O(1e-9), the solver concludes
it has converged after 1 iteration and returns the initial point unchanged.

The fix is to map each variable to [0, 1] before passing to the optimiser:

```python
scale = upper_phys - lower_phys          # [9.9e9, 0.059, 0.095]
v = (u - lower_phys) / scale             # v in [0, 1]^3
```

All gradients in normalised space are O(1), so the solver runs properly.

Always normalise. This is non-negotiable for any new optimiser added here.

---

## Why scipy L-BFGS-B, not jax.scipy.optimize

As of JAX 0.6.2, `jax.scipy.optimize.minimize` does not support box constraints
(bounds). PGSE optimisation has hard box constraints from hardware limits
(G_max, TE bounds). Without bounds, L-BFGS-B would explore physically impossible
regions (negative delta, b = 0, delta > Delta) and produce NaN FIM values.

`scipy.optimize.minimize` with `method='L-BFGS-B'` supports bounds natively.
We bridge to JAX by calling `jax.value_and_grad` inside a wrapper that returns
`(float, np.ndarray)` — scipy receives numpy arrays; JAX handles the autodiff.

Do not replace this with Adam or SGD. L-BFGS-B uses second-order curvature
information (approximate Hessian via BFGS updates) and converges in ~50–200
iterations for typical dMRI OED problems. Adam requires O(1e4) iterations on
the same problem and does not support hard bounds.

Do not replace scipy with `jax.scipy.optimize` until JAX adds bounds support.
Check the JAX changelog before making this change.

---

## FIM averaging over prior samples

A single-point FIM `F(theta_0)` is optimal only at `theta_0`. It would design
a protocol that is maximally informative only for tissue with exactly those
parameters. In vivo, tissue parameters vary continuously across the brain and
across subjects.

The solution is to average the FIM over samples from the prior:

```
F_avg = (1/M) * sum_m F(theta_m),   theta_m ~ p(theta)
```

This ensures the designed protocol is robust across the expected population.
The number of prior samples `M` trades off stability against cost. Default
`M = 512`; do not drop below `M = 64` without checking that the objective
landscape is not noisy.

The averaging is vmapped in `compute_fim_averaged` — it is a single
`jax.vmap` over samples, not a Python loop. Do not replace vmap with a loop.

---

## MC-bridge: fixed seed across FD perturbations

The MC forward function is stochastic: two calls with the same inputs but
different seeds return slightly different signals due to MC noise. If you use
a different seed for `E0` and `E_plus`, the finite difference

```
(E_plus - E0) / eps
```

contains an MC noise term of O(sigma_MC / eps) which swamps the true gradient
for small `eps`.

The fix: fix the seed in `build_mc_forward_fn` / `build_mc_forward_fn_fixed`
and never change it between calls. With the same seed, the MC noise is identical
in `E0` and `E_plus` and cancels to first order:

```
E_plus = E_true(theta + eps) + noise(seed)
E0     = E_true(theta)       + noise(seed)
(E_plus - E0) / eps ≈ dE_true/dtheta + O(eps)
```

Default seed is `42`. Do not randomise the seed between calls. If you need to
check MC noise sensitivity, use the `sigma_MC ≪ eps * |dE/dtheta|` rule:
increase `n_walkers` until this holds.

---

## B-tensor encoding: scheme dict format and OGSE t_eff convention

**STE (Spherical Tensor Encoding)**

`encode_ste(u, bvecs)` returns a dict with key `"encoding": "STE"`. The
b-tensor is `B = (b/3) * I` — isotropic. Because the b-tensor is spherically
symmetric, the signal is direction-independent: the bvecs in the dict are
stored for shape bookkeeping only and have no physical effect on the signal.

When computing the FIM for an STE scheme, `_scheme_to_jaxscheme` converts the
dict to a `JaxScheme` by copying `b_values → bvalues` and passing `delta`/`Delta`
directly. The forward function receives a standard `JaxScheme` and sees an
isotropic signal that depends only on the b-value trace.

**OGSE (Oscillating Gradient Spin Echo)**

`encode_ogse(u_ogse, bvecs)` takes `u_ogse = [b_value, frequency (Hz), n_cycles]`
and returns a dict with `"encoding": "OGSE"` and `"t_eff"`.

The effective diffusion time convention is:

```
t_eff = 1 / (4 * frequency)
```

This is the GPA result for sinusoidal gradients: the mean-squared displacement
is governed by an effective time equal to one quarter period. It is NOT the same
as `Delta - delta/3` for a PGSE scheme.

When `_scheme_to_jaxscheme` converts an OGSE dict to a `JaxScheme`, it sets
`delta = t_eff` and `Delta = t_eff + t_eff/3` so that `Delta - delta/3 = t_eff`
— the forward model receives the correct effective diffusion time through the
standard PGSE timing fields. This is a proxy convention; do not interpret the
`delta` and `Delta` fields of an OGSE-converted `JaxScheme` as physical pulse
timings.

**Dict key convention for non-PGSE schemes**

STE and OGSE dicts use `"b_values"` (with underscore) as the key, while
`JaxScheme` uses `.bvalues` (no underscore). The `_scheme_to_jaxscheme`
function handles both. Do not mix the two naming conventions in new code —
use `"b_values"` in dict encoders and `.bvalues` in `JaxScheme`.

---

## Tensor-valued waveform designer (`optimizers/waveform_designer.py`)

`encode_ste` only *asserts* `B=(b/3)I` with a guessed b — it is not a real,
hardware-realizable waveform.  `design_waveform(b_delta, ...)` subsumes it: it
optimizes an actual physical gradient `g(t)` with a **finite-180 spin echo built
in** (the sign flip enters `q(t)`, so refocusing `q(TE)=0` is intrinsic — unlike
loaded effective-only waveforms such as OPTICUBE, which have no 180), achieving
any target b-tensor shape (`b_delta`: 1 LTE, 0 STE, -0.5 PTE) while maximizing b
under Prisma hardware (G_max=0.08, slew=200, TE).  Use `design.to_sim_waveform()`
to hand the real gradient to the dmipy-sim MC forward / mc_bridge.

Solver: JAX augmented Lagrangian (slew + amplitude are structural via radial
tanh; refocus/shape/endpoints are equality constraints with multipliers), inner
jaxopt L-BFGS, vmapped multi-restart on GPU.  **Use the AL — do not revert to
fixed-penalty continuation**: the penalty weights cannot be balanced (too weak →
constraint violations, too strong → b collapses to a tiny on-shape blob).
Validated in three tiers (`tests/test_waveform_designer.py`): validity gates;
LTE recovers the analytic max-b (triangular q, ~19500 s/mm² at TE=80 ms); and the
**MC arbiter** — STE is orientation-invariant on an anisotropic cylinder (CV~0.5%
vs LTE ~29%) and every shape gives `exp(-bD)` on free diffusion.

NB: the small (3×3) b-tensor contraction must be an explicit outer-product sum,
not `einsum`/matmul — the matmul triggers an XLA "too small divisible part of the
contracting dimension" failure inside `jax.vmap` on GPU.
