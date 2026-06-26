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

## Waveform design — NOW (design oracle) + AL (co-opt engine)

There are **two** gradient-waveform solvers, doing different jobs. Don't treat them as
competing designers:

- **NOW — `optimizers/now.py::design_waveform_now` — the DESIGN ORACLE.** A NumPy+SciPy
  port of NOW's own recipe (Sjölund 2015; jsjol/NOW): maximize `b = gᵀQg` with
  active-set SQP (scipy SLSQP), constraints expressed the way NOW expresses them —
  LINEAR matrices (per-axis slew, refocus `q(TE)=0`, moments M1/M2), amplitude as box
  bounds, nonlinear b-tensor SHAPE + Maxwell + OGSE-spectral, with **analytic** objective
  + constraint Jacobians (NO autodiff). All shapes in one solver (LTE/PTE/STE; OGSE via
  `spectral_freq`), and the **full deliverability set**: slew / amplitude / M1 / M2 /
  shape / Maxwell / spectral / **PNS (SAFE model)** / **heat (∫g²)**. It gets the *best b*
  with machine-precision constraints because SQP handles them exactly (the objective is
  never dwarfed). Returns a `NowDesign` with `.effective_G()` / `.to_sim_waveform()`.
  **This is the default designer — use it.** Why NumPy+SciPy and not JAX: active-set SQP
  is sequential/combinatorial (bad for vmap/autodiff), and feeding scipy a JAX-autodiff
  objective crosses a per-iteration numpy↔JAX bridge that dominates wall-clock (minutes).
  Pure-numpy-analytic derivatives → scipy calls NumPy directly (LTE ~1 s). The lesson:
  for scipy problems go fully numpy-analytic OR fully end-to-end-JAX; never scipy-over-autodiff.

- **AL — `optimizers/waveform_designer.py::design_waveform` — the CO-OPT ENGINE.** The
  JAX augmented-Lagrangian solver. NOW supersedes it as a *b-maximizer* (exact constraints,
  best b), so it is **kept for what only it can do**: it is end-to-end differentiable, the
  substrate for simulator-in-the-loop **co-optimization** (RF + gradient through a
  differentiable Bloch). It is NOT a competing designer; reach for it when you need to
  backprop through the forward model, not to design a max-b waveform.

Shared, solver-agnostic pieces (`SequenceTiming`, `encoding_spectrum`) live in the
JAX-free `optimizers/timing.py`, so NOW doesn't import the JAX module. Hardware/safety
limits come from the **cited `dmipy_sim.sequences.scanner_constants` catalogue** (the
acquisition-side analogue of `biophysical_constants`) — `gradient_limits(model, regime=)`,
`get_limit(..., si=True)`, `sar_limit(...)`. Don't hard-code G_max/slew/peak-B1/SAR; read
them from there (and note `regime='diffusion'` gives the PNS-derated slew, e.g. Connectom
200→62.5 T/m/s).

### The AL designer (now the co-opt engine) — physics still applies

`design_waveform(b_delta, ...)` optimizes a hardware-realizable physical gradient
`g(t)` with a finite-180 spin echo built in, achieving any target b-tensor shape
(`b_delta`: 1 LTE, 0 STE, -0.5 PTE) while maximizing b under Prisma hardware
(G_max=0.08, slew=200, TE).  The 180 is intrinsic — the sign flip enters `q(t)`,
so refocusing `q(TE)=0` is part of the optimization (an effective-only waveform
with no 180 cannot refocus a static field).  `design.to_sim_waveform()` hands the
gradient to the dmipy-sim forward / mc_bridge.  (All the robustness/timing/spectral
rules below apply equally to NOW, which enforces the same constraints exactly.)

**Robustness constraints — ON by default**, opt out per flag: `null_M1`
(velocity, ∫t·g_eff=0), `null_M2` (acceleration, ∫t²·g_eff=0), `maxwell`
(concomitant field, ∫s·g·gᵀ=0; Szczepankiewicz 2019).  They default on because a
*needed-but-off* constraint BIASES the measurement, while an unneeded one only
costs b (SNR).  Turn one off only when its confound is absent — `null_M1`/`null_M2`
for static samples (ex-vivo/phantom) or low b; `maxwell` for time-symmetric
waveforms (where it is ~0 anyway) or near-isocenter/high-B0. The full rationale
(bias, cost, when-safe) lives in the module docstring's "Robustness constraints"
section — read it before disabling a flag.  All three indices (`m1_index`,
`m2_index`, `maxwell_index`) are always reported, so the residual is visible even
when a flag is off.  Cost is real: e.g. velocity comp is ~4× b (the flow-comp
penalty), Maxwell on a symmetric waveform is free.  Fully-constrained designs are
the hardest to converge — raise `n_restarts`/`n_outer` if `feasible=False`.

Solver: JAX augmented Lagrangian (slew + amplitude structural via radial tanh;
refocus/shape/endpoints/M1/M2/Maxwell are equality constraints with multipliers),
inner jaxopt L-BFGS, vmapped multi-restart on GPU.  **Use the AL — do not revert
to fixed-penalty continuation**: penalty weights cannot be balanced (too weak →
violations, too strong → b collapses to a tiny on-shape blob).  Validated in
`tests/test_waveform_designer.py`: validity gates; LTE recovers the analytic
max-b (triangular q, ~19500 s/mm² at TE=80 ms); MC arbiter via dmipy-sim — STE is
orientation-invariant on an anisotropic cylinder (CV~0.5% vs LTE ~29%) and every
shape gives `exp(-bD)` on free diffusion; and the constraint indices drop to ~0
when their flag is on.

**Real timing → `SequenceTiming` (no asymmetry knob).** Don't ask the user for
"0.3 asymmetry" — asymmetry is a *consequence* of the sequence timing, not a
choice.  `SequenceTiming(t_excite, t_refocus, t_readout_pre_echo, t_prep)` holds
the physical budget; the 180 is pinned at TE/2 (spin-echo) and the encoding is
OFF during the excitation lead-in, the 180, and the readout tail.  Because
lead-in ≠ readout-pre-echo, the pre/post-180 windows come out unequal — the
optimum is asymmetric as an *output*.  Build it from a readout
(`SequenceTiming.from_readout(..., partial_fourier=…)` — PF<1 shortens the
post-180 window, the real TE-shortening mechanism) or from a scanner sequence
(`SequenceTiming.from_pulseq(seq)`, reading the 90/180/ADC schedule via
`dmipy_sim.sequences.pulseq.pulseq_timing`).  Pass `design_waveform(..., timing=…)`
and the masks + 180 position are derived (echo_frac/rf_duration ignored).
Round-trip in practice: scanner Opts (`PULSEQ_SYSTEMS`) + `.seq` skeleton →
`SequenceTiming.from_pulseq` → design within hardware → MC-validate
(`to_sim_waveform`) → `to_pulseq`.  `from_btensor_waveform` carries the design's
own `echo_idx` and the Bloch builder un-folds/places the 180 there (not hardcoded
TE/2); an **off-centre 180 is guarded** (raises unless `allow_offcenter_180=True`)
— it's a misaligned spin echo (static field refocuses at 2·t_180, not at TE).  The
real asymmetry (encoding *windows*) keeps the 180 at TE/2 and round-trips freely.

**Spectral paradigm (OGSE).** OGSE and PGSE share the *same* b-tensor (both linear);
what distinguishes OGSE is temporal/spectral content, which the b-tensor can't see.
So `design_waveform` **always reports the encoding power spectrum** (`spectral_rms_hz`
/ `spectral_centroid_hz` / `spectral_bandwidth_hz`) for any waveform, and
`spectral_freq=f` optionally drives the RMS encoding frequency to `f` (an OGSE-like
oscillating waveform — for any `b_delta`).  Per `ln S ≈ −∫ D(ω)|q̃(ω)|² dω` you
propagate the *realized* spectrum (bandwidth = how monochromatic it actually is)
rather than assuming a pure single frequency, so frequency precision is a reported /
optionally-constrained quantity, not a requirement.  `f_rms` is the FFT-free,
differentiable constraint (`sqrt(γ²Σ|g|²/Σ|q|²)/2π`); centroid/bandwidth are a
post-hoc FFT (`encoding_spectrum`).  Verified: no-spectral f_rms≈7Hz (PGSE-like,
b≈19700), spectral_freq=80 → f_rms≈80 (b≈120 — the OGSE efficiency cost).

**PGSTE groundwork** (`optimizers/stimulated_echo.py`).  PGSTE reuses the optimizer
core *unchanged* (same q / b-tensor); only the structure differs (3×90 stimulated
echo, no 180).  `StimulatedEchoTiming` is the timing analog: encoding in the two
matched transverse periods τ₁/τ₃ around a long gradient-off mixing time TM (the
effective sign flip / `echo_idx` sits in the dead middle), with **τ₁ = τ₃ enforced**
for static refocus (the PGSTE analog of 180@TE/2).  `design_stimulated_echo` wraps
`design_waveform` via the duck-typed `timing`.  TM gives a long *T1-limited*
diffusion time; static field is immune during TM.  Done as groundwork; NOT yet:
GPU validation of the encodings + a dmipy-sim 3×90 *playback* builder (the spin-echo
`from_btensor_waveform` correctly refuses a PGSTE design via the off-centre guard).

## RF + gradient co-optimization (`benchmarks/now_coopt_pipeline.py`)

The "best of both worlds" pipeline: **NOW designs the constraint-optimal gradient
(Stage 1)**, then a **differentiable-Bloch JAX co-opt refines the 180 RF over a
(B1⁺ × off-resonance) ensemble on that exact gradient (Stage 2)** — the spin-physics
robustness NOW's closed-form, ideal-hard-pulse `b`/refocus cannot see. The design solve
(max b) wants active-set SQP and does NOT need to be differentiable; what must be
differentiable is the *Bloch forward model*, not the design solver. So the architecture
is a pipeline (NOW warm-starts the co-opt), not one merged solver.

**RF deliverability** is the RF analogue of the gradient constraints, and it must be
enforced or the optimizer produces an undeliverable pulse (huge RF slew, 0↔π phase jumps,
unbounded SAR). The co-opt enforces: band-limited envelope (cosine basis → bounds RF
slew/bandwidth structurally), peak-B1 penalty (↔ amplitude box), SAR `∫B1²` penalty
(↔ heat). Limits come from the scanner_constants catalogue. NOTE: there is **no
authoritative RF-deliverability checker to delegate to** — pulseq carries RF *timing/raster*
(`rf_raster_time` 1 µs, `rf_dead_time`, `rf_ringdown_time`) and gradient PNS, but **no
peak-B1 cap and no SAR** (its `SAR_calc` is a deprecated placeholder). Peak-B1 is a vendor
number (coil/patient-dependent), SAR/B1⁺rms are IEC 60601-2-33 (patient-mass-dependent),
enforced by the scanner's runtime supervision. So RF deliverability is encoded here, not
looked up.

CAVEAT (current state, do not overclaim): the co-opt's forward is still a **toy
single-vector Bloch** with analytic `exp(-bD)` diffusion over a coarse (B1, off-res) grid,
and only the **180** is optimized. The real version uses `dmipy_sim.trajectories.
apply_waveform_bloch[_jax]` over cached white-matter walker trajectories with the 90 AND
180 as `rf_events` — the differentiable MC over real spins. The spacing IS already real:
`Sequence.from_btensor_waveform(design.effective_G(), dt, echo_idx=…)` builds the spin echo
and enforces 180-at-TE/2 (off-centre guarded).

NB: the small (3×3) b-tensor contraction must be an explicit outer-product sum,
not `einsum`/matmul — the matmul triggers an XLA "too small divisible part of the
contracting dimension" failure inside `jax.vmap` on GPU.
