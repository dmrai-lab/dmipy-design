# dmipy-design — Frequently Asked Questions

This document collects 20 questions that arise regularly when using or extending
dmipy-design. They are grouped by theme.

---

## Optimisation & Algorithms

### 1. Why CRLB/FIM over mutual information or posterior variance?

The Cramér-Rao Lower Bound is the tightest achievable lower bound on the
variance of any unbiased estimator. Because it is a function of the Jacobian
`dE/dtheta`, it is analytically differentiable through JAX autodiff — the
gradient `d(CRLB objective)/d(scheme parameters)` is exact, not approximated.
The FIM/CRLB framework matches the operating regime of Maximum Likelihood
Estimation at moderate-to-high SNR, which is the regime dMRI protocols are
designed for. Mutual information requires computing or approximating an entropy
integral over the posterior, which is expensive and sensitive to the choice of
density estimator. Bayesian posterior variance (expected posterior covariance)
requires full posterior integration — tractable with MCMC but not inside an
optimisation loop that evaluates the objective hundreds of times. The CRLB
gives the right answer in the MLE regime at a fraction of the computational
cost.

### 2. Why L-BFGS-B not Adam/SGD?

L-BFGS-B accumulates curvature information through rank-one BFGS updates of an
approximate inverse Hessian. For the smooth, low-dimensional OED problem
(typically 3–10 protocol parameters), this means convergence in approximately
50–200 iterations. L-BFGS-B also supports hard box constraints natively through
the active-set mechanism — this is essential because PGSE parameters must
satisfy hardware limits (G_max, TE bounds, delta < Delta) that cannot be
enforced by penalty terms without careful tuning. Adam and SGD are stochastic
gradient methods designed for high-dimensional non-convex problems (neural
network weights). They require O(1e4) iterations for the same OED problem and
have no built-in mechanism for box constraints. Introducing constraints via
projected gradient or penalty terms adds hyperparameters and convergence
instability. L-BFGS-B is the correct tool here.

### 3. Why is variable normalisation to [0, 1] essential?

Protocol variables span wildly different physical scales: b-values are O(1e8)
to O(1e10) s/m², while delta and Delta are O(1e-3) to O(1e-1) s. Because
`dE/db ~ E * b_eff * b_scale ~ 1e-9`, the gradient with respect to b in
physical units has magnitude O(1e-9). L-BFGS-B uses a gradient norm termination
criterion (`gtol`, default 1e-5). A gradient of O(1e-9) satisfies this criterion
immediately, so the solver concludes it has converged after a single iteration
and returns the initial point unchanged. Mapping each variable linearly to
[0, 1] by dividing by `upper - lower` brings all gradients to O(1), and the
solver runs correctly. Without normalisation, the optimiser always returns the
starting point — a silent failure that produces no error message. Normalisation
is non-negotiable for any optimiser added to dmipy-design.

### 4. Why average FIM over prior samples?

A single-point FIM `F(theta_0, scheme)` is maximally informative only for
tissue with exactly the parameters `theta_0`. Because tissue parameters vary
continuously across the brain and across subjects, a single-point design
produces a protocol that is optimal in a narrow parameter regime and potentially
poor elsewhere. Averaging the FIM over samples from a prior distribution,
`F_avg = (1/M) * sum_m F(theta_m, scheme)` with `theta_m ~ p(theta)`,
produces a protocol that is near-optimal across the expected population — this
is the standard Bayesian experimental design argument applied to the FIM. The
prior is drawn from the dmipy-core model's `parameter_ranges` via
`sample_prior()`. The vmap over samples in `compute_fim_averaged` means the
cost is a constant factor M, not M sequential evaluations. In practice, M=512
is sufficient for most smooth models.

### 5. D- vs A- vs E-optimality: which to use?

**D-optimality** minimises `-log det(F)`, which is equivalent to minimising the
volume of the confidence ellipsoid in parameter space. It is invariant to linear
reparameterisation of the parameter vector, making it appropriate when all
parameters are equally important and no single parameter dominates. Use
D-optimality as the default criterion.

**A-optimality** minimises `trace(F^-1)`, the sum of individual parameter
variances. It treats all parameters symmetrically but is more sensitive than
D-optimality to poorly constrained parameters (a single large diagonal entry
dominates the trace). Use A-optimality when average precision across parameters
is the explicit goal, or combined with `parameter_selective_crlb` to restrict
attention to a subset.

**E-optimality** minimises `max_eigenvalue(F^-1)`, the worst-case variance over
all linear combinations of parameters. It provides a minimax guarantee: the
designed protocol is best in the worst case. Use E-optimality when you need a
guaranteed precision floor regardless of which parameter direction is hardest to
estimate.

### 13. Why scipy not jax.scipy.optimize?

As of JAX 0.6.2, `jax.scipy.optimize.minimize` does not support box constraints
(bounds). PGSE protocol optimisation has hard box constraints from hardware
limits (G_max, TE bounds, delta < Delta). Without bounds enforcement, L-BFGS-B
would explore physically impossible regions — negative delta, zero b-value,
delta greater than Delta — and produce NaN FIM values. `scipy.optimize.minimize`
with `method='L-BFGS-B'` supports bounds natively through its active-set
mechanism. The bridge is a thin wrapper: `jax.value_and_grad` is called inside
a Python function that accepts and returns numpy arrays; scipy receives
`(float, ndarray)` and handles bounds transparently. Do not replace this with
`jax.scipy.optimize` until JAX adds native bounds support; check the JAX
changelog before considering the change.

### 16. What is greedy sequential OED and when is it preferable?

Greedy sequential OED selects, at each step, the single measurement that most
increases `log det(FIM)` given the measurements already chosen. The algorithm
is O(K) per step where K is the number of candidate measurements, compared to
O(N * K) for joint optimisation over all N measurements. Greedy OED is
preferable for online protocol adaptation (adding measurements to an existing
scan based on interim data), for very large N where joint optimisation is
computationally prohibitive, or for exploratory analysis of which measurement
types contribute the most information. The limitation is that greedy choices are
not globally optimal: a measurement that is second-best in isolation may be part
of a globally better pair. For offline design of a fixed protocol, `gradient_oed`
or `multishell_oed` will outperform the greedy approach.

### 11. Why does adding more measurements show diminishing returns?

The FIM is additive: `F(scheme1 + scheme2) = F(scheme1) + F(scheme2)`. Early
measurements fill the largest information gaps in parameter space — they probe
the sensitivity directions that contribute most to reducing `log det(F^-1)`.
Once those directions are well-constrained, additional measurements can only
improve directions that are already accurately estimated; the marginal gain in
`log det(F)` decreases with each added measurement. The optimal protocol
concentrates measurements at parameter regimes that maximise sensitivity to the
most poorly-constrained parameter combinations, not at regimes that simply
produce large signal. In practice, an optimised 30-measurement protocol can
outperform a 60-measurement heuristic protocol because the heuristic wastes
measurements in redundant or low-sensitivity regimes.

---

## Physics & Information Theory

### 14. What is parameter_selective_crlb?

`parameter_selective_crlb` computes an A-optimal CRLB criterion restricted to
a nominated subset of the full parameter vector. When some parameters are
nuisance parameters — present in the model but not of direct scientific interest
— it is appropriate to design the protocol to be maximally informative about
the parameters of interest rather than the full set. Formally, this is
equivalent to computing `trace(C_SS)` where `C_SS` is the sub-matrix of the
full CRLB matrix `F^-1` corresponding to the selected parameter indices. The
function accepts a list of parameter names or indices and returns the scalar
objective. It is compatible with JAX autodiff and can be used as a drop-in
replacement for the standard A-optimal objective in `gradient_oed` and
`multishell_oed`.

### 15. How many prior samples M are needed?

The default is M=512. Stability of the averaged FIM scales as O(1/sqrt(M)):
doubling M halves the Monte Carlo variance of the objective. For smooth,
low-dimensional models (two to three compartments, no sharp nonlinearities),
M=64 is often sufficient and noticeably faster. For models with sharp
nonlinearities in signal-vs-parameter curves — such as cylinder models near
zero radius — M=512 or above is advisable because the sensitivity landscape
varies rapidly and sparse sampling can miss high-sensitivity regions. The
practical check is to compare `F_avg(M)` and `F_avg(2M)`: if the CRLB
objective changes by less than 1%, M is sufficient. If it changes substantially,
double M and repeat.

### 18. Why float32 for prior samples but float64 for FIM?

Prior samples from `sample_prior()` are stored as float32 to match the
production convention for tissue parameter arrays in dmipy-core and to reduce
memory consumption when M is large (M=512 samples across a 10-parameter model
is still small, but consistency with the GPU production path matters). The FIM
computation itself, and particularly the CRLB matrix inversion, must use float64
because the Jacobian outer product involves small signal gradients; accumulating
these in float32 introduces cancellation errors that corrupt the inversion of
ill-conditioned FIM matrices. The rule: any computation involving matrix
inversion or eigenvalue decomposition of the FIM must use float64.

---

## Encoding & Schemes

### 8. What is STE and why is it direction-independent?

Spherical Tensor Encoding (STE) produces an isotropic b-tensor `B = (b/3) * I`.
Because the b-tensor is spherically symmetric, the signal attenuation depends
only on the b-value (the trace of B) and not on the gradient direction. For a
rotationally symmetric tissue model, `E(theta, STE, b, n) = E(theta, b)`
regardless of the unit vector `n`. As a consequence, for OED with STE the
gradient directions carry no information and only the b-value matters. The
`encode_ste` encoder stores bvecs in the scheme dict for bookkeeping but the
forward function effectively ignores direction. When optimising an STE protocol,
the effective optimisation dimensionality reduces to a scalar (b-value), making
the landscape significantly simpler than PGSE.

### 9. OGSE effective diffusion time convention

For oscillating gradient spin echo (OGSE) sequences with sinusoidal gradients
at frequency f, the effective diffusion time from the Gaussian Phase
Approximation is `t_eff = 1 / (4 * f)`. This is one quarter of the oscillation
period and represents the timescale over which mean-squared displacement
accumulates. The `encode_ogse` encoder derives `t_eff` from the frequency, then
sets proxy fields `delta = t_eff` and `Delta = t_eff + t_eff/3` so that
`Delta - delta/3 = t_eff` — the standard effective diffusion time formula for
PGSE. The forward model receives a `JaxScheme` with these proxy timings and
computes the signal correctly through the standard `Delta - delta/3` path. The
proxy `delta` and `Delta` values are not physical pulse timings; they should not
be interpreted as such or used to compute gradient amplitudes.

### 12. PGSE hardware constraint and OED

For a PGSE sequence, the gradient amplitude is `G = sqrt(b / (gamma^2 * delta^2 * (Delta - delta/3)))`.
The hardware limit `G_max` (typically 0.08 T/m on a clinical 3T scanner)
imposes a constraint on the feasible (b, delta, Delta) combinations. In
`HardwareConstraints`, this constraint is implemented as a box bound by finding,
for each candidate (delta, Delta) pair, the maximum feasible b-value. The
`gradient_oed` and `multishell_oed` optimisers enforce box constraints through
the L-BFGS-B bounds mechanism. Any (b, delta, Delta) point returned by the
optimiser is guaranteed to satisfy `G <= G_max` by construction. It is the
caller's responsibility to pass a valid `HardwareConstraints` object; the
optimiser will not check physical consistency of manually constructed protocol
vectors.

---

## MC-Bridge

### 6. How does the MC-bridge work and when should it be used?

The MC-bridge wraps `dmipy-sim` as a forward function compatible with the FIM
machinery. `build_mc_forward_fn` and `build_mc_forward_fn_fixed` return a
callable `mc_fwd(scheme)` that runs a Monte Carlo particle simulation and
returns the mean signal over walkers. Because the MC forward function is not
JAX-traceable, the Jacobian cannot be computed via autodiff; instead,
`compute_fim_fd` uses central or forward finite differences to approximate
`dE/dtheta`. Use the MC-bridge when the Gaussian Phase Approximation breaks
down: short diffusion times where delta or Delta approaches the structural
correlation length, large gradient amplitudes where higher-order GPA terms are
non-negligible, or geometries with curvature, branching, or finite membrane
permeability. For standard PGSE at clinical parameters with well-separated
length scales, the analytical forward function is accurate and faster by several
orders of magnitude.

### 7. Why is the MC seed fixed across finite-difference perturbations?

Monte Carlo signal estimates have noise with standard deviation sigma_MC
approximately 0.01 for n_walkers = 5000. The finite-difference step size is
eps ~ 1e-7. If different seeds are used for the unperturbed signal `E0` and
the perturbed signal `E_plus`, the finite difference `(E_plus - E0) / eps`
contains a noise term of magnitude `sigma_MC / eps ~ 1e5`, which completely
swamps the true gradient. Fixing the seed ensures that `E_plus` and `E0` are
evaluated with the same random walker trajectories. The MC noise is then
identical in both evaluations and cancels exactly in the difference to first
order: `(E_plus - E0) / eps = dE_true/dtheta + O(eps)`. The default seed
is 42. Never randomise the seed between forward calls in the FD loop.

### 20. Rule of thumb for n_walkers for stable MC FD gradients

With the seed fixed across FD perturbations, MC noise cancels to first order
and the finite-difference gradient is accurate even with modest n_walkers.
The binding constraint is instead that the signal estimate itself is physically
accurate — that is, that the statistical error in E0 is small compared to the
signal itself. A practical threshold is n_walkers >= 5000, which gives
sigma_MC / E0 < 0.1% for typical cylinder geometries at b = 1000 s/mm².
The diagnostic check is to evaluate the signal at fixed parameters with two
different seeds: if the signal varies by more than 0.1% between runs, increase
n_walkers. For permeable membranes or complex multi-compartment geometries,
sigma_MC may be larger and n_walkers = 10000 or more may be required.

---

## Integration & Architecture

### 10. Why use the same forward model for OED and fitting?

dmipy-design uses the identical `forward_fn` from `dmipy.jax.multicompartment_jax`
for both computing the FIM (design) and fitting data (inference). This
guarantees zero approximation gap between the model used to design the protocol
and the model used to analyse the data collected with that protocol. The
original MATLAB ActiveAx framework used a separate, simplified forward model
for the OED step and a different model for fitting — any mismatch between the
two introduced a systematic bias whose magnitude was hard to quantify. In
dmipy-design, the forward function is passed as an argument to `gradient_oed`
and `multishell_oed`; if the caller passes the same `forward_fn` to both OED
and fitting, consistency is guaranteed by construction.

### 17. What does Protocol.to_dmipy_scheme() do?

`Protocol.to_dmipy_scheme()` converts the optimised `Protocol` dataclass into
a `PGSEAcquisitionScheme` from dmipy-core. This closes the design-to-fitting
loop: once the optimiser returns a `Protocol`, `to_dmipy_scheme()` produces the
scheme object that can be passed directly to dmipy-core fitting functions
(`model.fit(data, scheme)`). The conversion extracts b-values, gradient
directions, delta, and Delta from the `Protocol`, applies any unit conversions
required by dmipy-core (which uses SI units throughout), and returns a fully
initialised `PGSEAcquisitionScheme`. Because the same `forward_fn` accepts both
`JaxScheme` and `PGSEAcquisitionScheme` inputs, the end-to-end pipeline —
design, simulate, fit — uses identical physics at every stage.

### 19. How does multishell_oed differ from independent per-shell gradient_oed?

`multishell_oed` jointly optimises over all shells simultaneously. The FIM is
computed over the concatenated multi-shell scheme, so the optimiser can exploit
complementarity between shells: information that one shell cannot provide about
a parameter can be provided by another, and the optimiser will trade off
b-values and timings across shells accordingly. Running independent `gradient_oed`
calls per shell ignores this complementarity; each shell is designed as if the
others do not exist, and the result is typically suboptimal in total information.
The practical difference matters most for models where different parameters are
sensitive to different b-value regimes (e.g., intra-axonal signal is high-b
sensitive, free water is low-b sensitive): joint optimisation naturally
concentrates each shell where it contributes the most.
