# dmipy-design — OED Theory

This document derives the key theoretical results underlying dmipy-design:
the Fisher Information Matrix, the Cramér-Rao Lower Bound, optimality criteria,
Bayesian robustness via prior averaging, variable normalisation, and the
MC-bridge finite-difference argument.

---

## 1. Fisher Information Matrix

### 1.1 Gaussian noise model

Let `E_k(theta)` denote the predicted normalised signal at measurement `k`,
where `theta` is a vector of tissue parameters. Observations are modelled as:

```
y_k = E_k(theta) + epsilon_k,    epsilon_k ~ N(0, sigma^2)
```

The log-likelihood for a set of K measurements is:

```
log p(y | theta) = -K/2 * log(2*pi*sigma^2)
                  - (1 / (2*sigma^2)) * sum_k (y_k - E_k(theta))^2
```

The Fisher Information Matrix is defined as:

```
F_ij(theta) = E_y[ (d log p / d theta_i) * (d log p / d theta_j) ]
```

Differentiating the log-likelihood:

```
d log p / d theta_i = (1/sigma^2) * sum_k (y_k - E_k(theta)) * dE_k/d theta_i
```

Taking the expectation (using E[y_k - E_k] = 0 and E[(y_k - E_k)^2] = sigma^2):

```
F_ij(theta) = (1 / sigma^2) * sum_k (dE_k/d theta_i) * (dE_k/d theta_j)
```

In matrix form, defining the Jacobian `J` with `J_ki = dE_k/d theta_i`:

```
F(theta) = (1 / sigma^2) * J^T J
```

`J` has shape (K, n_params). The FIM is symmetric positive semi-definite with
rank at most `min(K, n_params)`.

### 1.2 Rician noise model

MRI magnitude images follow a Rician distribution due to magnitude detection.
For SNR > 3 (typical in dMRI at clinical field strengths), the Rician
distribution is well approximated by a Gaussian with the same variance.
The Gaussian FIM is therefore appropriate for protocol design at clinical SNR.

For low-SNR regimes (SNR < 3), the exact Rician FIM element is:

```
F_ij^Rice(theta) = sum_k  [dE_k/d theta_i * dE_k/d theta_j]
                          * I_1(E_k * y_k / sigma^2)^2
                          / (sigma^2 * I_0(E_k * y_k / sigma^2)^2)
```

where `I_0` and `I_1` are modified Bessel functions of the first kind.
At high SNR, `I_1/I_0 -> 1` and the Rician FIM reduces to the Gaussian FIM.
dmipy-design uses the Gaussian FIM throughout; extending to the Rician case
is future work.

---

## 2. Cramér-Rao Lower Bound

**Theorem (Cramér-Rao).** Let `theta_hat(y)` be any unbiased estimator of
`theta`, i.e. `E[theta_hat] = theta`. Then the covariance matrix of
`theta_hat` satisfies:

```
Cov[theta_hat] >= F(theta)^-1
```

in the positive semi-definite sense: `Cov[theta_hat] - F^-1` is positive
semi-definite. In particular, for each parameter individually:

```
Var[theta_hat_i] >= [F^-1]_ii
```

The bound is tight: Maximum Likelihood Estimators achieve equality
asymptotically as the number of observations grows (or, in the single-sample
case, at high SNR). For dMRI protocols at clinical SNR (sigma ~ 0.02–0.05
times the b=0 signal), the MLE is the standard nonlinear least-squares fit
and the CRLB is a practical prediction of estimation variance.

The CRLB depends on the acquisition scheme through the Jacobian `J`. OED
minimises a scalar summary of `F^-1` over the space of feasible schemes —
this directly minimises the lower bound on estimation error without requiring
a specific estimator to be implemented.

---

## 3. Optimality criteria

Let `C = F^-1` denote the CRLB matrix (assuming F is full rank).

### D-optimality

```
objective_D = -log det(F) = log det(C)
```

Minimising `-log det(F)` is equivalent to minimising the volume of the
concentration ellipsoid `{delta_theta : delta_theta^T F delta_theta <= 1}`,
which is the geometric mean of the eigenvalues of `C`. D-optimality is
invariant to linear reparameterisation of `theta`: if `phi = A theta`, then
`det(F_phi) = det(A)^{-2} det(F_theta)` and the optimal scheme is the same.
D-optimality is the default criterion in dmipy-design.

### A-optimality

```
objective_A = trace(C) = trace(F^-1) = sum_i lambda_i^{-1}
```

where `lambda_i` are the eigenvalues of F. A-optimality minimises the
arithmetic mean of `C`'s eigenvalues, equivalently the sum of individual
parameter variances. It is more sensitive than D-optimality to small
eigenvalues of F (poorly constrained directions), because those contribute
large terms to the trace. A-optimality is not reparameterisation-invariant.

### E-optimality

```
objective_E = max_eigenvalue(C) = 1 / min_eigenvalue(F)
```

E-optimality minimises the worst-case variance over all unit-norm linear
combinations of parameters: `max_{||v||=1} Var[v^T theta_hat] = ||C||_2 = lambda_max(C)`.
It provides a minimax guarantee and is appropriate when no prior knowledge
exists about which parameter combination is most scientifically important.
E-optimality concentrates effort on the most poorly-constrained direction.

### Relationship between criteria

For a FIM with eigenvalues `lambda_1 >= ... >= lambda_p`:

```
D: minimise sum_i log lambda_i  (geometric mean of eigenvalues)
A: minimise sum_i 1/lambda_i    (harmonic mean)
E: maximise lambda_p            (smallest eigenvalue)
```

All three are monotone functions of the eigenvalue spectrum. D-optimality
typically gives the best-balanced protocols; E-optimality is most conservative.

---

## 4. Bayesian robustness via prior averaging

A single-point FIM `F(theta_0, scheme)` is optimal only at `theta_0`. In vivo,
tissue parameters vary across brain regions (white matter, grey matter, CSF),
across individuals, and with pathology. A protocol optimal at typical healthy
white matter parameters may be poorly informative for demyelinated tissue.

The Bayesian experimental design solution is to maximise the expected utility
over the prior:

```
U(scheme) = E_{theta ~ p(theta)}[utility(F(theta, scheme))]
```

For D-optimality, the utility is `log det(F)`. The expectation is:

```
E[log det(F)] ≈ (1/M) * sum_{m=1}^{M} log det(F(theta_m, scheme))
```

However, dmipy-design uses a simpler and computationally cheaper approximation:
average the FIM matrices first, then apply the criterion:

```
F_avg = (1/M) * sum_{m=1}^{M} F(theta_m, scheme)
objective = -log det(F_avg)
```

This is equivalent to Bayesian D-optimality under the approximation that the
expected log-determinant equals the log-determinant of the expected FIM. The
approximation is conservative (Jensen's inequality implies
`E[log det(F)] <= log det(E[F])`) and is standard in the OED literature.
The result is a protocol that is near-optimal across the expected population
distribution `p(theta)`.

The prior is sampled uniformly from the model's `parameter_ranges` via
`sample_prior(model, n_samples=M)`. The averaging is implemented as a
`jax.vmap` over the sample batch, not a Python loop.

---

## 5. Variable normalisation

### The problem

Protocol parameters in physical units span orders of magnitude:

| Variable | Typical range | Order of magnitude |
|---|---|---|
| b-value | 100e6 – 10000e6 s/m² | 1e8 – 1e10 |
| delta | 0.001 – 0.060 s | 1e-3 – 1e-1 |
| Delta | 0.010 – 0.100 s | 1e-2 – 1e-1 |

The gradient of the D-optimal objective with respect to b in physical units is:

```
d(-log det(F_avg)) / db  ~  d(-log det) / dF  *  dF/dJ  *  dJ/dE  *  dE/db
```

For a Gaussian model at b = 1000 s/mm²:

```
dE/db  =  -E(b) * D_eff  ~  0.6 * 1e-9 m²/s  ~  6e-10 m²/s  =  6e-10 (s/m²)^-1
```

in SI units (b in s/m²). The full chain gives `d(obj)/db ~ 1e-9`.

L-BFGS-B uses a gradient norm convergence criterion (`gtol = 1e-5` by default).
The gradient norm in physical units is dominated by the b-value component and
equals approximately 1e-9, which is well below `gtol`. The solver immediately
declares convergence and returns the starting point without taking a single
step. This is a silent failure: no error is raised, the objective is not
minimised, and the returned protocol is simply the initial guess.

### The fix: linear normalisation to [0, 1]

Map each variable `u_i` linearly to a normalised variable `v_i in [0, 1]`:

```
v_i = (u_i - lower_i) / (upper_i - lower_i)
```

The gradient in normalised space is:

```
d(obj)/dv_i = d(obj)/du_i * (upper_i - lower_i)
```

With `upper_b - lower_b ~ 1e10` and `d(obj)/du_b ~ 1e-9`:

```
d(obj)/dv_b ~ 1e-9 * 1e10 = 10
```

Similarly for delta and Delta. All gradient components in normalised space are
O(1), the gradient norm is O(1), and L-BFGS-B terminates only when the
objective has genuinely converged.

All optimisers in dmipy-design normalise before calling L-BFGS-B and
de-normalise the result before returning the optimised protocol. This is
non-negotiable.

---

## 6. MC-bridge: finite-difference Jacobian with fixed seed

### Setup

The MC forward function evaluates the signal by simulating N_w walkers in a
geometry:

```
E_k(theta) ≈ (1/N_w) * sum_{w=1}^{N_w} exp(-b_k * D_w)
```

where `D_w` is the apparent diffusion coefficient of walker w, which depends
on both the geometry parameters `theta` and the random seed `s`. Explicitly:

```
E_k(theta, s) = E_k^true(theta) + eta_k(s)
```

where `eta_k(s)` is zero-mean MC noise with standard deviation
`sigma_MC ~ 1 / sqrt(N_w)`.

### The fixed-seed argument

The finite-difference approximation to the Jacobian is:

```
J_ki = (E_k(theta + eps * e_i, s_+) - E_k(theta, s_0)) / eps
```

If `s_+ != s_0`:

```
J_ki = (E_k^true(theta + eps*e_i) - E_k^true(theta)) / eps
       + (eta_k(s_+) - eta_k(s_0)) / eps
```

The noise term has magnitude:

```
|eta_k(s_+) - eta_k(s_0)| / eps  ~  2 * sigma_MC / eps
```

For `sigma_MC = 0.01` (N_w = 5000) and `eps = 1e-7`:

```
noise contribution ~ 2 * 0.01 / 1e-7 = 2e5
```

This is approximately 1e5 times larger than a typical true gradient
(`dE/dtheta ~ 1` for normalised parameters), making the FD gradient useless.

With `s_+ = s_0 = s` (fixed seed):

```
J_ki = (E_k^true(theta + eps*e_i) + eta_k(s) - E_k^true(theta) - eta_k(s)) / eps
     = (E_k^true(theta + eps*e_i) - E_k^true(theta)) / eps
     = dE_k^true/dtheta_i + O(eps)
```

The noise cancels exactly to first order. The remaining error is O(eps), the
standard finite-difference truncation error. With `eps = 1e-7` and a smooth
signal model, the FD gradient is accurate to approximately 1e-7 relative error.

### Choosing eps and n_walkers

The finite-difference error has two contributions:

1. Truncation error: O(eps) from higher-order Taylor terms.
2. Residual noise: O(sigma_MC^2 / (N_w * eps)) from imperfect seed cancellation
   (second-order noise terms that do not cancel).

The optimal `eps` balances these:

```
eps_opt ~ sigma_MC / sqrt(N_w * |d^2E/dtheta^2|)
```

In practice, `eps = 1e-7` with `N_w = 5000` is stable for typical dMRI signal
models. The practical check: evaluate E at two different seeds; if the signal
varies by more than 0.1% (`sigma_MC / E0 > 1e-3`), increase `n_walkers`.
