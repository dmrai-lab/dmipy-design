# dmipy-design — Architecture

---

## End-to-end OED pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUTS                                              │
│                                                                             │
│  prior_samples          forward_fn              initial scheme u0           │
│  (M x n_params,         forward_fn(theta,       [b, delta, Delta, ...]      │
│   float32)              scheme) -> signal        + HardwareConstraints      │
│  from sample_prior()    from dmipy.jax.mc        (G_max, TE bounds)         │
└────────────┬────────────────────┬────────────────────┬───────────────────── ┘
             │                    │                    │
             ▼                    ▼                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    FIM COMPUTATION LAYER                                   │
│                                                                            │
│  encode_pgse / encode_ste / encode_ogse / encode_multishell_pgse           │
│           │                                                                │
│           ▼                                                                │
│       JaxScheme   ──────────────────────────────────────────────────────  │
│   (bvalues, bvecs,         forward_fn(theta_m, scheme)                    │
│    delta, Delta)                    │                                      │
│                                     ▼                                      │
│                        Jacobian J_mk = dE_k/dtheta_m                      │
│                        (via jax.jacfwd or finite diff)                    │
│                                     │                                      │
│                                     ▼                                      │
│               F(theta_m, scheme) = (1/sigma^2) * J^T J                    │
│                                     │                                      │
│                                     ▼                                      │
│           compute_fim_averaged: F_avg = (1/M) * sum_m F(theta_m)          │
│                                 (jax.vmap over prior samples)              │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      OBJECTIVE (CRLB SCALAR)                               │
│                                                                            │
│   D-optimal:  obj = -log det(F_avg)                                       │
│   A-optimal:  obj = trace(F_avg^-1)                                       │
│   E-optimal:  obj = max_eigenvalue(F_avg^-1)                              │
│   Selective:  obj = trace(C_SS)   (sub-matrix for selected params)        │
│                                                                            │
│   All objectives are JAX-differentiable (or FD for MC-bridge)             │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         OPTIMISERS                                         │
│                                                                            │
│  gradient_oed           multishell_oed        greedy_sequential_oed        │
│  (single-shell)         (joint multi-shell)   (one measurement at a time) │
│                                                                            │
│  All use scipy L-BFGS-B with bounds.                                       │
│  Variables normalised to [0,1] before passing to L-BFGS-B.               │
│  jax.value_and_grad bridges JAX autodiff to scipy's numpy interface.      │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      Protocol (optimised result)                           │
│                                                                            │
│  Protocol dataclass: b_values, bvecs, delta, Delta, encoding              │
│  .save(path) / .load(path)  — .npz serialisation                          │
│  .to_dmipy_scheme()         — PGSEAcquisitionScheme for dmipy-core        │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    FITTING (dmipy-core)                                    │
│                                                                            │
│  scheme = protocol.to_dmipy_scheme()                                      │
│  model.fit(data, scheme)                                                   │
│                                                                            │
│  Same forward_fn used for design and fitting — zero approximation gap.    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Optimiser comparison

| Property | `gradient_oed` | `multishell_oed` | `greedy_sequential_oed` |
|---|---|---|---|
| **Use case** | Single-shell PGSE protocol design | Joint optimisation over N shells | Online / incremental measurement selection |
| **Dimensionality** | 3 variables (b, delta, Delta) + fixed bvecs | 3N variables for N shells | 1 measurement added per call |
| **Algorithm** | L-BFGS-B (scipy) + JAX autodiff | L-BFGS-B (scipy) + JAX autodiff | Exhaustive search over candidate pool |
| **Convergence** | ~50–200 L-BFGS-B iterations | ~100–500 L-BFGS-B iterations | O(K) evaluations per step (K = candidate pool size) |
| **Global optimality** | Local optimum (smooth landscape, good in practice) | Local optimum | Not globally optimal (greedy choices irrevocable) |
| **Shell complementarity** | Not applicable | Yes — shells can trade information | Partial — each step sees all prior measurements |
| **Hardware constraints** | Yes — box bounds on (b, delta, Delta) | Yes — per-shell box bounds | Yes — candidate pool pre-filtered by constraints |
| **When to prefer** | Default for single-shell design | Multi-shell design, offline | Online adaptation, exploratory analysis, large N |

---

## FIM pipeline detail

The FIM computation proceeds in four stages:

### 1. Scheme encoding

```
encode_pgse(u, bvecs)      → JaxScheme(bvalues, bvecs, delta, Delta)
encode_ste(u, bvecs)       → dict{"encoding":"STE", "b_values":..., "delta":..., "Delta":...}
encode_ogse(u_ogse, bvecs) → dict{"encoding":"OGSE", "t_eff":1/(4f), ...}
encode_multishell_pgse(U, bvecs_list) → list[JaxScheme]
```

STE and OGSE dicts are converted to `JaxScheme` by `_scheme_to_jaxscheme`
before entering the FIM computation. For OGSE, proxy timing fields are set so
that `Delta - delta/3 = t_eff = 1/(4f)`.

### 2. Forward evaluation and Jacobian

For analytical models (GPA regime):

```
E_k(theta) = forward_fn(theta, scheme)[k]
J_mk = dE_k / dtheta_m   (computed by jax.jacfwd)
```

For MC-bridge (non-GPA regimes):

```
J_mk ≈ (E_k(theta + eps*e_m) - E_k(theta)) / eps   (finite differences)
       with fixed random seed across all evaluations
```

### 3. FIM assembly

```
F(theta, scheme) = (1 / sigma^2) * J^T @ J
                 = (1 / sigma^2) * sum_k J_k ⊗ J_k
```

`J_k` is the k-th row of the Jacobian (gradient of signal at measurement k).
The outer product sum is computed in float64 to avoid cancellation errors.

### 4. Prior averaging

```
F_avg = (1/M) * sum_{m=1}^{M} F(theta_m, scheme),   theta_m ~ p(theta)
```

Implemented via `jax.vmap` over the prior sample batch — a single vectorised
forward pass, not a Python loop. The result is a (n_params, n_params) matrix
in float64.

### 5. Objective and gradient

```
obj = scalar_objective(F_avg)       # D/A/E/selective
grad = jax.grad(obj)(scheme_params) # exact autodiff (or FD for MC-bridge)
```

The gradient flows back through F_avg → J → forward_fn → scheme_params.
Variable normalisation ensures all gradient components are O(1) at the
L-BFGS-B interface.
