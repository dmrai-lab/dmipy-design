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
│  encode_pgse_shell / encode_ogse_shell / encode_ste_shell                 │
│  encode_pgste_shell                                                        │
│           │                                                                │
│           ▼                                                                │
│       JaxScheme   ──────────────────────────────────────────────────────  │
│   (bvalues, bvecs,         forward_fn(theta_m, scheme)                    │
│    delta, Delta,                    │                                      │
│    TE, TM, gradient_strengths,      │                                      │
│    encoding_type)                   │                                      │
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

| Property | `gradient_oed` | `multishell_oed` | `greedy_sequential_oed` | `column_generation_oed` |
|---|---|---|---|---|
| **Use case** | Single-shell PGSE protocol design | Joint optimisation over N shells | Online / incremental measurement selection | Mixed waveform (PGSE+OGSE+PGSTE+STE) protocol discovery |
| **Dimensionality** | 3 variables (b, delta, Delta) + fixed bvecs | 3N variables for N shells | 1 measurement added per call | 2–3 variables per atom; number of atoms grows adaptively |
| **Algorithm** | L-BFGS-B (scipy) + JAX autodiff | L-BFGS-B (scipy) + JAX autodiff | Exhaustive search over candidate pool | Master (SLSQP simplex) + Pricing (jaxopt LBFGS vmapped) |
| **Convergence** | ~50–200 L-BFGS-B iterations | ~100–500 L-BFGS-B iterations | O(K) evaluations per step | Kiefer-Wolfowitz gap ≤ tol (default 5%) |
| **Global optimality** | Local optimum | Local optimum | Greedy | Exact D-optimality of the continuous design over the discovered atom library |
| **Shell complementarity** | Not applicable | Yes | Partial | Yes — master problem allocates weights jointly over all atoms |
| **Waveform types** | PGSE only | PGSE/multi-shell | Any (candidate pool) | PGSE, OGSE, STE, PGSTE (extensible) |
| **Hardware constraints** | Box bounds on (b, delta, Delta) | Per-shell box bounds | Candidate pool filter | `HardwarePreset` (G_max, delta_min, f_max, b_max) |
| **When to prefer** | Default for single-shell design | Multi-shell offline design | Online / incremental | Discovering which waveform mix is optimal for a given model+prior |

---

## FIM pipeline detail

The FIM computation proceeds in four stages:

### 1. Scheme encoding

Shell-level encoders (used by column generation and direct FIM evaluation):

```
encode_pgse_shell(b, delta, Delta, bvecs)
    → JaxScheme(bvalues, bvecs, delta, Delta, TE=2·Delta, gradient_strengths,
                encoding_type='pgse')

encode_ogse_shell(freq, G, bvecs)
    → JaxScheme(bvalues, bvecs, delta=t_eff, Delta=4t_eff/3,
                TE=2/freq, gradient_strengths=G, encoding_type='ogse')
    t_eff = 1/(4·freq);  b = γ²G²t_eff³

encode_ste_shell(b, delta, Delta, bvecs)
    → JaxScheme(..., TE=2·Delta, encoding_type='ste')
    Signal is direction-independent (B = b/3·I).

encode_pgste_shell(b, delta, Delta, bvecs)
    → JaxScheme(..., TE=2·delta, TM=Delta-delta, encoding_type='pgste')
    T2 acts only on the two gradient lobes (2·delta);
    T1 acts during the mixing time TM = Delta - delta.
```

All shell encoders populate `TE` and (for PGSTE) `TM` so that `make_snr_forward`
can apply the correct relaxation weighting without any per-type special-casing.

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

---

## Column generation OED

Column generation finds the D-optimal *continuous design* over a mixed
waveform library by alternating between two subproblems:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Atom library  {(F_k, w_k)}                                             │
│  Each atom = one acquisition shell (30 isotropic directions).           │
│  Weight w_k = fraction of total shells allocated to shell k.            │
│  Constraint: sum_k w_k = 1, w_k >= 0.                                  │
└──────────────┬──────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MASTER PROBLEM  (each iteration)                                       │
│                                                                         │
│  min_{w} -log det(sum_k w_k F_k)    s.t. sum_k w_k = 1, w_k >= 0      │
│                                                                         │
│  Solved via SLSQP (scipy).  Gradient: d/dw_k = -trace(F_total^-1 F_k) │
│  FIM linearity: F_total = sum_k w_k F_k  (convex in w).               │
└──────────────┬──────────────────────────────────────────────────────────┘
               │  w_opt, F_total, F_total_inv
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PRICING PROBLEM  (each waveform type)                                  │
│                                                                         │
│  max_{u} trace(F_total^-1 @ F_new(u))                                  │
│                                                                         │
│  F_new(u) = FIM of a new 30-direction shell at parameters u.            │
│  Solved via jaxopt.LBFGS + jax.vmap over n_restarts random starts.    │
│  All restarts compiled as one JIT kernel — GPU parallel.               │
│                                                                         │
│  Parameterisation (sigmoid reparameterisation: v = sigmoid(u)):        │
│    PGSE/PGSTE: v ∈ [0,1]^3 → (G, delta, Delta/delta ratio)            │
│                b derived: b = γ²G²δ²(Δ-δ/3)                           │
│    OGSE:       v ∈ [0,1]^2 → (f, G)                                    │
│                b derived: b = γ²G²t_eff³,  t_eff = 1/(4f)             │
└──────────────┬──────────────────────────────────────────────────────────┘
               │  best_rc, best_atom
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  KIEFER-WOLFOWITZ CONVERGENCE CHECK                                     │
│                                                                         │
│  KW gap = max_k trace(F_total^-1 F_k) - P                              │
│                                                                         │
│  At D-optimality: max_k trace(F^-1 F_k) = P  (number of parameters).  │
│  Proof: sum_k w_k trace(F^-1 F_k) = trace(F^-1 F_total) = P.          │
│  So the max can only equal P at optimality (all atoms are tied).       │
│                                                                         │
│  kw_gap_rel = kw_gap / P.  Converge when kw_gap_rel <= tol (5%).      │
└──────────────┬──────────────────────────────────────────────────────────┘
               │  kw_gap_rel > tol: add best_atom to library
               │  kw_gap_rel <= tol: DONE — prune low-weight atoms
               ▼
           CGResult(atoms, weights, history, final_obj)
```

### Hardware presets

`HardwarePreset` encodes scanner-specific constraints applied uniformly
to all waveform types:

| Preset | G_max | delta_min | f_max | b_max |
|--------|-------|-----------|-------|-------|
| `CLINICAL_3T` | 80 mT/m | 15 ms | 300 Hz | 10 000 s/mm² |
| `CONNECTOM_3T` | 300 mT/m | 5 ms | 500 Hz | 10 000 s/mm² |

`b_max` is a safety clip applied in `decode_pgse`/`decode_ste`; it is
rarely binding when `make_snr_forward` is used (T2 attenuation naturally
suppresses high-b shells).

---

## SNR weighting model

The base FIM assumes noise variance `sigma^2` is constant across all
measurements.  This is incorrect when shells differ in echo time TE:
longer TE → stronger T2 attenuation → lower absolute signal → higher
effective noise-to-signal ratio.

`make_snr_forward` wraps any forward function with:

```
S(theta, scheme) = S0 · exp(-TE/T2) · [exp(-TM/T1)] · E_diff(theta, scheme)
```

where `TM` is only present for PGSTE shells.  The FIM of the
T2/T1-weighted signal then becomes:

```
F_SNR(theta) = (1/sigma^2) · J_SNR^T J_SNR
             = (1/sigma^2) · diag(w)^2 · J_diff^T J_diff
```

where `w_k = S0 · exp(-TE_k/T2) · [exp(-TM_k/T1)]` is the per-measurement
SNR weight.  Shells with long TE contribute exponentially less to the FIM,
naturally suppressing unphysical high-b or long-Delta shells without an
explicit box constraint.

### PGSTE SNR advantage

For identical (b, delta, Delta) the SNR weights compare as:

```
PGSE:   w = S0 · exp(-2·Delta / T2)
PGSTE:  w = S0 · exp(-2·delta / T2) · exp(-(Delta-delta) / T1)
```

At Delta=150 ms, delta=20 ms, T2=80 ms, T1=1000 ms:

```
PGSE:   exp(-300/80) = 0.024
PGSTE:  exp(-40/80) · exp(-130/1000) = 0.607 · 0.878 = 0.533  (22× better)
```

The optimizer therefore discovers PGSTE atoms whenever long diffusion
times are required (large axons, slow exchange) while preferring OGSE
for short diffusion times (small cell diameters).
