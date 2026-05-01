"""
Column generation OED for dMRI acquisition scheme optimization.

The algorithm alternates between:
1. Master problem: D-optimal weight allocation over the current shell library.
2. Pricing problem: find the new shell (waveform type + parameters) with
   maximum reduced cost (most informative addition).

Convergence criterion: Kiefer-Wolfowitz (KW) gap < tol.

    KW gap = max_k trace(F_total^{-1} F_k) - P

The threshold P is exact regardless of n_dirs per shell, because
sum_k w_k trace(F^{-1} F_k) = trace(F^{-1} F_total) = P by construction.
At optimality KW gap ≤ 0. We converge when kw_gap_rel = kw_gap/P <= tol (default 5%).

Atoms
-----
Each atom = one complete acquisition shell with n_dirs=30 isotropic gradient
directions. Weight w_k = fraction of total shells allocated to shell k.
Constraint: sum_k w_k = 1, w_k >= 0.

Usage
-----
    from dmipy_design.optimizers.column_generation import column_generation_oed, Atom
    from dmipy_design.analytical_forward import ball_c4cylinder_forward
    from dmipy_design.jax_scheme_encoder import encode_pgse_shell, BVECS_30

    initial_atom = Atom(
        type='pgse',
        params={'b': 1e9, 'delta': 0.02, 'Delta': 0.04},
        scheme=encode_pgse_shell(1e9, 0.02, 0.04, BVECS_30),
    )
    result = column_generation_oed(
        forward_fn=ball_c4cylinder_forward,
        prior_samples=prior,
        sigma=0.05,
        waveform_types=['pgse', 'ogse'],
        initial_atoms=[initial_atom],
        max_iter=15,
    )
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import jax.numpy as jnp

from .master_problem import solve_master
from .pricing_problem import solve_pricing, N_DIRS
from ..fim import compute_fim_averaged

EPS_FIM = 1e-10  # regularisation for FIM inversion


@dataclass
class Atom:
    """A single acquisition shell (atom) in the column generation library.

    Attributes
    ----------
    type   : str           waveform type ('pgse', 'ogse', 'ste')
    params : dict          physical parameters (b, delta/Delta or f/G)
    scheme : JaxScheme     30-direction JaxScheme for this shell
    fim    : np.ndarray or None   shape (P, P), cached FIM; None until computed
    """
    type:   str
    params: dict
    scheme: object
    fim:    Optional[np.ndarray] = None


@dataclass
class CGResult:
    """Output of a column generation OED run.

    Attributes
    ----------
    atoms      : list of Atom    active shells after pruning
    weights    : np.ndarray      shape (K,), sums to 1
    history    : list of dict    one entry per iteration with keys
                                 iter, n_atoms, obj, best_rc, kw_gap,
                                 kw_gap_rel, new_atom_type, new_atom_params
    final_obj  : float           -log det F_total at convergence
    """
    atoms:     list
    weights:   np.ndarray
    history:   list
    final_obj: float


def column_generation_oed(
    forward_fn,
    prior_samples,
    sigma: float,
    waveform_types: list,
    initial_atoms: list,
    max_iter: int = 15,
    reduced_cost_tol: float = 0.05,
    n_pricing_restarts: int = 8,
    lbfgs_maxiter: int = 200,
    prune_threshold: float = 1e-3,
    verbose: bool = True,
) -> CGResult:
    """Column generation OED for dMRI acquisition scheme optimization.

    Parameters
    ----------
    forward_fn : callable (theta: jnp.ndarray, scheme: JaxScheme) -> jnp.ndarray
        Analytical forward function, JAX-differentiable.
    prior_samples : jnp.ndarray, shape (M, P)
        Parameter prior samples (physical units, float64).
    sigma : float
        Noise standard deviation.
    waveform_types : list of str
        Waveform types to consider in pricing, e.g. ['pgse', 'ogse'].
    initial_atoms : list of Atom
        Starting atom library.  At least one atom required.
    max_iter : int
        Maximum column generation iterations.
    reduced_cost_tol : float
        KW gap tolerance (relative to P) for convergence.  Default 0.05 (5%).
    n_pricing_restarts : int
        Random restarts per waveform type in the pricing problem.
    lbfgs_maxiter : int
        Maximum LBFGS iterations per restart in solve_pricing.  All restarts ×
        maxiter are unrolled by JAX; lower this on memory-constrained GPUs.
        Default 200; use 50–100 when GPU memory is limited.
    prune_threshold : float
        Remove atoms with weight below this threshold after convergence.
    verbose : bool
        Print per-iteration progress.

    Returns
    -------
    CGResult

    Notes
    -----
    KW gap convergence criterion (Kiefer & Wolfowitz 1960):
        kw_gap = max_k trace(F_total^{-1} F_k) / n_dirs - P
    At optimality kw_gap <= 0.  We normalise: kw_gap_rel = kw_gap / P
    and stop when kw_gap_rel <= reduced_cost_tol.
    """
    if not initial_atoms:
        raise ValueError("initial_atoms must contain at least one Atom.")

    P = prior_samples.shape[1]
    atoms = list(initial_atoms)

    # --- Pre-compute FIMs for initial atoms that don't have one yet ---
    for atom in atoms:
        if atom.fim is None:
            atom.fim = np.array(
                compute_fim_averaged(forward_fn, prior_samples, atom.scheme, sigma)
            )

    history = []

    for iteration in range(max_iter):
        fim_list = [a.fim for a in atoms]

        # -- Master problem --
        w_opt, obj_val = solve_master(fim_list)

        F_total = sum(w * F for w, F in zip(w_opt, fim_list))
        F_total_inv = np.linalg.inv(F_total + EPS_FIM * np.eye(P))

        # -- Pricing problem: find best atom across all waveform types --
        best_rc      = -np.inf
        best_atom    = None
        best_wtype   = None

        for wtype in waveform_types:
            rc, params, scheme = solve_pricing(
                forward_fn,
                prior_samples,
                sigma,
                F_total_inv,
                wtype,
                n_restarts=n_pricing_restarts,
                rng_seed=iteration * 100 + waveform_types.index(wtype),
                lbfgs_maxiter=lbfgs_maxiter,
            )
            if rc > best_rc:
                best_rc   = rc
                best_wtype = wtype
                best_atom = Atom(type=wtype, params=params, scheme=scheme)

        # KW gap (normalised by n_dirs=30; P is number of parameters)
        # rc = trace(F_inv @ F_new) / n_dirs
        # KW condition at optimum: max rc <= P  (when F_new has max n_dirs measurements)
        kw_gap     = best_rc - P
        kw_gap_rel = kw_gap / P

        iter_info = {
            'iter':             iteration,
            'n_atoms':          len(atoms),
            'obj':              obj_val,
            'best_rc':          best_rc,
            'kw_gap':           kw_gap,
            'kw_gap_rel':       kw_gap_rel,
            'new_atom_type':    best_atom.type  if best_atom else None,
            'new_atom_params':  best_atom.params if best_atom else None,
        }
        history.append(iter_info)

        if verbose:
            params_str = ', '.join(
                f"{k}={v:.4g}" for k, v in best_atom.params.items()
                if k != 'type'
            )
            print(
                f"Iter {iteration:2d} | obj={obj_val:.4f} | n_atoms={len(atoms)} "
                f"| KW_gap={kw_gap:.4f} ({kw_gap_rel * 100:.1f}%) "
                f"| new: {best_atom.type}  {params_str}"
            )

        if kw_gap_rel <= reduced_cost_tol:
            if verbose:
                print(
                    f"Converged: KW gap {kw_gap_rel * 100:.1f}% "
                    f"< {reduced_cost_tol * 100:.0f}%"
                )
            break

        # -- Add new atom to library --
        best_atom.fim = np.array(
            compute_fim_averaged(forward_fn, prior_samples, best_atom.scheme, sigma)
        )
        atoms.append(best_atom)

    # --- Final master solve ---
    fim_list = [a.fim for a in atoms]
    w_final, obj_final = solve_master(fim_list)

    # --- Prune negligible atoms ---
    active  = w_final > prune_threshold
    atoms   = [a for a, m in zip(atoms, active) if m]
    w_final = w_final[active]
    if w_final.sum() > 0:
        w_final = w_final / w_final.sum()

    return CGResult(
        atoms=atoms,
        weights=w_final,
        history=history,
        final_obj=float(obj_final),
    )
