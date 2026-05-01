"""
Master problem for column generation OED.

Given a fixed library of K shells with pre-computed FIMs {F_k},
finds the D-optimal allocation weights w = (w_1, ..., w_K) over the simplex.

Objective: maximise log det(sum_k w_k F_k)
           equivalently: minimise -log det(sum_k w_k F_k)

Gradient: d/dw_k (-log det F_total) = -trace(F_total^{-1} F_k)

Uses scipy SLSQP with analytic gradient for fast convergence.
float64 throughout.
"""

import numpy as np
import scipy.optimize

EPS_REG = 1e-10  # regularisation added to FIM before inversion


def solve_master(fim_atoms: list, w_init=None):
    """Solve the master problem (D-optimal weight allocation over simplex).

    Parameters
    ----------
    fim_atoms : list of np.ndarray, each shape (P, P)
        Per-shell FIM matrices (float64).  Must have at least one element.
    w_init : np.ndarray of shape (K,), optional
        Initial weights. Defaults to uniform (1/K).

    Returns
    -------
    w_opt : np.ndarray of shape (K,)
        Optimal weights (non-negative, sum to 1).
    obj_val : float
        Optimal objective value (-log det F_total).

    Notes
    -----
    The objective is concave in w (log-det of a positive-semidefinite matrix
    is concave in its arguments), so SLSQP converges to the global optimum.
    """
    K = len(fim_atoms)
    if K == 0:
        raise ValueError("fim_atoms must be non-empty")

    P = fim_atoms[0].shape[0]
    F_stack = np.stack([np.asarray(F, dtype=np.float64) for F in fim_atoms])  # (K, P, P)
    I_reg = EPS_REG * np.eye(P, dtype=np.float64)

    def objective_and_grad(w):
        F_total = np.einsum('k,kij->ij', w, F_stack) + I_reg
        sign, logdet = np.linalg.slogdet(F_total)
        if sign <= 0:
            # Near-singular or indefinite: return large value with zero gradient
            return 1e12, np.zeros(K)
        obj = -logdet
        F_inv = np.linalg.inv(F_total)
        # Analytic gradient: d/dw_k (-log det F_total) = -trace(F^{-1} F_k)
        grad = np.array([-np.trace(F_inv @ F_stack[k]) for k in range(K)],
                        dtype=np.float64)
        return obj, grad

    # Initialise weights
    if w_init is None:
        w0 = np.ones(K, dtype=np.float64) / K
    else:
        w0 = np.clip(np.asarray(w_init, dtype=np.float64), 1e-8, 1.0)
        w0 = w0 / w0.sum()

    constraints = [
        {'type': 'eq',
         'fun': lambda w: w.sum() - 1.0,
         'jac': lambda w: np.ones(K, dtype=np.float64)},
    ]
    bounds = [(0.0, 1.0)] * K

    result = scipy.optimize.minimize(
        objective_and_grad,
        w0,
        method='SLSQP',
        jac=True,
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-10, 'maxiter': 1000, 'disp': False},
    )

    w_opt = np.clip(result.x, 0.0, 1.0)
    w_opt /= w_opt.sum()
    return w_opt, float(result.fun)
