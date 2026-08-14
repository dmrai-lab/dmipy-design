"""Minimum-TE-at-a-target-b design — the SNR-optimal counterpart to max-b-at-fixed-TE.

``design_waveform_now`` maximises b at a FIXED TE.  In practice you often want the mirror
image: given a *required* b-value, find the **shortest TE** that still reaches it.  A shorter
TE means less T2 decay before the echo, so it is the **SNR-optimal** design for that b.

Because the achievable b is monotonically increasing in TE (more encoding time → more area
under ``q``), the minimum TE that reaches a target b is found by **bisecting TE** around the
max-b primitive — one ``design_waveform_now`` call per bracket/bisection step, no hand-rolled
scan.
"""
from __future__ import annotations

from .now import design_waveform_now, _validate_b_delta


def min_te_for_b(b_target, b_delta=1.0, *, timing=None, te_lo=None, te_hi=None,
                 te_max=0.25, tol_te=1e-3, n_seeds=1, verbose=False, **design_kwargs):
    """Smallest TE whose max-b NOW design reaches ``b_target`` (the SNR-optimal mode).

    Parameters
    ----------
    b_target : float
        Required b-value (s/m²).
    b_delta : float
        Target b-tensor shape (1 LTE, 0 STE, -0.5 PTE), passed to ``design_waveform_now``.
    timing : SequenceTiming or None
        The timing budget; its ``min_TE()`` is the hard lower floor for the bracket.
    te_lo, te_hi : float or None
        Optional TE bracket (s).  Defaults: ``te_lo`` = the timing floor (else 1 ms);
        ``te_hi`` is grown ×1.5 until ``b_target`` is reached.
    te_max : float
        Cap (s) for the upward search; raises if ``b_target`` is unreachable below it.
    tol_te : float
        Stop when the TE bracket is narrower than this (s).
    n_seeds : int
        Design each TE over this many restart-RNG seeds and keep the best feasible (max b),
        so ``b(TE)`` is robust to seed-to-seed optimizer variance on tight problems.
    **design_kwargs
        Forwarded to ``design_waveform_now`` (``G_max``, ``slew_rate_max``, ``n_t``,
        ``n_restarts``, ``null_M1``/``null_M2``/``maxwell``, ``spectral_freq``, ``pns`` …;
        any ``seed`` is overridden by the ``n_seeds`` sweep).

    Returns
    -------
    (design, te) : tuple
        The feasible ``NowDesign`` at the smallest TE reaching ``b_target``, and that TE (s).
    """
    # Validate the shape up front. The bracket search below treats ValueError as "TE below the
    # encoding-window floor" and keeps going, so an unrealisable b_delta raised from inside the solver
    # would be swallowed and re-reported as an unreachable b_target.
    _validate_b_delta(b_delta)
    dkw = {k: v for k, v in design_kwargs.items() if k != 'seed'}

    def reached(te):
        best = None
        for s in range(max(1, n_seeds)):
            try:
                d = design_waveform_now(b_delta, TE=te, timing=timing, seed=s, **dkw)
            except ValueError:
                continue                                  # TE below the encoding-window floor
            if best is None or (bool(d.feasible), d.b_value) > (bool(best.feasible),
                                                               best.b_value):
                best = d
        if best is None:                                  # no valid design at this TE
            if verbose:
                print(f"  min_te_for_b: TE={te*1e3:6.2f}ms  (below window floor) -> short")
            return False, None
        ok = bool(best.feasible) and (best.b_value >= b_target)
        if verbose:
            print(f"  min_te_for_b: TE={te*1e3:6.2f}ms  b={best.b_value/1e6:7.0f}  "
                  f"feasible={best.feasible}  -> {'reaches' if ok else 'short'}")
        return ok, best

    floor = timing.min_TE() if timing is not None else 1e-3
    lo = max(float(te_lo) if te_lo is not None else floor, floor)
    ok_lo, d_lo = reached(lo)
    if ok_lo:
        return d_lo, lo                                   # floor already reaches it
    hi = float(te_hi) if te_hi is not None else lo * 1.6
    ok_hi, d_hi = reached(hi)
    while not ok_hi:                                      # grow the upper bracket
        if hi > te_max:
            best_b = f"{d_hi.b_value/1e6:.0f}" if d_hi is not None else "n/a"
            raise ValueError(
                f"b_target={b_target/1e6:.0f} s/mm^2 not reached below te_max="
                f"{te_max*1e3:.0f} ms (best b={best_b}).")
        hi *= 1.5
        ok_hi, d_hi = reached(hi)
    best_d = d_hi                                         # invariant: lo short, hi reaches
    while hi - lo > tol_te:
        mid = 0.5 * (lo + hi)
        ok_mid, d_mid = reached(mid)
        if ok_mid:
            hi, best_d = mid, d_mid
        else:
            lo = mid
    return best_d, hi
