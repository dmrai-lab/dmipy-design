"""Deliverable, B1-robust refocusing-RF (180°) design via a Bloch forward.

Design the *B1 envelope* of a spin-echo refocusing pulse so that it refocuses well across a
realistic operating ensemble — transmit-field inhomogeneity (B1⁺ scale) × static
off-resonance (B0 / susceptibility spread) — while staying scanner-deliverable.  This is the
RF analogue of what ``design_waveform_now`` does for the diffusion *gradient*: maximise a
physical objective (here the refocused signal fraction) subject to the hardware/safety box.

The optimisation variable is a **band-limited** envelope (a few low-frequency cosine / DCT-II
coefficients).  Band-limiting bounds the RF slew-rate and bandwidth *structurally* — the RF
analogue of a gradient slew limit — while two soft penalties enforce the remaining box:

* **peak B1** ``max|B1| ≤ B1_max``           — the RF analogue of the gradient-amplitude box;
* **SAR proxy** ``∫ B1² dt ≤ budget``         — the RF analogue of the gradient heat limit,
  as a headroom multiple of a plain hard (flat) 180° over the same window.

The objective is a differentiable spin-echo **Bloch forward** (a chain of x/z rotations) over
the (B1⁺ × off-resonance) ensemble; the refocused fraction is the ensemble-mean transverse
magnitude at the echo.  A B1-robust 180° retains coherence that a hard 180° loses under
transmit inhomogeneity.

Needs only NumPy + SciPy (the problem is a handful of cosine coefficients, so SciPy's
``minimize`` is more than adequate).  Hardware limits (``B1_max``) are plain arguments; source
them from the dmipy-sim scanner catalogue (the ``[sim]`` extra) if you have it.

NOTE — this is an original NumPy/SciPy implementation for dmipy-design.  Optimising an RF
envelope through a Bloch/optimal-control forward is a standard technique (optimal-control RF
design, Conolly et al. 1986; GRAPE, Khaneja et al. 2005); the band-limited + peak-B1 + SAR
deliverability recipe here is our formulation, not lifted from any external library.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

GAMMA = 2.675e8   # rad/s/T, proton gyromagnetic ratio (matches HardwareConstraints)


@dataclass
class RfPulseDesign:
    """Result of :func:`design_refocusing_rf`.

    Attributes
    ----------
    B1 : np.ndarray
        Optimised RF B1 envelope over the pulse, in Tesla (length ``n_rf``).
    flip_angles : np.ndarray
        Per-sample nominal flip angles (rad) at B1⁺ = 1; ``sum|flip| = π``.
    dt : float
        RF raster (s).
    rf_duration : float
        Pulse duration (s) = ``n_rf * dt``.
    refocused_fraction : float
        Ensemble-mean refocused transverse fraction (0–1) of the designed pulse.
    refocused_fraction_hard : float
        Same metric for a plain hard (flat) 180° over the window — the baseline.
    peak_B1 : float
        Peak |B1| of the designed pulse (T).
    sar_proxy : float
        SAR proxy ``∫ B1² dt`` of the designed pulse (T²·s).
    sar_ratio : float
        ``sar_proxy`` as a multiple of the hard-180° reference.
    max_rf_slew : float
        Peak |dB1/dt| of the designed pulse (T/s).
    B1_max : float
        Peak-B1 limit used (T).
    sar_budget : float
        SAR-proxy budget used (T²·s) = ``sar_headroom × hard-180° SAR``.
    feasible : bool
        True if peak-B1 and SAR are within 2 % of their limits.
    n_basis : int
        Number of cosine-basis coefficients optimised.
    """
    B1: np.ndarray
    flip_angles: np.ndarray
    dt: float
    rf_duration: float
    refocused_fraction: float
    refocused_fraction_hard: float
    peak_B1: float
    sar_proxy: float
    sar_ratio: float
    max_rf_slew: float
    B1_max: float
    sar_budget: float
    feasible: bool
    n_basis: int

    def times(self) -> np.ndarray:
        """Sample times of the RF envelope (s), centred at 0."""
        n = self.B1.shape[0]
        return (np.arange(n) - (n - 1) / 2.0) * self.dt


# ── differentiable Bloch primitives (vectorised over the ensemble) ──────────────
def _rotx(M, ang):
    """Rotate magnetisation (3, E) about x by per-member angle ``ang`` (E,)."""
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([M[0], c * M[1] - s * M[2], s * M[1] + c * M[2]])


def _rotz(M, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * M[0] - s * M[1], s * M[0] + c * M[1], M[2]])


def _normalize(env: np.ndarray) -> np.ndarray:
    """Scale a raw envelope to per-sample flips summing (in |·|) to π."""
    return np.pi * env / (np.sum(np.abs(env)) + 1e-9)


def _refoc_fraction(flips_win, b1, dw, dt, n_t, win_mask):
    """Ensemble-mean refocused transverse fraction of a shaped 180°.

    90ₓ excitation, then free precession over ``n_t`` steps with the shaped flips applied
    inside the (centred) RF window; off-resonance precesses every step, so the 180° forms a
    spin echo.  Symmetric guard time each side makes the echo well-defined.
    """
    E = b1.size
    ff = np.zeros(n_t)
    ff[win_mask] = flips_win
    M = np.stack([np.zeros(E), np.zeros(E), np.ones(E)])
    M = _rotx(M, (np.pi / 2.0) * b1)                     # 90ₓ excitation (B1-scaled)
    zang = dw * dt
    for i in range(n_t):
        if win_mask[i]:
            M = _rotx(M, ff[i] * b1)                      # 180° sub-rotation (B1-scaled)
        M = _rotz(M, zang)                                # off-resonance precession
    return float(np.abs(np.mean(M[0] + 1j * M[1])))


def _rf_metrics(flips_win, dt):
    """(peak_B1 [T], SAR proxy ∫B1²dt [T²·s], peak RF slew [T/s]) of a flip envelope."""
    B1t = flips_win / (GAMMA * dt)
    peak = float(np.max(np.abs(B1t)))
    sar = float(np.sum(B1t ** 2) * dt)
    slew = float(np.max(np.abs(np.diff(B1t))) / dt) if B1t.size > 1 else 0.0
    return peak, sar, slew, B1t


def design_refocusing_rf(rf_duration=6e-3, *, dt=1e-5, guard_duration=None,
                         B1_max=20e-6, sar_headroom=1.30,
                         b1_range=(0.7, 1.3), n_b1=7,
                         off_resonance_hz=250.0, n_off_resonance=7,
                         n_basis=8, penalty_weight=50.0,
                         n_restarts=4, maxiter=300, seed=0) -> RfPulseDesign:
    """Design a B1-robust, deliverable 180° refocusing envelope.

    Maximise the ensemble refocused fraction of a shaped 180° over a (B1⁺ transmit scale ×
    static off-resonance) ensemble, subject to a band-limited envelope (bounded RF slew) plus
    peak-B1 and SAR penalties.  Returns an :class:`RfPulseDesign`.

    Parameters
    ----------
    rf_duration : float
        Pulse duration (s).  With ``dt`` it sets the number of RF samples ``n_rf``.
    dt : float
        RF raster (s).
    guard_duration : float, optional
        Free-precession time on *each* side of the pulse that defines the spin echo
        (defaults to ``rf_duration``, giving a symmetric echo interval).
    B1_max : float
        Peak-B1 limit (T).  E.g. GE SIGNA Premier body coil ≈ 19 µT.
    sar_headroom : float
        SAR budget as a multiple of the plain hard-180° power over the same window.
    b1_range, n_b1 : tuple, int
        Transmit-inhomogeneity ensemble: ``n_b1`` scales spanning ``b1_range``.
    off_resonance_hz, n_off_resonance : float, int
        Off-resonance ensemble: ``n_off_resonance`` shifts spanning ±``off_resonance_hz``.
    n_basis : int
        Number of low-frequency cosine (DCT-II) coefficients optimised.
    penalty_weight : float
        Weight on the peak-B1 and SAR soft penalties.
    n_restarts : int
        Random restarts (best kept); the first starts from a flat pulse.
    maxiter, seed : int
        SciPy ``minimize`` iteration cap and RNG seed.
    """
    n_rf = max(1, int(round(rf_duration / dt)))
    guard = rf_duration if guard_duration is None else guard_duration
    guard_steps = max(0, int(round(guard / dt)))
    n_t = n_rf + 2 * guard_steps
    win_mask = np.zeros(n_t, dtype=bool)
    win_mask[guard_steps:guard_steps + n_rf] = True

    # ensemble: transmit scale × static off-resonance (flattened)
    b1s = np.linspace(b1_range[0], b1_range[1], n_b1)
    dws = np.linspace(-off_resonance_hz, off_resonance_hz, n_off_resonance) * 2.0 * np.pi
    b1 = np.repeat(b1s, dws.size)
    dw = np.tile(dws, b1s.size)

    # band-limited envelope basis: DCT-II over the RF window
    ii = (np.arange(n_rf)[:, None] + 0.5) / n_rf
    Bmat = np.cos(np.pi * np.arange(n_basis)[None, :] * ii)   # (n_rf, n_basis)

    # hard (flat) 180° reference → SAR budget + baseline refocused fraction
    hard_flips = _normalize(np.ones(n_rf))
    _, sar_hard, _, _ = _rf_metrics(hard_flips, dt)
    sar_budget = sar_headroom * sar_hard
    refoc_hard = _refoc_fraction(hard_flips, b1, dw, dt, n_t, win_mask)

    def objective(c):
        flips = _normalize(Bmat @ c)
        refoc = _refoc_fraction(flips, b1, dw, dt, n_t, win_mask)
        peak, sar, _, _ = _rf_metrics(flips, dt)
        pen = (penalty_weight * max(0.0, peak / B1_max - 1.0) ** 2
               + penalty_weight * max(0.0, sar / sar_budget - 1.0) ** 2)
        return -refoc + pen

    c_flat = np.zeros(n_basis)
    c_flat[0] = 1.0                                           # flat start
    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restarts)):
        c0 = c_flat if r == 0 else c_flat + 0.3 * rng.standard_normal(n_basis)
        res = minimize(objective, c0, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    flips = _normalize(Bmat @ best.x)
    refoc = _refoc_fraction(flips, b1, dw, dt, n_t, win_mask)
    peak, sar, slew, B1t = _rf_metrics(flips, dt)
    feasible = (peak <= B1_max * 1.02) and (sar <= sar_budget * 1.02)

    return RfPulseDesign(
        B1=B1t, flip_angles=flips, dt=dt, rf_duration=n_rf * dt,
        refocused_fraction=refoc, refocused_fraction_hard=refoc_hard,
        peak_B1=peak, sar_proxy=sar, sar_ratio=sar / (sar_hard + 1e-30),
        max_rf_slew=slew, B1_max=B1_max, sar_budget=sar_budget,
        feasible=feasible, n_basis=n_basis,
    )
