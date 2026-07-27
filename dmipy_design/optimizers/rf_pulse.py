"""Deliverable, B1-robust refocusing-RF (180°) design via a Bloch forward.

Design the *B1 envelope* of a spin-echo refocusing pulse so that it refocuses across a
realistic operating ensemble — transmit-field inhomogeneity (B1⁺ scale) × static
off-resonance (B0 / susceptibility) — while staying scanner-deliverable.  This is the RF
analogue of what ``design_waveform_now`` does for the diffusion *gradient*: maximise a physical
objective under the real hardware box, evaluated through the Bloch equation itself.

**Objective — per-spin refocusing fidelity.**  A refocusing pulse should act as a true 180°
rotation for every spin, regardless of the transmit strength or off-resonance it happens to
see.  The figure of merit is the crushed spin-echo **refocusing efficiency**

    η = (1 − M_z) / 2   ∈ [0, 1]      (M_z = the pulse acting on +z; η = |β|² in SLR terms)

which is 1 for a perfect 180° (M_z → −1) and 0 for no rotation.  The objective is the **mean of
η over the ensemble** — a per-spin scalar, so (unlike a coherent-sum metric) it cannot be gamed
by cross-spin phase cancellation: every spin must genuinely invert.

**Variable — a band-limited COMPLEX envelope.**  A real hard 180° gives the wrong flip to any
spin with B1⁺ ≠ 1; robust refocusing needs amplitude *and phase* structure (like a composite or
adiabatic pulse).  The optimisation variable is therefore a complex envelope built from a few
low-frequency cosine (DCT-II) coefficients per quadrature — band-limited, so the RF slew /
bandwidth stay bounded structurally.

**Deliverability box.**  ``peak |B1| ≤ B1_max`` is the hard hardware limit (always enforced as
a penalty).  RF *energy* — the SAR proxy ``∫|B1|²dt`` — is the price of robustness: a genuinely
robust 180° costs several× the energy of a minimal hard 180°.  ``sar_headroom`` optionally caps
it (as a multiple of the hard-180° energy); left ``None`` the design is peak-limited (maximally
robust) and simply *reports* the SAR it spent, so you can see the trade.

Needs only NumPy + SciPy.  Hardware limits (``B1_max``) are plain arguments; source them from
the dmipy-sim scanner catalogue (the ``[sim]`` extra) if you have it.

NOTE — original NumPy/SciPy implementation for dmipy-design.  Optimising an RF envelope through
a Bloch / optimal-control forward is an established technique (optimal-control RF design, Conolly
et al. 1986; GRAPE, Khaneja et al. 2005); the band-limited + peak-B1 + SAR recipe and the
crushed-echo refocusing objective here are our formulation, not lifted from an external library.
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
    B1 : np.ndarray (complex)
        Optimised RF B1 envelope over the pulse, in Tesla (length ``n_rf``). Complex: the
        magnitude is |B1(t)|, the argument is the transmit phase.
    dt : float
        RF raster (s).
    rf_duration : float
        Pulse duration (s) = ``n_rf * dt``.
    refocusing_efficiency : float
        Ensemble-mean crushed-echo refocusing efficiency η = ⟨(1−M_z)/2⟩ (0–1) of the design.
    refocusing_efficiency_hard : float
        Same metric for a plain hard (flat) 180° over the window — the baseline.
    nominal_flip_deg : float
        Integrated on-resonance nutation γ∫|B1|dt at B1⁺=1 (deg); ≈180 for a simple pulse,
        more for a multi-lobe robust pulse.
    peak_B1 : float
        Peak |B1| of the designed pulse (T).
    sar_proxy : float
        SAR proxy ``∫|B1|²dt`` of the designed pulse (T²·s).
    sar_ratio : float
        ``sar_proxy`` as a multiple of a plain hard-180° over the same window (the cost of
        robustness).
    max_rf_slew : float
        Peak |dB1/dt| of the designed pulse (T/s).
    B1_max : float
        Peak-B1 limit used (T).
    sar_budget : float or None
        SAR-proxy budget used (T²·s), or None if peak-limited.
    feasible : bool
        True if peak-B1 (and SAR, when budgeted) are within 2 % of their limits.
    n_basis : int
        Number of cosine-basis coefficients per quadrature.
    """
    B1: np.ndarray
    dt: float
    rf_duration: float
    refocusing_efficiency: float
    refocusing_efficiency_hard: float
    nominal_flip_deg: float
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

    def to_b1pulse(self, label="refocus"):
        """Build a dmipy-sim ``B1Pulse`` from the designed envelope (needs the ``[sim]`` extra).

        The optimised complex ``B1`` array (Tesla) is the ground-truth transmit waveform, so the
        designed pulse drops straight into dmipy-sim's Bloch forward / slice-profile — the RF
        mirror of ``NowDesign.to_sim_waveform``.
        """
        from dmipy_sim.rf import B1Pulse
        return B1Pulse(b1=self.B1.astype(np.complex128), dt=self.dt, label=label,
                       flip_deg=180.0)


# ── Bloch forward: net rotation of +z by a complex B1(t) over an ensemble ───────
def _inversion_mz(b1c, b1_scale, dw, dt):
    """M_z after applying the complex pulse ``b1c`` (Tesla) to +z, per ensemble member.

    Per-step rotation is the exact (Rodrigues) rotation about the effective field
    ``(γ B1x b1_scale, γ B1y b1_scale, Δω) dt``.  No free precession — the crushed spin-echo
    refocusing efficiency ``(1−M_z)/2`` depends only on the pulse's net rotation.
    """
    E = b1_scale.size
    Mx = np.zeros(E); My = np.zeros(E); Mz = np.ones(E)
    zang = dw * dt
    for k in range(b1c.shape[0]):
        nx = GAMMA * (b1c[k].real * b1_scale) * dt
        ny = GAMMA * (b1c[k].imag * b1_scale) * dt
        theta = np.sqrt(nx * nx + ny * ny + zang * zang)
        small = theta < 1e-30
        th = np.where(small, 1.0, theta)
        kx, ky, kz = nx / th, ny / th, zang / th
        c = np.cos(th); s = np.sin(th); omc = 1.0 - c
        kdotM = kx * Mx + ky * My + kz * Mz
        cx = ky * Mz - kz * My; cy = kz * Mx - kx * Mz; cz = kx * My - ky * Mx
        Mx2 = Mx * c + cx * s + kx * kdotM * omc
        My2 = My * c + cy * s + ky * kdotM * omc
        Mz2 = Mz * c + cz * s + kz * kdotM * omc
        Mx = np.where(small, Mx, Mx2)
        My = np.where(small, My, My2)
        Mz = np.where(small, Mz, Mz2)
    return Mz


def _efficiency(b1c, b1_scale, dw, dt):
    """Ensemble-mean crushed-echo refocusing efficiency ⟨(1−M_z)/2⟩."""
    return float(np.mean((1.0 - _inversion_mz(b1c, b1_scale, dw, dt)) / 2.0))


def _rf_metrics(b1c, dt):
    """(peak_B1 [T], SAR proxy ∫|B1|²dt [T²·s], peak RF slew [T/s])."""
    mag = np.abs(b1c)
    peak = float(np.max(mag))
    sar = float(np.sum(mag ** 2) * dt)
    slew = float(np.max(np.abs(np.diff(b1c))) / dt) if b1c.size > 1 else 0.0
    return peak, sar, slew


def design_refocusing_rf(rf_duration=6e-3, *, dt=1e-4,
                         B1_max=20e-6, sar_headroom=None,
                         b1_range=(0.7, 1.3), n_b1=7,
                         off_resonance_hz=250.0, n_off_resonance=7,
                         n_basis=10, peak_weight=80.0, sar_weight=50.0,
                         n_restarts=8, maxiter=400, seed=0) -> RfPulseDesign:
    """Design a B1-robust, deliverable 180° refocusing envelope.

    Maximise the ensemble-mean crushed-echo refocusing efficiency of a shaped, phase-modulated
    180° over a (B1⁺ transmit scale × static off-resonance) ensemble, subject to a peak-B1 limit
    (and optionally a SAR budget).  Returns an :class:`RfPulseDesign`.

    Parameters
    ----------
    rf_duration : float
        Pulse duration (s).  With ``dt`` it sets the number of RF samples ``n_rf``.
    dt : float
        RF raster (s).
    B1_max : float
        Peak-B1 limit (T).  E.g. GE SIGNA Premier body coil ≈ 19 µT.
    sar_headroom : float, optional
        SAR budget as a multiple of the plain hard-180° energy.  ``None`` (default) is
        peak-limited — the most robust pulse the coil can deliver, with the SAR it costs simply
        reported.  A robust 180° typically costs several× the hard-180° energy.
    b1_range, n_b1 : tuple, int
        Transmit-inhomogeneity ensemble: ``n_b1`` scales spanning ``b1_range``.
    off_resonance_hz, n_off_resonance : float, int
        Off-resonance ensemble: ``n_off_resonance`` shifts spanning ±``off_resonance_hz``.
    n_basis : int
        Cosine (DCT-II) coefficients per quadrature (2·``n_basis`` real parameters).
    peak_weight, sar_weight : float
        Penalty weights on the peak-B1 and (when budgeted) SAR constraints.
    n_restarts, maxiter, seed : int
        Random restarts (best kept; first starts from a hard 180°), SciPy iteration cap, RNG.
    """
    n_rf = max(3, int(round(rf_duration / dt)))

    # ensemble: transmit scale × static off-resonance (flattened)
    b1s = np.linspace(b1_range[0], b1_range[1], n_b1)
    dws = np.linspace(-off_resonance_hz, off_resonance_hz, n_off_resonance) * 2.0 * np.pi
    b1_scale = np.repeat(b1s, dws.size)
    dw = np.tile(dws, b1s.size)

    # band-limited basis: DCT-II over the RF window, one set per quadrature
    ii = (np.arange(n_rf)[:, None] + 0.5) / n_rf
    Bmat = np.cos(np.pi * np.arange(n_basis)[None, :] * ii)   # (n_rf, n_basis)

    def envelope(x):
        return Bmat @ (x[:n_basis] + 1j * x[n_basis:])

    # hard (flat) 180° reference: amplitude for a π on-resonance flip over the window
    A0 = np.pi / (GAMMA * n_rf * dt)
    hard = np.full(n_rf, A0, dtype=np.complex128)
    eff_hard = _efficiency(hard, b1_scale, dw, dt)
    _, sar_hard, _ = _rf_metrics(hard, dt)
    sar_budget = None if sar_headroom is None else sar_headroom * sar_hard

    def objective(x):
        b1c = envelope(x)
        eff = _efficiency(b1c, b1_scale, dw, dt)
        peak, sar, _ = _rf_metrics(b1c, dt)
        pen = peak_weight * max(0.0, peak / B1_max - 1.0) ** 2
        if sar_budget is not None:
            pen += sar_weight * max(0.0, sar / sar_budget - 1.0) ** 2
        return -eff + pen

    x_hard = np.zeros(2 * n_basis)
    x_hard[0] = A0                                            # DC term → flat hard 180°
    x0 = x_hard.copy(); x0[0] = 2.5 * A0                      # start hotter (helps find robust optima)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restarts)):
        xs = x_hard if r == 0 else x0 * (1.0 + 0.4 * rng.standard_normal(2 * n_basis))
        res = minimize(objective, xs, method="L-BFGS-B", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    b1c = envelope(best.x)
    eff = _efficiency(b1c, b1_scale, dw, dt)
    peak, sar, slew = _rf_metrics(b1c, dt)
    nominal_flip = float(GAMMA * np.sum(np.abs(b1c)) * dt)
    feasible = (peak <= B1_max * 1.02) and (sar_budget is None or sar <= sar_budget * 1.02)

    return RfPulseDesign(
        B1=b1c, dt=dt, rf_duration=n_rf * dt,
        refocusing_efficiency=eff, refocusing_efficiency_hard=eff_hard,
        nominal_flip_deg=np.degrees(nominal_flip),
        peak_B1=peak, sar_proxy=sar, sar_ratio=sar / (sar_hard + 1e-30),
        max_rf_slew=slew, B1_max=B1_max,
        sar_budget=(np.inf if sar_budget is None else sar_budget),
        feasible=feasible, n_basis=n_basis,
    )
