"""Deliverable, B1-robust refocusing RF: adiabatic (hyperbolic-secant) pulse design.

A hard 180° is exactly π only where the transmit field is nominal; across a head $B_1^+$
varies by tens of percent, so off-nominal spins under/over-flip and the spin echo built on them
loses signal.  The standard, scanner-deliverable fix is an **adiabatic pulse**: sweep the RF
frequency slowly through resonance while the amplitude rises and falls, and — provided the
sweep is slow enough (the *adiabatic condition*) — the magnetisation **follows the effective
field** from +z to −z regardless of the exact $B_1^+$.  That is what makes it B1-robust, and its
trajectory is a smooth spiral down the Bloch sphere, not a delicate balance of flip angles.

This module designs the classic **hyperbolic-secant (HS) adiabatic full passage** (Silver,
Joseph & Hoult 1985), the complex envelope

    B1(t) = A0 · sech(β τ)^(1 + i μ),      τ = 2t/T − 1 ∈ [−1, 1]

— amplitude ``A0 sech(βτ)`` and a tanh frequency sweep of half-bandwidth ``μβ/(πT)``.  ``A0`` is
the peak amplitude (the deliverable knob, capped at ``B1_max``), ``β`` sets the truncation, and
``μ`` sets the sweep bandwidth / adiabaticity.  The design chooses ``μ`` (and, under a SAR cap,
``A0``) to maximise the ensemble **refocusing efficiency** through the Bloch equation.

**Objective — per-spin refocusing fidelity.**  The figure of merit is the crushed spin-echo
refocusing efficiency ``η = (1 − M_z)/2 ∈ [0,1]`` (M_z = the pulse acting on +z; η = |β|² in
Shinnar–Le-Roux terms), averaged over the ``(B1⁺ scale × off-resonance)`` ensemble.  It is 1
when every spin is genuinely inverted; being a per-spin scalar it cannot be gamed by cross-spin
phase cancellation.

**Deliverability.**  ``peak |B1| = A0 ≤ B1_max`` is the hard hardware limit.  Adiabatic pulses
are power-hungry (they spend well over the energy of a minimal hard 180° — that is the price of
following the field robustly); ``sar_headroom`` optionally caps the SAR proxy ``∫|B1|²dt`` as a
multiple of the hard-180° energy by lowering ``A0``.

Needs only NumPy + SciPy.  ``d.to_b1pulse()`` hands the design to dmipy-sim's Bloch forward.

NOTE — adiabatic *full passage* imparts a $B_0/B_1$-dependent phase, so as a **refocusing** pulse
it is used as a matched **pair** (double adiabatic refocusing, e.g. LASER) to cancel that phase;
the crushed-echo η here is the single-pass ``|β|²`` refocusing coefficient.  References:
Silver, Joseph & Hoult, *Phys. Rev. A* **31** (1985) 2753; Tannús & Garwood, *NMR Biomed.* **10**
(1997) 423 (review); Kupče & Freeman, *J. Magn. Reson. A* **115** (1995) 273 (offset-independent
adiabaticity).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

GAMMA = 2.675e8   # rad/s/T, proton gyromagnetic ratio (matches HardwareConstraints)


@dataclass
class RfPulseDesign:
    """Result of :func:`design_refocusing_rf` — a hyperbolic-secant adiabatic pulse.

    Attributes
    ----------
    B1 : np.ndarray (complex)
        RF B1 envelope over the pulse, in Tesla (length ``n_rf``); |B1| is the amplitude, the
        argument is the swept transmit phase.
    dt : float
        RF raster (s).
    rf_duration : float
        Pulse duration (s).
    mu : float
        HS sweep / adiabaticity parameter (dimensionless).
    beta : float
        HS truncation parameter (``sech(β)`` at the pulse edges).
    bandwidth_hz : float
        Frequency-sweep bandwidth ``2μβ/(πT)`` (Hz) — the inversion band.
    refocusing_efficiency : float
        Ensemble-mean crushed-echo refocusing efficiency η = ⟨(1−M_z)/2⟩ (0–1).
    refocusing_efficiency_hard : float
        Same metric for a plain hard 180° over the window — the baseline.
    peak_B1 : float
        Peak |B1| = A0 (T).
    sar_proxy : float
        SAR proxy ``∫|B1|²dt`` (T²·s).
    sar_ratio : float
        ``sar_proxy`` as a multiple of a plain hard-180° over the same window.
    max_rf_slew : float
        Peak |dB1/dt| (T/s).
    B1_max : float
        Peak-B1 limit used (T).
    sar_budget : float
        SAR-proxy budget used (T²·s), or ``inf`` if peak-limited.
    feasible : bool
        True if the ensemble refocusing efficiency exceeds 0.9 (a robust adiabatic passage) and
        peak-B1 / SAR are within their limits.
    """
    B1: np.ndarray
    dt: float
    rf_duration: float
    mu: float
    beta: float
    bandwidth_hz: float
    refocusing_efficiency: float
    refocusing_efficiency_hard: float
    peak_B1: float
    sar_proxy: float
    sar_ratio: float
    max_rf_slew: float
    B1_max: float
    sar_budget: float
    feasible: bool

    def times(self) -> np.ndarray:
        """Sample times of the RF envelope (s), centred at 0."""
        n = self.B1.shape[0]
        return (np.arange(n) - (n - 1) / 2.0) * self.dt

    def to_b1pulse(self, label="refocus"):
        """Build a dmipy-sim ``B1Pulse`` from the designed envelope (needs the ``[sim]`` extra).

        The complex ``B1`` array (Tesla) is the ground-truth transmit waveform, so the pulse
        drops straight into dmipy-sim's Bloch forward / slice-profile — the RF mirror of
        ``NowDesign.to_sim_waveform``.
        """
        from dmipy_sim.rf import B1Pulse
        return B1Pulse(b1=self.B1.astype(np.complex128), dt=self.dt, label=label,
                       flip_deg=180.0)


def _hs_envelope(A0, mu, beta, n):
    """Hyperbolic-secant adiabatic envelope B1(t) = A0 sech(βτ)^(1+iμ), τ∈[−1,1] (complex, T)."""
    tau = np.linspace(-1.0, 1.0, n)
    sech = 1.0 / np.cosh(beta * tau)
    return A0 * sech * np.exp(1j * mu * np.log(sech + 1e-300))


# ── Bloch forward: net rotation of +z by a complex B1(t) over an ensemble ───────
def _inversion_mz(b1c, b1_scale, dw, dt):
    """M_z after applying the complex pulse ``b1c`` (Tesla) to +z, per ensemble member.

    Exact per-step (Rodrigues) rotation about the effective field
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
                         B1_max=20e-6, sar_headroom=None, beta=5.3,
                         b1_range=(0.7, 1.3), n_b1=7,
                         off_resonance_hz=250.0, n_off_resonance=7,
                         mu_range=(0.7, 6.0), n_mu=18) -> RfPulseDesign:
    """Design a B1-robust hyperbolic-secant adiabatic refocusing pulse.

    Amplitude is set to the deliverable peak (``A0 = B1_max``, or lowered to meet ``sar_headroom``
    — adiabaticity improves with amplitude), and the sweep parameter ``μ`` is chosen to maximise
    the ensemble-mean refocusing efficiency ``η = ⟨(1−M_z)/2⟩`` over the ``(B1⁺ × off-resonance)``
    ensemble, evaluated through the Bloch equation.  Returns an :class:`RfPulseDesign`.

    Parameters
    ----------
    rf_duration : float
        Pulse duration (s).  Longer pulses are more adiabatic (more robust) but more $T_2$-costly.
    dt : float
        RF raster (s).
    B1_max : float
        Peak-B1 limit (T), e.g. GE SIGNA Premier body coil ≈ 19 µT.
    sar_headroom : float, optional
        SAR budget as a multiple of the hard-180° energy; lowers ``A0`` to fit.  ``None`` (default)
        uses the full peak — the most adiabatic (robust) pulse — and reports the SAR it costs.
    beta : float
        HS truncation parameter; ``sech(β)`` is the relative amplitude at the pulse edges
        (β≈5.3 ⇒ 1 % truncation, the common choice).
    b1_range, n_b1 : tuple, int
        Transmit-inhomogeneity ensemble the pulse must invert across.
    off_resonance_hz, n_off_resonance : float, int
        Off-resonance ensemble (the pulse's sweep bandwidth must cover it).
    mu_range, n_mu : tuple, int
        Search range and grid density for the sweep parameter ``μ``.
    """
    n_rf = max(3, int(round(rf_duration / dt)))
    T = n_rf * dt

    b1s = np.linspace(b1_range[0], b1_range[1], n_b1)
    dws = np.linspace(-off_resonance_hz, off_resonance_hz, n_off_resonance) * 2.0 * np.pi
    b1_scale = np.repeat(b1s, dws.size)
    dw = np.tile(dws, b1s.size)

    # hard (flat) 180° reference and its energy (for the SAR ratio / budget)
    A_hard = np.pi / (GAMMA * n_rf * dt)
    hard = np.full(n_rf, A_hard, dtype=np.complex128)
    eff_hard = _efficiency(hard, b1_scale, dw, dt)
    _, sar_hard, _ = _rf_metrics(hard, dt)

    # peak amplitude: full B1_max, lowered to meet a SAR budget if one is set
    A0 = float(B1_max)
    sar_budget = np.inf
    if sar_headroom is not None:
        sar_budget = sar_headroom * sar_hard
        sech2 = (1.0 / np.cosh(beta * np.linspace(-1, 1, n_rf))) ** 2
        A0_sar = np.sqrt(sar_budget / (np.sum(sech2) * dt))
        A0 = min(A0, A0_sar)

    # choose the sweep μ that maximises ensemble refocusing efficiency (grid + local refine)
    def neg_eff(mu):
        return -_efficiency(_hs_envelope(A0, mu, beta, n_rf), b1_scale, dw, dt)

    mus = np.linspace(mu_range[0], mu_range[1], n_mu)
    grid = [neg_eff(m) for m in mus]
    j = int(np.argmin(grid))
    lo = mus[max(0, j - 1)]; hi = mus[min(len(mus) - 1, j + 1)]
    res = minimize_scalar(neg_eff, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    mu = float(res.x) if -res.fun >= -grid[j] else float(mus[j])

    b1c = _hs_envelope(A0, mu, beta, n_rf)
    eff = _efficiency(b1c, b1_scale, dw, dt)
    peak, sar, slew = _rf_metrics(b1c, dt)
    bandwidth = 2.0 * mu * beta / (np.pi * T)
    feasible = (eff > 0.9) and (peak <= B1_max * 1.02) and (sar <= sar_budget * 1.02)

    return RfPulseDesign(
        B1=b1c, dt=dt, rf_duration=T, mu=mu, beta=float(beta), bandwidth_hz=float(bandwidth),
        refocusing_efficiency=eff, refocusing_efficiency_hard=eff_hard,
        peak_B1=peak, sar_proxy=sar, sar_ratio=sar / (sar_hard + 1e-30),
        max_rf_slew=slew, B1_max=float(B1_max), sar_budget=float(sar_budget),
        feasible=feasible,
    )
