"""Solver-agnostic encoding timing + spectral utilities (NumPy only, no JAX).

These pieces are shared by every waveform designer — the NOW SQP solver
(``now.py``), the differentiable JAX augmented-Lagrangian solver
(``waveform_designer.py``), and the stimulated-echo front-end — so they live here,
JAX-free, rather than inside any one solver.  ``waveform_designer`` re-exports them
for backward compatibility.

  * ``SequenceTiming`` — the physical spin-echo encoding-window budget (pins where
    the diffusion gradient may live and where the 180 sits).
  * ``encoding_spectrum`` — the rigorous Stepišnik encoding power spectrum |q̃(f)|².
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAMMA = 267.513e6  # rad/s/T — matches dmipy_sim.constants.GAMMA


def encoding_spectrum(G, dt, echo_idx):
    """Encoding power spectrum |q̃(f)|² of a physical gradient + its summary.

    The rigorous spectral-content quantity (Stepišnik): the diffusion signal is
    ``ln S ≈ −∫ D(ω)·|q̃(ω)|² dω``, so a waveform is characterized — for ANY shape,
    pure or broadband — by this spectrum, not by a nominal "frequency".  Returns
    ``(freqs_Hz, power, centroid_Hz, bandwidth_Hz, rms_Hz)`` (one-sided), letting
    you quantify and propagate the actual spectral content (and its imprecision).
    """
    G = np.asarray(G, dtype=np.float64)
    s = np.where(np.arange(G.shape[0]) < echo_idx, 1.0, -1.0)[:, None]
    q = GAMMA * np.cumsum(s * G, axis=0) * dt                      # (n_t,3) rad/m
    P = np.sum(np.abs(np.fft.rfft(q, axis=0)) ** 2, axis=1)        # (nf,) power
    f = np.fft.rfftfreq(G.shape[0], dt)
    Psum = P.sum() + 1e-30
    centroid = float((f * P).sum() / Psum)
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * P).sum() / Psum))
    rms = float(np.sqrt((GAMMA ** 2 * np.sum(G ** 2)) / (np.sum(q ** 2) + 1e-30))
                / (2 * np.pi))
    return f, P, centroid, bandwidth, rms


@dataclass
class SequenceTiming:
    """Physical diffusion spin-echo timing budget that pins the encoding windows.

    The 180 sits at TE/2 (spin-echo refocus condition).  Diffusion encoding is OFF
    during the excitation lead-in, across the 180 (+crushers), and during the
    readout tail; the two remaining windows (pre-/post-180) are generally UNEQUAL
    because ``t_prep+t_excite ≠ t_readout_pre_echo``.  So any pre/post asymmetry of
    the optimized waveform is a *consequence* of this budget, never a free knob::

        [prep+excite] [== pre-180 encode ==] [180] [== post-180 encode ==] [readout→echo]
        0             t_lead                 TE/2∓t_refocus/2          TE−t_ro_pre   TE

    All times in seconds.  Pass to ``design_waveform(..., timing=...)`` or
    ``design_waveform_now(..., timing=...)`` and the encoding-window masks + 180
    position are derived (overriding echo_frac / rf_duration).  Build it from a real
    sequence with ``from_pulseq``, or from a readout description with ``from_readout``.
    """
    t_excite: float                  # 90 RF duration; encoding starts after it
    t_refocus: float                 # 180 RF (+crusher) duration; off across it at TE/2
    t_readout_pre_echo: float        # readout start → echo; post-180 encode ends by TE−this
    t_prep: float = 0.0              # optional fat-sat/prep before encoding
    TE: float | None = None          # native echo time (e.g. read from a .seq); masks() default
    symmetric: bool = False          # VANILLA mode: mirror the pre/post-180 windows about the
    #                                  echo (equal durations), dead-timing the surplus of the
    #                                  longer window.  See masks() — this is the conventional
    #                                  "symmetric" waveform you reach by REFUSING the asymmetry.

    @property
    def t_lead(self) -> float:
        """Dead time from t=0 (excitation centre) until encoding may begin."""
        return self.t_prep + self.t_excite

    def min_TE(self) -> float:
        """Smallest TE for which both the pre- and post-180 encoding windows exist."""
        return max(2.0 * (self.t_lead + self.t_refocus / 2.0),
                   2.0 * (self.t_readout_pre_echo + self.t_refocus / 2.0))

    def masks(self, TE=None, n_t=256):
        """Return ``(slew_off_mask (n_t,1) float, echo_idx int)`` for a given TE.

        ``slew_off_mask`` is 1 in the two encoding windows and 0 in the off-regions
        (excitation lead-in, the 180, the readout tail), so the optimizer's gradient
        lives only where the hardware allows it.  The 180 (echo_idx) is at TE/2.

        ``symmetric`` (VANILLA mode): the inner edges are already ±t_refocus/2 from the
        echo, so a symmetric (mirror about the echo) encoding requires equal OUTER
        extents — both windows reach ``W = min(pre_dur, post_dur)`` out from the 180.
        The surplus of whichever real window was longer is forced to 0 → it becomes
        dead time the spins spend transverse (extra T2 loss).  This is the conventional
        symmetric waveform: the cost of REFUSING the budget's natural asymmetry.
        """
        TE = float(TE if TE is not None else self.TE)
        if TE < self.min_TE() - 1e-9:
            raise ValueError(
                f"TE={TE*1e3:.2f} ms is below min_TE={self.min_TE()*1e3:.2f} ms for "
                f"this timing (encoding windows would vanish).")
        dt = TE / (n_t - 1)
        t = np.arange(n_t) * dt
        echo = TE / 2.0
        on = np.ones(n_t, dtype=np.float64)
        on[t < self.t_lead] = 0.0                                   # excitation lead-in
        on[np.abs(t - echo) <= self.t_refocus / 2.0] = 0.0          # 180 (+crusher)
        on[t > TE - self.t_readout_pre_echo] = 0.0                  # readout tail
        if self.symmetric:
            pre_dur = (echo - self.t_refocus / 2.0) - self.t_lead
            post_dur = (TE - self.t_readout_pre_echo) - (echo + self.t_refocus / 2.0)
            W = max(0.0, min(pre_dur, post_dur))                    # mirror extent from 180
            on[t < echo - self.t_refocus / 2.0 - W] = 0.0           # dead-time the longer side
            on[t > echo + self.t_refocus / 2.0 + W] = 0.0
        return on[:, None], int(round(echo / dt))

    @classmethod
    def from_readout(cls, *, t_excite, t_refocus, readout_duration, partial_fourier,
                     t_prep=0.0, TE=None):
        """Build from a readout description.  The echo (k-space centre) sits
        ``(pf−0.5)/pf`` into the readout, so partial Fourier (pf<1) shortens the
        post-180 window — exactly the mechanism that makes the optimum asymmetric."""
        pf = float(partial_fourier)
        if not (0.5 <= pf <= 1.0):
            raise ValueError(f"partial_fourier must be in [0.5, 1.0]; got {pf}")
        return cls(float(t_excite), float(t_refocus),
                   float(readout_duration) * (pf - 0.5) / pf, float(t_prep), TE)

    @classmethod
    def from_pulseq(cls, src):
        """Read the timing budget (and native TE) from a Pulseq ``.seq`` via
        ``dmipy_sim.sequences.pulseq.pulseq_timing`` (first RF=90, second=180, one ADC)."""
        from dmipy_sim.sequences.pulseq import pulseq_timing
        d = pulseq_timing(src)
        return cls(t_excite=d['t_excite'], t_refocus=d['t_refocus'],
                   t_readout_pre_echo=d['t_readout_pre_echo'], TE=d['TE'])
