"""The timing budget a design is built to, and the encoding masks the NOW core optimises inside.

The budget itself -- where the gradient may not be on: the excitation lead-in, the refocusing window at
TE/2, the readout tail -- is dmipy-sim's :class:`~dmipy_sim.acquisition.timing.SequenceTiming`, re-exported
here: the object a designed sequence carries is the object the designer read. What this module adds is the
designer's reading of a budget on a grid:

* :func:`encoding_mask` -- the spin-echo encoding windows (and the 180's sample) for a TE, with the
  ``symmetric`` option that mirrors the windows about the echo (the vanilla waveform, at the cost of dead
  time);
* :func:`stimulated_echo_mask` -- the two matched transverse windows of a PGSTE around its mixing time, and
  the recall sample where the effective gradient's sign flips;
* :func:`encoding_spectrum` -- the Stepišnik encoding power spectrum of a sequence.

All times in seconds; NumPy only.
"""

from __future__ import annotations

import numpy as np

from dmipy_sim.acquisition.timing import SequenceTiming
from dmipy_sim.constants import GAMMA

__all__ = ["SequenceTiming", "DEFAULT_TIMING", "encoding_mask", "stimulated_echo_mask", "encoding_spectrum"]

#: A typical clinical diffusion spin-echo budget (a 3 ms 90, a 6 ms 180 with its crushers, a 14 ms readout
#: tail before the echo): what a designer uses when the caller states none. It is a statement about the
#: SEQUENCE, not about a scanner; scanner limits come from dmipy-sim's catalogue.
DEFAULT_TIMING = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)


def encoding_mask(timing, TE, n_t, *, symmetric=False):
    """The spin-echo encoding windows of ``timing`` on an ``n_t`` grid over ``[0, TE]``.

    Returns ``(on, echo_idx)``: ``on`` is ``(n_t, 1)`` float, 1 where the gradient may live (the pre- and the
    post-180 window) and 0 in the lead-in, across the 180 and in the readout tail; ``echo_idx`` is the 180's
    sample (``TE/2``). The two windows are generally UNEQUAL because ``t_lead != t_readout_pre_echo``, so a
    pre/post asymmetry of the optimised waveform is a consequence of the budget, never a knob.

    ``symmetric`` (the VANILLA waveform): mirror the windows about the echo -- both reach the same extent
    from the 180, the surplus of the longer real window becoming dead time the spins spend transverse. This
    is the conventional symmetric waveform: the cost of refusing the budget's natural asymmetry.
    """
    TE = timing.resolve_TE(TE)
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    on = timing.on_mask(t, TE).astype(np.float64)
    echo = TE / 2.0
    if symmetric:
        pre_dur = (echo - timing.t_refocus / 2.0) - timing.t_lead
        post_dur = (TE - timing.t_readout_pre_echo) - (echo + timing.t_refocus / 2.0)
        W = max(0.0, min(pre_dur, post_dur))                     # mirror extent from the 180
        on[t < echo - timing.t_refocus / 2.0 - W] = 0.0          # dead-time the longer side
        on[t > echo + timing.t_refocus / 2.0 + W] = 0.0
    return on[:, None], int(round(echo / dt))


def stimulated_echo_mask(timing, TM, TE, n_t):
    """The two matched transverse windows of a stimulated echo on an ``n_t`` grid over ``[0, TE]``.

    dmipy-sim's stimulated-echo assembler: the excitation, a lead ``d``, the first encoding period, the store
    (a 90 of ``t_excite``), the mixing time ``TM`` on z, the recall (another), the second period, the same
    lead before the echo -- so the time transverse before the store equals the time after the recall (the
    PGSTE analogue of a 180 at TE/2, which is what refocuses a static field) and ``TE = 2 t_store + TM``.
    Returns ``(on, recall_idx, store_idx)``: the mask, the recall's sample (where the effective gradient's sign
    flips: the conjugation the store/recall pair applies) and the store's.
    """
    TM, TE = float(TM), float(TE)
    w = float(timing.t_excite)
    d = max(float(timing.t_lead), float(timing.t_readout_pre_echo))
    tau = (TE - TM - w) / 2.0 - d                                 # each transverse encoding period
    if tau <= 0.0:
        raise ValueError(f"TE = {TE*1e3:.2f} ms leaves no room for the two encoding periods around TM = "
                         f"{TM*1e3:.1f} ms (the budget needs {(TM + w + 2*d)*1e3:.2f} ms before any encoding)")
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    t_store = d + tau + w / 2.0
    t_recall = t_store + TM
    on = np.zeros(n_t, dtype=np.float64)
    on[(t >= d) & (t < t_store - w / 2.0)] = 1.0                 # first encoding period
    on[(t >= t_recall + w / 2.0) & (t < TE - d)] = 1.0           # second, the same length
    on[-1] = 0.0                                                 # the readout sample acts over nothing
    return on[:, None], int(round(t_recall / dt)), int(round(t_store / dt))


def encoding_spectrum(seq):
    """Encoding power spectrum |q̃(f)|² of a sequence's effective gradient, and its summary.

    The rigorous spectral-content quantity (Stepišnik): the diffusion signal is ``ln S ≈ −∫ D(ω)·|q̃(ω)|² dω``,
    so a waveform is characterised -- for ANY shape, pure or broadband -- by this spectrum, not by a nominal
    "frequency". Returns ``(freqs_Hz, power, centroid_Hz, bandwidth_Hz, rms_Hz)`` (one-sided) for the first
    measurement of ``seq`` (a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`).
    """
    G = np.asarray(seq.G_eff, dtype=np.float64)[0]                # (n_t, 3), the pulses folded in
    dt = float(seq.dt)
    q = GAMMA * np.cumsum(G, axis=0) * dt                          # (n_t,3) rad/m
    P = np.sum(np.abs(np.fft.rfft(q, axis=0)) ** 2, axis=1)        # (nf,) power
    f = np.fft.rfftfreq(G.shape[0], dt)
    Psum = P.sum() + 1e-30
    centroid = float((f * P).sum() / Psum)
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * P).sum() / Psum))
    rms = float(np.sqrt((GAMMA ** 2 * np.sum(G ** 2)) / (np.sum(q ** 2) + 1e-30)) / (2 * np.pi))
    return f, P, centroid, bandwidth, rms
