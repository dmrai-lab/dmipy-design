"""dmipy-design — hardware-constrained diffusion-MRI gradient-waveform design.

Design deliverable gradient waveforms for diffusion MRI under real scanner limits, in the
**instant-pulse** approximation (ideal hard RF). This package is the OPTIMISATION; what it optimises
inside and what it produces are dmipy-sim's:

- the scanner's limits are :class:`dmipy_sim.acquisition.scanners.ScannerLimits` (the cited catalogue; every
  designer takes one as ``limits``);
- the timing budget is :class:`dmipy_sim.acquisition.timing.SequenceTiming` (re-exported here);
- a design's ``to_sequence()`` is a :class:`dmipy_sim.acquisition.scanner_sequence.ScannerSequence`, the one
  acquisition object dmipy-sim simulates, replays and exports.

- **NOW** (``design_waveform_now``) — the SQP design oracle: maximise the b-value of a
  direct gradient waveform for any b-tensor shape (LTE / PTE / STE, and OGSE via a spectral
  constraint) under the full deliverability set (slew / amplitude / M1 / M2 / Maxwell /
  spectral / PNS-SAFE / heat), with machine-precision constraints.
- **min-TE** (``min_te_for_b``) — the SNR-optimal inverse: given a *required* b-value, find
  the shortest TE that still reaches it (shorter TE ⇒ less T2 decay ⇒ higher SNR), by
  bisecting TE around the NOW max-b primitive.
- **PGSTE** (``design_stimulated_echo``) — stimulated-echo diffusion encoding through the
  same NOW core (matched τ₁ = τ₃ periods around a long, T1-limited mixing time).
- **Refocusing RF** (``design_refocusing_rf``) — beyond the instant-pulse approximation:
  design a B1-robust, deliverable 180° RF *envelope* (band-limited + peak-B1 + SAR) via a
  Bloch forward over a transmit-inhomogeneity × off-resonance ensemble; ``to_rf_event`` puts it
  in a schedule.
- **Encoding masks** (``encoding_mask``, ``stimulated_echo_mask``) — a budget read on a grid; the pre/post-180
  window asymmetry is a *derived consequence* of the scanner timing, not a free knob.
- **Pulseq I/O** (``dmipy_design.pulseq_export``) — dmipy-sim's export of a design's sequence to a
  scanner-runnable ``.seq``, and offline checks (timing, realized Gmax/slew, b-tensor round-trip, PNS).
  Requires the ``[pulseq]`` extra.

The NOW / timing / PGSTE core needs NumPy + SciPy and dmipy-sim (the object).
"""
from dmipy_sim.acquisition.scanners import ScannerLimits

from .constraints import TimeConstraints
from .optimizers import (
    SequenceTiming,
    DEFAULT_TIMING,
    encoding_mask,
    stimulated_echo_mask,
    encoding_spectrum,
    design_waveform_now,
    NowDesign,
    min_te_for_b,
    design_stimulated_echo,
    design_refocusing_rf,
    RfPulseDesign,
)

__all__ = [
    "ScannerLimits",
    "TimeConstraints",
    "SequenceTiming",
    "DEFAULT_TIMING",
    "encoding_mask",
    "stimulated_echo_mask",
    "encoding_spectrum",
    "design_waveform_now",
    "NowDesign",
    "min_te_for_b",
    "design_stimulated_echo",
    "design_refocusing_rf",
    "RfPulseDesign",
]

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("dmipy-design")
except Exception:   # not installed (e.g. run from a source tree)
    __version__ = "unknown"
