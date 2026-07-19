"""dmipy-design — hardware-constrained diffusion-MRI gradient-waveform design.

Design deliverable gradient waveforms for diffusion MRI under real scanner limits, in the
**instant-pulse** approximation (ideal hard RF):

- **NOW** (``design_waveform_now``) — the SQP design oracle: maximise the b-value of a
  direct gradient waveform for any b-tensor shape (LTE / PTE / STE, and OGSE via a spectral
  constraint) under the full deliverability set (slew / amplitude / M1 / M2 / Maxwell /
  spectral / PNS-SAFE / heat), with machine-precision constraints.
- **min-TE** (``min_te_for_b``) — the SNR-optimal inverse: given a *required* b-value, find
  the shortest TE that still reaches it (shorter TE ⇒ less T2 decay ⇒ higher SNR), by
  bisecting TE around the NOW max-b primitive.
- **PGSTE** (``design_stimulated_echo``) — stimulated-echo diffusion encoding through the
  same NOW core (matched τ₁ = τ₃ periods around a long, T1-limited mixing time).
- **Timing** (``SequenceTiming``) — the physical encoding-window budget; the pre/post-180
  window asymmetry is a *derived consequence* of the scanner timing, not a free knob.
- **Pulseq I/O** (``dmipy_design.pulseq_export``) — export a design to a scanner-runnable
  ``.seq`` and check it offline (timing, realized Gmax/slew, b-tensor round-trip, PNS).
  Requires the ``[pulseq]`` extra.

The NOW / timing / PGSTE core needs only NumPy + SciPy. The pulseq export and the dmipy-sim
bridge (``NowDesign.to_sim_waveform``, ``SequenceTiming.from_pulseq``) use dmipy-sim; install
the ``[sim]`` / ``[pulseq]`` extras for those.
"""
from .constraints import HardwareConstraints, TimeConstraints
from .optimizers import (
    SequenceTiming,
    encoding_spectrum,
    design_waveform_now,
    NowDesign,
    min_te_for_b,
    StimulatedEchoTiming,
    design_stimulated_echo,
)

__all__ = [
    "HardwareConstraints",
    "TimeConstraints",
    "SequenceTiming",
    "encoding_spectrum",
    "design_waveform_now",
    "NowDesign",
    "min_te_for_b",
    "StimulatedEchoTiming",
    "design_stimulated_echo",
]

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("dmipy-design")
except Exception:   # not installed (e.g. run from a source tree)
    __version__ = "unknown"
