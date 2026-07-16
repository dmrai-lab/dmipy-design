"""dmipy-design — hardware-constrained diffusion-MRI gradient-waveform design.

Design deliverable gradient waveforms for diffusion MRI under real scanner limits, in the
**instant-pulse** approximation (ideal hard RF):

- **NOW** (``design_waveform_now``) — the SQP design oracle: maximise the b-value of a
  direct gradient waveform for any b-tensor shape (LTE / PTE / STE, and OGSE via a spectral
  constraint) under the full deliverability set (slew / amplitude / M1 / M2 / Maxwell /
  spectral / PNS-SAFE / heat), with machine-precision constraints.
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
    "StimulatedEchoTiming",
    "design_stimulated_echo",
]

__version__ = "0.1.0"
