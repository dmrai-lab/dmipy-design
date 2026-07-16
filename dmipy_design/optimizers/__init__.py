"""Gradient-waveform designers and shared timing utilities.

- ``design_waveform_now`` / ``NowDesign`` — the NOW SQP design oracle (LTE/PTE/STE/OGSE).
- ``SequenceTiming`` / ``encoding_spectrum`` — solver-agnostic encoding-window timing
  (incl. the derived pre/post-180 asymmetry) and the encoding power spectrum.
- ``StimulatedEchoTiming`` / ``design_stimulated_echo`` — PGSTE design via NOW.
"""
from .timing import SequenceTiming, encoding_spectrum
from .now import design_waveform_now, NowDesign
from .stimulated_echo import StimulatedEchoTiming, design_stimulated_echo

__all__ = [
    "SequenceTiming",
    "encoding_spectrum",
    "design_waveform_now",
    "NowDesign",
    "StimulatedEchoTiming",
    "design_stimulated_echo",
]
