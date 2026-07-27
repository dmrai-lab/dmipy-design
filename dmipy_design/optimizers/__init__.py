"""Gradient-waveform designers and shared timing utilities.

- ``design_waveform_now`` / ``NowDesign`` — the NOW SQP design oracle (LTE/PTE/STE/OGSE).
- ``SequenceTiming`` / ``encoding_spectrum`` — solver-agnostic encoding-window timing
  (incl. the derived pre/post-180 asymmetry) and the encoding power spectrum.
- ``StimulatedEchoTiming`` / ``design_stimulated_echo`` — PGSTE design via NOW.
- ``design_refocusing_rf`` / ``RfPulseDesign`` — B1-robust, deliverable 180° RF envelope
  design via a Bloch forward (the RF analogue of NOW's gradient box).
"""
from .timing import SequenceTiming, encoding_spectrum
from .now import design_waveform_now, NowDesign
from .min_te import min_te_for_b
from .stimulated_echo import StimulatedEchoTiming, design_stimulated_echo
from .rf_pulse import design_refocusing_rf, RfPulseDesign

__all__ = [
    "SequenceTiming",
    "encoding_spectrum",
    "design_waveform_now",
    "NowDesign",
    "min_te_for_b",
    "StimulatedEchoTiming",
    "design_stimulated_echo",
    "design_refocusing_rf",
    "RfPulseDesign",
]
