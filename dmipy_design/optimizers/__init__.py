"""Gradient-waveform designers and the shared timing utilities.

- ``design_waveform_now`` / ``NowDesign`` — the NOW SQP design oracle (LTE/PTE/STE/OGSE).
- ``SequenceTiming`` (dmipy-sim's budget) / ``DEFAULT_TIMING`` / ``encoding_mask`` / ``stimulated_echo_mask`` /
  ``encoding_spectrum`` — a budget read on a grid (incl. the derived pre/post-180 asymmetry) and the encoding
  power spectrum.
- ``design_stimulated_echo`` — PGSTE design via the NOW core.
- ``design_refocusing_rf`` / ``RfPulseDesign`` — B1-robust, deliverable 180° RF envelope
  design via a Bloch forward (the RF analogue of NOW's gradient box).
"""
from .timing import SequenceTiming, DEFAULT_TIMING, encoding_mask, stimulated_echo_mask, encoding_spectrum
from .now import design_waveform_now, NowDesign
from .min_te import min_te_for_b
from .stimulated_echo import design_stimulated_echo
from .rf_pulse import design_refocusing_rf, RfPulseDesign

__all__ = [
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
