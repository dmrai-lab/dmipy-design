from .gradient_based import gradient_oed
from .greedy_sequential import greedy_add_measurement
from .master_problem import solve_master
from .pricing_problem import solve_pricing, decode_pgse, decode_ogse, BVECS_30
from .column_generation import column_generation_oed, Atom, CGResult
# Shared, solver-agnostic encoding timing/spectrum (JAX-free).
from .timing import SequenceTiming, encoding_spectrum
# Design oracle: NOW SQP solver — best-b, machine-precision constraints (LTE/PTE/STE/OGSE,
# slew/amp/M1/M2/shape/Maxwell/PNS/heat).  The default waveform designer.
from .now import design_waveform_now, NowDesign
# Differentiable JAX augmented-Lagrangian solver — kept as the simulator-in-the-loop
# (co-optimization) engine, not as a competing designer.  min_te_for_b lives here too.
from .waveform_designer import design_waveform, min_te_for_b, WaveformDesign
from .stimulated_echo import (
    StimulatedEchoTiming, design_stimulated_echo, pgste_store_recall_idx)

__all__ = [
    "gradient_oed",
    "greedy_add_measurement",
    "solve_master",
    "solve_pricing",
    "decode_pgse",
    "decode_ogse",
    "BVECS_30",
    "column_generation_oed",
    "Atom",
    "CGResult",
    # design oracle (default)
    "design_waveform_now",
    "NowDesign",
    "SequenceTiming",
    "encoding_spectrum",
    # differentiable co-opt engine + helpers
    "design_waveform",
    "min_te_for_b",
    "WaveformDesign",
    "StimulatedEchoTiming",
    "design_stimulated_echo",
    "pgste_store_recall_idx",
]
