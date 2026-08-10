"""Joint gradient + B1 co-design through replay packs — the culminating substrate-informed optim.

A diffusion spin echo has two knobs the substrate cares about: the **gradient waveform** (sets the
diffusion contrast) and the **refocusing B1 pulse** (sets how much of the echo survives across the
transmit ``B1⁺`` × off-resonance ensemble). At leading order in a single spin echo these *factorize* —
diffusion is irreversible, so the 180's quality scales the echo but does not alter the diffusion contrast
between substrates. So "optimize waveforms and B1" is correctly a **pipeline** (the repo's co-opt
pattern): design the discriminating gradient by replaying the packs (``replay_design``), then design the
substrate-robust refocusing pulse (``optimizers.rf_pulse``), and report the combined substrate echo
``E = <refocusing efficiency>_ensemble x E_diffusion(pack)``.

The strongly-coupled regime — multiple refocusing pulses (CPMG/LASER), where the diffusion evolves
between pulses and imperfect refocusing seeds substrate-dependent stimulated-echo pathways — requires a
multi-echo (EPG-over-pack) vector-Bloch replay and is the documented next generalization. NumPy/SciPy
only; needs the ``[sim]`` extra (the replay engine).
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["codesign_waveform_and_b1", "CoDesignResult"]


@dataclass
class CoDesignResult:
    gradient: "object"       # DiscriminationResult (the discriminating gradient waveform)
    rf: "object"             # RfPulseDesign (the substrate-robust refocusing pulse)
    contrast_ideal: float    # |E_A - E_B| with an ideal 180
    refocus_efficiency: float   # <eta> over the B1+ x off-res ensemble (designed pulse)
    refocus_efficiency_hard: float
    contrast_delivered: float   # contrast x refocusing efficiency (the echo-observable contrast)


def codesign_waveform_and_b1(pack_a, pack_b, *, direction=(1.0, 0.0, 0.0), G_max=0.08, te=None,
                             slew_max=None, rf_duration=6e-3, B1_max=20e-6, b1_range=(0.7, 1.3),
                             off_resonance_hz=250.0, grad_kwargs=None, rf_kwargs=None):
    """Co-design a discriminating gradient waveform and a robust refocusing B1 pulse for the pair of
    substrates ``pack_a``/``pack_b``.

    Stage 1 — replay the packs to design the gradient maximizing ``|E_A - E_B|`` (see
    :func:`dmipy_design.replay_design.design_discriminating_waveform`; ``te``/``slew_max`` make it
    deliverable). Stage 2 — design the ``B1``-robust refocusing pulse maximizing the ensemble refocusing
    efficiency over ``(B1⁺ × off-resonance)`` (see
    :func:`dmipy_design.optimizers.rf_pulse.design_refocusing_rf`). The delivered contrast is the ideal
    diffusion contrast scaled by the refocusing efficiency. Returns a :class:`CoDesignResult`.
    """
    from dmipy_design.replay_design import design_discriminating_waveform
    from dmipy_design.optimizers.rf_pulse import design_refocusing_rf

    grad = design_discriminating_waveform(
        pack_a, pack_b, direction=direction, G_max=G_max, te=te, slew_max=slew_max,
        **(grad_kwargs or {}))
    rf = design_refocusing_rf(
        rf_duration=rf_duration, B1_max=B1_max, b1_range=b1_range,
        off_resonance_hz=off_resonance_hz, **(rf_kwargs or {}))
    eta = float(rf.refocusing_efficiency)
    return CoDesignResult(
        gradient=grad, rf=rf,
        contrast_ideal=grad.contrast,
        refocus_efficiency=eta,
        refocus_efficiency_hard=float(rf.refocusing_efficiency_hard),
        contrast_delivered=grad.contrast * eta)
