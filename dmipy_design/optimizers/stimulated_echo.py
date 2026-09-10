"""PGSTE (stimulated-echo) diffusion-encoding waveform design.

PGSTE is the one diffusion family that is NOT a spin echo, so it has its own structure -- but it **reuses
the entire encoding optimizer core unchanged** (the NOW core): the diffusion encoding is still described by
the same effective wavevector ``q(t) = γ·cumsum(s·g)·dt`` and b-tensor ``B = ∫ q qᵀ dt``. Only the
*layout* differs, and the layout is dmipy-sim's stimulated-echo assembler
(:func:`dmipy_design.optimizers.timing.stimulated_echo_mask`):

Spin echo            : 90 — encode — 180@TE/2 — encode — echo
Stimulated echo (STE): 90 — encode(τ₁) — 90(store) — TM(on z) — 90(recall) — encode(τ₃) — echo
                       └─ three 90s, NO 180 ────────────────────────────────────────────┘

* **No 180 -- the sign flip is the stimulated-echo conjugation.** The store/recall pair conjugates the
  stored grating, so in the ``q = γ·cumsum(s·g)`` picture ``s`` flips at the recall -- exactly where
  dmipy-sim's ``RFSchedule.sign`` flips it.
* **Mixing time TM, gradient OFF, magnetisation on z.** A long TM gives a long, T1-limited (not T2-limited)
  effective diffusion time; a static field accrues no phase during it.
* **Static refocus needs matched encoding periods τ₁ = τ₃** -- the PGSTE analogue of "180 at TE/2"; the
  layout enforces it by construction.

The result is a :class:`~dmipy_design.optimizers.now.NowDesign` whose ``to_sequence()`` is dmipy-sim's
:func:`~dmipy_sim.sequences.builders.from_pgste_waveform` of the designed physical gradient, carrying the
three pulses and the budget. Running it as a real 3×90 stimulated echo (RF playback) is dmipy-sim's job.
"""

from __future__ import annotations

from .timing import DEFAULT_TIMING, stimulated_echo_mask


def design_stimulated_echo(b_delta, *, limits, TM, TE, timing=DEFAULT_TIMING, n_t=140, **design_kwargs):
    """Design a PGSTE diffusion-encoding waveform under ``limits`` (a
    :class:`~dmipy_sim.acquisition.scanners.ScannerLimits`, or anything ``ScannerLimits.of`` resolves).

    The two matched encoding periods around the mixing time ``TM`` follow from the budget ``timing`` and
    ``TE`` (:func:`stimulated_echo_mask`); the NOW core then maximises b inside them, so b-tensor shape,
    refocusing, hardware limits and the optional M1 / M2 / Maxwell / spectral / PNS / heat constraints all apply
    unchanged. ``design_kwargs`` are forwarded to the core.
    """
    from .now import _validate_b_delta, _design_in_mask
    b_delta = _validate_b_delta(b_delta)                 # up front: the mask below raises ValueError too
    on, recall_idx, store_idx = stimulated_echo_mask(timing, TM, TE, n_t)
    return _design_in_mask(b_delta, limits=limits, TE=float(TE), n_t=n_t, on=on, sign_idx=recall_idx,
                           timing=timing, store_idx=store_idx, recall_idx=recall_idx, **design_kwargs)
