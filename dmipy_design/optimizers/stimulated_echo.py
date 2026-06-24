"""
Groundwork for the PGSTE (stimulated-echo) waveform designer.

PGSTE is the one diffusion family that is NOT a spin echo, so it needs its own
timing/structure — but, crucially, it **reuses the entire spin-echo optimizer
core unchanged** (`design_waveform`): the diffusion encoding is still described
by the same effective wavevector ``q(t) = γ·cumsum(s·g)·dt`` and b-tensor
``B = ∫ q qᵀ dt``.  Only the *structure* differs.

Spin echo            : 90 — encode — 180@TE/2 — encode — echo
Stimulated echo (STE): 90 — encode(τ₁) — 90(store) — TM(on z) — 90(recall) — encode(τ₃) — echo
                       └─ three 90s, NO 180 ────────────────────────────────────────────┘

What PGSTE changes (and how it maps onto the existing core)
-----------------------------------------------------------
* **No 180 — the sign flip is the stimulated-echo conjugation.** The store/recall
  90 pair conjugates the stored grating, which acts as the effective sign flip in
  ``q``.  So in the ``q = γ·cumsum(s·g)`` picture, ``s`` flips once in the
  *gradient-off middle* (store/TM/recall) — exactly the role ``echo_idx`` plays in
  the spin-echo core, just placed at the storage rather than at TE/2.  (Feeding a
  PGSTE design to the spin-echo ``from_btensor_waveform`` therefore correctly
  trips the off-centre-180 guard — PGSTE must be *played* as a 3×90, not a 180.)
* **Mixing time TM, gradient OFF, magnetization on z.** During TM the spins are
  stored along z: ``q`` is held flat (cumsum of g=0), so a long TM gives a long,
  **T1-limited (not T2-limited)** effective diffusion time — PGSTE's whole reason
  to exist.  Static field (susceptibility/off-resonance) accrues NO phase during
  TM (it is the transverse periods τ₁/τ₃ that matter).
* **Static refocus needs MATCHED encoding periods τ₁ = τ₃** — the PGSTE analog of
  "180 at TE/2".  The stimulated echo refocuses static field only when the two
  transverse periods match; ``StimulatedEchoTiming`` enforces τ₁ = τ₃ by
  construction and guards that the budget leaves room for them.
* **The 1/2 storage factor and T1 weighting** are signal/relaxation effects
  (handled at replay time by the dmipy-sim stimulated-echo Bloch engine), not
  encoding-design constraints.

Status (groundwork)
-------------------
* ``StimulatedEchoTiming`` (below) — the timing budget + masks + matched-period
  guard.  CPU, testable now.
* ``design_stimulated_echo`` — a thin wrapper that builds the masks and calls the
  validated ``design_waveform`` core (duck-typed ``timing``).  It produces the
  *effective encoding* (q, b-tensor, shape, hardware, optional M1/M2/Maxwell/
  spectral all work identically).
* NOT yet done (deliberate, the build-out): GPU validation of the produced PGSTE
  encodings, and a dmipy-sim ``from_pgste_waveform`` *playback* builder that lays
  the designed lobes into a real 3×90 stimulated echo (dmipy-sim already has
  ``from_pgste`` + the stimulated-echo Bloch engine to extend).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StimulatedEchoTiming:
    """PGSTE timing budget — the stimulated-echo analog of ``SequenceTiming``.

    Diffusion encoding lives in the two matched transverse periods τ₁ (after the
    excitation, before the store) and τ₃ (after the recall, before the echo),
    straddling the mixing time TM during which the gradient is OFF (spins on z).
    τ₁ = τ₃ by construction (static-field refocus).  All times in seconds.

        [excite] [== τ₁ ==] [store] [===== TM (g off, on z) =====] [recall] [== τ₃ ==] echo
        0        t_lead              ...                                            TE

    The effective sign flip (the stimulated-echo conjugation, the ``echo_idx`` the
    optimizer core uses) sits in the gradient-off middle, where ``q`` is held — so
    its exact position there is immaterial to the encoding.
    """
    t_excite: float                 # excitation 90 duration; τ₁ starts after it
    TM: float                       # mixing time (gradient off, magnetization on z)
    t_store: float = None           # store-90 duration (default = t_excite)
    t_recall: float = None          # recall-90 duration (default = t_excite)
    t_readout_pre_echo: float = 0.0  # readout-start → echo (post-echo readout; informational)
    t_prep: float = 0.0             # optional fat-sat/prep before excitation
    TE: float | None = None         # native echo time (optional); masks() default

    def __post_init__(self):
        if self.t_store is None:
            self.t_store = self.t_excite
        if self.t_recall is None:
            self.t_recall = self.t_excite

    @property
    def t_lead(self) -> float:
        """Dead time from t=0 until τ₁ encoding may begin."""
        return self.t_prep + self.t_excite

    @property
    def _dead_middle(self) -> float:
        """Total gradient-off middle: store-90 + TM + recall-90."""
        return self.t_store + self.TM + self.t_recall

    def min_TE(self) -> float:
        """Smallest TE for which the matched encoding periods exist (τ → 0)."""
        return self.t_lead + self._dead_middle

    def tau(self, TE) -> float:
        """Matched encoding period τ₁ = τ₃ implied by a given TE."""
        return (float(TE) - self.t_lead - self._dead_middle) / 2.0

    def effective_diffusion_time(self, TE) -> float:
        """≈ time between the τ₁ and τ₃ encoding centroids — set mostly by TM
        (this is why PGSTE reaches long, T1-limited diffusion times)."""
        return self.TM + self.t_store / 2.0 + self.t_recall / 2.0 + self.tau(TE)

    def masks(self, TE=None, n_t=256):
        """Return ``(slew_off_mask (n_t,1) float, sign_flip_idx int)`` for a TE.

        Encoding ON only in the two matched τ windows; OFF during excitation, the
        store/recall 90s, and the whole TM.  ``sign_flip_idx`` (the conjugation,
        used by the optimizer core as ``echo_idx``) is the midpoint of the
        gradient-off middle.  Duck-compatible with ``design_waveform(timing=…)``.
        """
        TE = float(TE if TE is not None else self.TE)
        tau = self.tau(TE)
        if tau <= 0:
            raise ValueError(
                f"TE={TE*1e3:.2f} ms leaves no room for matched encoding periods "
                f"(min_TE={self.min_TE()*1e3:.2f} ms for TM={self.TM*1e3:.1f} ms).")
        dt = TE / (n_t - 1)
        t = np.arange(n_t) * dt
        tau1_end = self.t_lead + tau
        tau3_start = tau1_end + self._dead_middle
        on = np.zeros(n_t, dtype=np.float64)
        on[(t >= self.t_lead) & (t < tau1_end)] = 1.0          # τ₁ encode
        on[(t >= tau3_start) & (t < tau3_start + tau)] = 1.0   # τ₃ encode
        sign_flip = 0.5 * (tau1_end + tau3_start)              # in the dead middle
        return on[:, None], int(round(sign_flip / dt))


def design_stimulated_echo(b_delta, *, TM, TE, t_excite=2e-3,
                           t_store=None, t_recall=None, t_prep=0.0,
                           **design_kwargs):
    """Design a PGSTE diffusion-encoding waveform (GROUNDWORK).

    Builds a :class:`StimulatedEchoTiming` for the given mixing time ``TM`` and
    echo time ``TE`` and hands it to the validated ``design_waveform`` core via
    its duck-typed ``timing`` interface — so b-tensor shape, refocusing, hardware
    limits, and the optional M1/M2/Maxwell/spectral constraints all apply
    unchanged.  Returns a :class:`WaveformDesign` whose gradient is the *effective*
    PGSTE encoding (the long-TM hold gives the long, T1-limited diffusion time).

    NOTE (build-out, not yet done): the result is an encoding, not a runnable
    sequence.  Playing it requires a dmipy-sim stimulated-echo builder that lays
    these lobes into a real 3×90 PGSTE (the spin-echo ``from_btensor_waveform``
    will — correctly — refuse it via the off-centre-180 guard).  GPU validation
    of these encodings is also pending.
    """
    from .waveform_designer import design_waveform
    timing = StimulatedEchoTiming(t_excite=t_excite, TM=TM, t_store=t_store,
                                  t_recall=t_recall, t_prep=t_prep, TE=TE)
    return design_waveform(b_delta, timing=timing, TE=TE, **design_kwargs)
