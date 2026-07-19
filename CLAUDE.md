# dmipy-design — Agent Guide

**Read this file, not the whole tree.** dmipy-design designs deliverable diffusion gradient
waveforms under scanner limits, in the **instant-pulse approximation** (ideal hard RF). It is
part of the [dmipy](https://github.com/dmrai-lab/dmipy) ecosystem — it produces waveforms that
[dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) can simulate and that export to a scanner via
Pulseq.

## Scope — what belongs here (and what does not)

**Here:**
- **NOW** — the SQP gradient-waveform design oracle (`optimizers/now.py`).
- **Timing** — encoding-window budget + the derived pre/post-180 asymmetry (`optimizers/timing.py`).
- **PGSTE** — stimulated-echo diffusion encoding through NOW (`optimizers/stimulated_echo.py`).
- **Pulseq I/O** — scanner-runnable `.seq` export + offline checks (`pulseq_export.py`).
- **Scanner constraints** — hardware/time limits + the SAFE PNS model (`constraints.py`, `now.py`).

**NOT here (out of the instant-pulse public scope):**
- Finite-RF-pulse optimization (this package assumes ideal instantaneous pulses).
- CRLB / Fisher-information optimal experiment design.
- Column-generation multishell protocol design.

Do not add any of the above here — this package is deliberately scoped to the instant-pulse,
deliverable-waveform layer.

## NOW is the design oracle — use it

`optimizers/now.py::design_waveform_now` maximises `b = gᵀQg` with active-set SQP (scipy SLSQP),
constraints expressed the way NOW expresses them — LINEAR matrices (per-axis slew, refocus
`q(TE)=0`, moments M1/M2), amplitude as box bounds, nonlinear b-tensor **shape** + Maxwell +
OGSE-spectral, with **analytic** objective + constraint Jacobians (NO autodiff). One solver, all
shapes: **LTE / PTE / STE** (`b_delta = 1 / -0.5 / 0`), and **OGSE** via `spectral_freq`. The full
deliverability set: slew / amplitude / M1 / M2 / shape / Maxwell / spectral / **PNS (SAFE)** /
**heat (∫g²)**. Returns a `NowDesign` with `.effective_G()` / `.to_sim_waveform()`.

**Two modes.** `design_waveform_now` maximises b at a **fixed TE**. The inverse —
`min_te_for_b` (`optimizers/min_te.py`) — gives a *required* b-value the **shortest TE** that
reaches it (the SNR-optimal design: shorter TE ⇒ less T2 decay). It is a thin wrapper that
**bisects TE around the same max-b primitive** (achievable b is monotonic in TE); it does not
add a second solver. Returns `(NowDesign, te)`.

**Why NumPy + SciPy, not JAX:** active-set SQP is sequential (bad for vmap/autodiff), and feeding
scipy a JAX-autodiff objective crosses a per-iteration numpy↔JAX bridge that dominates wall-clock.
Pure-numpy-analytic derivatives → scipy calls NumPy directly (LTE ~1 s). For scipy problems go
fully numpy-analytic; never scipy-over-autodiff. **Do not introduce a JAX dependency here.**

## Robustness constraints — ON by default

`null_M1` (velocity), `null_M2` (acceleration), `maxwell` (concomitant field) default on: a
*needed-but-off* constraint biases the measurement, while an unneeded one only costs b (SNR). Turn
one off only when its confound is absent. All indices (`m1_index`, `m2_index`, `maxwell_index`)
are always reported. Fully-constrained designs are hardest to converge — raise `n_restarts` if
`feasible=False`.

## Timing → asymmetric encoding windows (no asymmetry knob)

Asymmetry is a *consequence* of the sequence timing, not a choice. `SequenceTiming(t_excite,
t_refocus, t_readout_pre_echo, t_prep)` holds the physical budget; the 180 is pinned at TE/2 and
the encoding is OFF during the excitation lead-in, the 180, and the readout tail. Because lead-in ≠
readout-pre-echo (and partial Fourier shortens the post-180 window), the pre/post windows come out
unequal — the optimum is asymmetric as an *output*. Build from a readout
(`SequenceTiming.from_readout(..., partial_fourier=…)`) or a scanner sequence
(`SequenceTiming.from_pulseq(seq)`). `symmetric=True` gives the conventional (dead-timed) waveform.

## PGSTE (`optimizers/stimulated_echo.py`)

PGSTE reuses the NOW core unchanged (same q / b-tensor); only the structure differs (3×90
stimulated echo, no 180). `StimulatedEchoTiming` enforces matched transverse periods `τ₁ = τ₃`
around a gradient-off mixing time TM (long, T1-limited diffusion time). `design_stimulated_echo`
wraps `design_waveform_now` via the duck-typed `timing`. This module designs the *effective*
encoding; RF playback of a real 3×90 is out of the instant-pulse scope here.

## Spectral paradigm (OGSE)

OGSE and PGSE share the same (linear) b-tensor; what distinguishes OGSE is spectral content, which
the b-tensor can't see. `design_waveform_now` always reports the encoding power spectrum
(`spectral_rms`), and `spectral_freq=f` drives the RMS encoding frequency to `f`. Per
`ln S ≈ −∫ D(ω)|q̃(ω)|² dω`, `encoding_spectrum` (in `timing.py`) is the FFT-based
centroid/bandwidth/rms post-hoc quantity.

## Pulseq (`pulseq_export.py`)

`design_to_pulseq(design, scanner=…)` builds a native spin echo on real `{Gmax, slew, raster,
dead-time}` limits (from dmipy-sim's `PULSEQ_SYSTEMS`), resampling the designed gradient onto the
gradient raster and placing the 180 in the reserved TE/2 gap. `pulseq_delivery_report` /
`pulseq_pns_report` run the offline acceptance checks (timing, realized Gmax/slew, b-tensor
round-trip, SAFE PNS). Needs the `[pulseq]` extra (dmipy-sim + pypulseq).

## Module map

| File | Role |
|------|------|
| `optimizers/now.py` | `design_waveform_now`, `NowDesign` — the SQP design oracle (LTE/PTE/STE/OGSE) |
| `optimizers/min_te.py` | `min_te_for_b` — shortest TE for a target b (SNR-optimal), bisecting NOW |
| `optimizers/timing.py` | `SequenceTiming` (asymmetric windows), `encoding_spectrum` — NumPy-only, JAX-free |
| `optimizers/stimulated_echo.py` | `StimulatedEchoTiming`, `design_stimulated_echo` — PGSTE via NOW |
| `pulseq_export.py` | `.seq` export + delivery/PNS reports (needs `[pulseq]`) |
| `constraints.py` | `HardwareConstraints`, `TimeConstraints` |

## Tests

```bash
pytest -q                                   # NOW / timing / PGSTE (NumPy+SciPy only)
pytest -q tests/test_pulseq_export.py       # Pulseq round-trip (needs [pulseq]; skips otherwise)
```

Correctness is defined by the suite: NOW designs are feasible with machine-precision constraint
residuals; LTE reaches higher b than STE; `spectral_freq` drives `spectral_rms` to target; the
timing windows are asymmetric-by-derivation; the assembled `.seq` encodes the designed b-tensor.
