# dmipy-design

**Hardware-constrained diffusion-MRI gradient-waveform design** — part of the
[dmipy](https://github.com/dmrai-lab/dmipy) ecosystem (see [dmipy-sim](https://github.com/dmrai-lab/dmipy-sim)
for the forward Monte-Carlo engine and [dmipy-fit](https://github.com/dmrai-lab/dmipy-fit) for the
analytical inverse).

Design deliverable diffusion gradient waveforms under real scanner limits, in the
**instant-pulse** approximation (ideal hard RF). Given a b-tensor shape, a target echo time, and a
scanner's hardware limits, it returns the constraint-optimal gradient and can export it to a
scanner-runnable Pulseq `.seq`.

## What's here

- **NOW — the design oracle** (`design_waveform_now`). A NumPy + SciPy port of the NOW recipe
  (Sjölund et al. 2015; [jsjol/NOW](https://github.com/jsjol/NOW)): maximise `b = gᵀQg` with
  active-set SQP, constraints expressed exactly (per-axis slew and amplitude, refocus `q(TE)=0`,
  moments M1/M2, b-tensor **shape**, Maxwell, OGSE spectral, **PNS (SAFE model)**, heat `∫g²`),
  analytic objective + constraint Jacobians. One solver, all shapes — **LTE / PTE / STE**, and
  **OGSE** via a spectral-frequency constraint.
- **PGSTE** (`design_stimulated_echo`). Stimulated-echo diffusion encoding through the same NOW
  core: matched transverse periods `τ₁ = τ₃` around a long, **T1-limited** mixing time TM.
- **Timing & asymmetric windows** (`SequenceTiming`). The physical encoding-window budget. The
  pre/post-180 window **asymmetry is a derived consequence** of the scanner timing (lead-in ≠
  readout-pre-echo, partial Fourier), never a free knob; `symmetric=True` gives the conventional
  (dead-timed) waveform for comparison.
- **Pulseq I/O** (`dmipy_design.pulseq_export`). Export a design to a scanner-runnable spin-echo
  `.seq` on real vendor limits, and run the offline acceptance checks (timing, realized
  Gmax/slew, b-tensor round-trip, PNS via the SAFE model).
- **Scanner constraints** (`HardwareConstraints`, `TimeConstraints`, and the SAFE PNS model in
  the NOW solver); the full vendor catalogue (`PULSEQ_SYSTEMS`) comes from dmipy-sim.

**Scope (instant-pulse):** RF pulses are ideal and instantaneous. Finite-RF-pulse optimization
and CRLB/Fisher-information experiment design are **not** part of this package.

## Install

```bash
pip install dmipy-design                 # NOW + timing + PGSTE (NumPy/SciPy only)
pip install "dmipy-design[sim]"          # + dmipy-sim bridge (to_sim_waveform, from_pulseq)
pip install "dmipy-design[pulseq]"       # + scanner-runnable .seq export (pypulseq)
```

## Quickstart

```python
from dmipy_design import design_waveform_now, SequenceTiming

# a real timing budget from a readout (partial Fourier shortens the post-180 window)
timing = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=6e-3,
                                     readout_duration=30e-3, partial_fourier=0.75)

# max-b LTE waveform under Prisma-class limits, with M1/M2 nulled (default)
d = design_waveform_now(b_delta=1.0, G_max=0.08, slew_rate_max=200.0,
                        TE=0.08, timing=timing)
print(d.b_value, d.feasible, d.max_slew, d.refocus_residual)

# STE (isotropic) and an OGSE-like waveform at ~80 Hz
ste  = design_waveform_now(b_delta=0.0, TE=0.08)
ogse = design_waveform_now(b_delta=1.0, TE=0.08, spectral_freq=80.0)

# PGSTE with a long mixing time
from dmipy_design import design_stimulated_echo
pgste = design_stimulated_echo(b_delta=1.0, TM=50e-3, TE=0.12)
```

Export to a scanner-runnable `.seq` (needs `[pulseq]`):

```python
from dmipy_design.pulseq_export import design_to_pulseq, pulseq_delivery_report
seq = design_to_pulseq(d, scanner="siemens_prisma", filename="design.seq")
print(pulseq_delivery_report(d, seq, scanner="siemens_prisma"))
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q                        # NOW / timing / PGSTE (NumPy+SciPy)
pip install -e ".[pulseq,dev]"
pytest -q tests/test_pulseq_export.py   # the Pulseq round-trip
```

## License

See [LICENSE](LICENSE).
