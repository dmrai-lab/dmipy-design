# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

dmipy-design designs deliverable diffusion-MRI acquisitions under real scanner limits.

- **Gradient waveforms.** `design_waveform_now` maximises b for any b-tensor shape (LTE, PTE, STE, OGSE) under slew,
  amplitude, moment, Maxwell, spectral, PNS (SAFE) and heat constraints; `min_te_for_b` finds the shortest TE for a
  required b; `design_stimulated_echo` designs PGSTE encoding around a T1-limited mixing time.
- **Refocusing RF.** `design_refocusing_rf` designs a B1-robust, peak-B1 and SAR-bounded 180 degree envelope over a
  transmit-inhomogeneity by off-resonance ensemble, from a built-in or a given warm start.
- **Through the substrate.** Replay packs are the model of the tissue: `design_discriminating_waveform` finds the
  deliverable gradient that separates two substrates, `pgse_response_lut` answers the PGSE case as a table lookup, and
  `codesign_waveform_and_b1` designs the gradient and the refocusing pulse together.
- **Certificates.** `certify` states the replay band K of a pack as a bandwidth, certified over the set of waveforms a
  scanner class can play.
- **Limits and output.** Every designer takes `limits=` (dmipy-replay's cited `ScannerLimits`, a certificate class, or
  a `(G_max, slew)` envelope) and returns a `ScannerSequence` built to its timing budget; Pulseq export writes a
  scanner-runnable `.seq`.
- **Pairs with** dmipy-replay `>=0.2,<0.3` and dmipy-sim `>=3,<4`; the `interop` extra adds dmipy-fit `>=3,<4`, the
  `pulseq` extra pypulseq `>=1.4`. Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10.
- **Licence** AGPL-3.0-only or commercial.
