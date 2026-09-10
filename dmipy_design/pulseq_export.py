"""Tier-2 deliverability: export a designed sequence to a scanner-runnable Pulseq spin echo and check it
against the vendor's limits + a b-tensor round-trip -- no scanner required.

A design's ``to_sequence()`` is a dmipy-sim :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`
built to its timing budget -- the physical gradient with the pulses in their own windows -- so the export IS
dmipy-sim's :func:`~dmipy_sim.sequences.pulseq.to_pulseq` on that object, on the scanner from dmipy-sim's cited
catalogue: nothing is resampled or re-derived here. ``pulseq_delivery_report`` then runs the checks a scanner
would at acceptance time that we CAN do offline:

  * ``seq.check_timing()``           -- raster / dead-time / block-contiguity,
  * realized peak |G| and |slew| on the fine raster vs the system limits,
  * the b-tensor recomputed from the assembled .seq vs the design (what we asked == what actually gets encoded);

and ``pulseq_pns_report`` the SAFE PNS prediction (Tier 3). NOT covered (scanner-free later tiers):
gradient thermal / duty-cycle, eddy-current / GIRF fidelity.
"""
from __future__ import annotations

import numpy as np

from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.constants import GAMMA
from dmipy_sim.sequences.pulseq import GAMMA_HZ, _require_pypulseq, make_system, to_pulseq

def design_to_pulseq(design, *, scanner='siemens_prisma', filename=None, system=None):
    """Write a design as a scanner-runnable Pulseq spin echo (a pypulseq ``Sequence``, written to ``filename``
    if given): dmipy-sim's export of ``design.to_sequence()`` on ``scanner`` (anything ``ScannerLimits.of``
    resolves, for its amplitude and slew limits).

    The design's grid must sit on the scanner's gradient raster (``dt`` a whole number of rasters) -- dmipy-sim
    refuses otherwise, naming the grids that would fit; a designer that wants a runnable file designs on the
    raster. Each step is then played as a linear ramp between its samples (what a scanner does with a coarser
    grid), which is why the delivered b differs from the designed step function by a few percent."""
    return to_pulseq(design.to_sequence(), system=system or make_system(scanner), filename=filename)


def seq_btensor(seq, *, gamma_hz=GAMMA_HZ, n=8000):
    """b-tensor (s/m^2, 3x3) recomputed from an assembled Pulseq ``Sequence`` by integrating its realized
    gradient waveforms with the 180 sign flip folded in."""
    wave, _, t_refocus, *_ = seq.waveforms_and_times()
    t_end = max((w[0, -1] for w in wave if w is not None and w.shape[1]), default=0.0)
    if t_end <= 0:
        return np.zeros((3, 3))
    t = np.linspace(0.0, t_end, n)
    dt = t[1] - t[0]
    G = np.zeros((n, 3))                                  # T/m
    for ax, w in enumerate(wave):
        if w is not None and w.shape[1] >= 2:
            G[:, ax] = np.interp(t, w[0], w[1], left=0.0, right=0.0) / gamma_hz
    t180 = float(np.ravel(t_refocus)[0]) if np.size(t_refocus) else t_end / 2.0
    s = np.where(t < t180, 1.0, -1.0)[:, None]
    q = GAMMA * np.cumsum(s * G, axis=0) * dt             # rad/m
    return (q[:, :, None] * q[:, None, :]).sum(0) * dt


def pulseq_pns_report(seq, *, hardware=None, time_range=None):
    """Tier-3 PNS prediction via the SAFE model -- the SAME model the scanner uses to accept/reject a sequence
    (Hebrank/Schulte; IEC 60601-2-33).

    Runs ``seq.calculate_pns`` on the assembled .seq. ``hardware`` is a SAFE coefficient namespace; default is
    pypulseq's representative Siemens-class example (``safe_example_hw``) -- NOT the exact Prisma .asc (which
    is vendor-confidential), so read the % as indicative, not certified. Returns max PNS as % of the
    stimulation limit (100% = limit; clinical normal mode ~80%) and the per-axis breakdown.
    """
    from pypulseq.utils.safe_pns_prediction import safe_example_hw
    hw = hardware if hardware is not None else safe_example_hw()
    res = seq.calculate_pns(hw, time_range=time_range, do_plots=False)
    ok = bool(res[0])
    pns_norm = np.asarray(res[1], dtype=float)            # total, normalized to the limit
    comp = np.asarray(res[2], dtype=float) if len(res) > 2 else None   # per-axis (n,3)
    per_axis = (np.max(np.abs(comp), axis=0) * 100.0).tolist() if comp is not None \
        and comp.ndim == 2 and comp.shape[1] == 3 else None
    return dict(pns_ok=ok, pns_max_pct=float(np.max(pns_norm)) * 100.0,
                pns_per_axis_pct=per_axis, hardware=getattr(hw, 'name', 'custom'))


def pulseq_delivery_report(design, seq, *, scanner='siemens_prisma'):
    """Run the offline acceptance checks: timing, realized peak Gmax/slew vs the scanner's cited limits, and
    the b-tensor round-trip (assembled .seq vs design)."""
    _require_pypulseq()
    ok, err = seq.check_timing()
    wave, *_ = seq.waveforms_and_times()
    gamma_hz = GAMMA_HZ
    # realized peak amplitude and slew on the fine raster (T/m, T/m/s)
    gmax = 0.0
    smax = 0.0
    for w in wave:
        if w is not None and w.shape[1] >= 2:
            g = w[1] / gamma_hz                           # T/m
            gmax = max(gmax, float(np.max(np.abs(g))))
            tt = w[0]
            dgs = np.diff(g) / np.maximum(np.diff(tt), 1e-12)
            smax = max(smax, float(np.max(np.abs(dgs))))
    B = seq_btensor(seq)
    b_seq = float(np.trace(B))
    lim = ScannerLimits.of(scanner)
    return dict(
        timing_ok=bool(ok), timing_error=err,
        b_design=float(design.b_value), b_seq=b_seq,
        b_rel_err=abs(b_seq - design.b_value) / (design.b_value + 1e-30),
        max_grad_mT=gmax * 1e3, limit_grad_mT=lim.G_max * 1e3,
        max_slew=smax, limit_slew=lim.slew_max,
        grad_ok=(gmax <= lim.G_max * 1.01),
        slew_ok=(smax <= lim.slew_max * 1.02),
    )
