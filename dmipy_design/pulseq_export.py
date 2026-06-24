"""Tier-2 deliverability: export a designed waveform to a scanner-runnable Pulseq
spin echo and check it against the vendor's limits + a b-tensor round-trip -- no
scanner required.

``design_to_pulseq`` builds a NATIVE spin echo

    [90] -- [pre-180 diffusion gradient] -- [180] -- [post-180 diffusion gradient] -- [ADC]

on REAL {Gmax, slew, raster, dead-time} limits (from dmipy-sim's PULSEQ_SYSTEMS),
resampling the designed gradient onto the scanner gradient raster (10 us) and placing
the 180 in the gradient-off gap the design reserves at TE/2.  Unlike the v1
``dmipy_sim ... to_pulseq`` (which emits the *effective* gradient in one block with the
180 as metadata, exact for round-trip but not runnable), this is a real spin echo.

``pulseq_delivery_report`` then runs the checks a scanner would at acceptance time that
we CAN do offline:
  * ``seq.check_timing()``           -- raster / dead-time / block-contiguity,
  * realized peak |G| and |slew| on the fine raster vs the system limits,
  * the b-tensor recomputed from the assembled .seq vs the design (what we asked ==
    what actually gets encoded).

NOT covered here (later tiers, still scanner-free): PNS (SAFE model, Tier 3), gradient
thermal/duty-cycle, eddy-current/GIRF fidelity (Tier 4).  RF is a hard block pulse;
slice-selective RF + readout train are refinements that do not change the gradient
deliverability or the b-tensor.
"""
from __future__ import annotations

import numpy as np

from dmipy_sim.sequences.pulseq import (
    make_system, GAMMA_HZ, _require_pypulseq, PULSEQ_SYSTEMS)

GAMMA = 267.513e6  # rad/s/T  (matches dmipy_sim.constants.GAMMA)

# realistic Siemens-class RF/ADC dead times so check_timing is meaningful
_DEAD = dict(rf_dead_time=100e-6, rf_ringdown_time=60e-6, adc_dead_time=10e-6)


def _ramp_to_zero(v, raster, slew_max, leading):
    """A slew-limited linear ramp between 0 and vector ``v`` (3,), on ``raster``; returns
    (n,3) NOT including the ``v`` endpoint.  ``leading`` True -> 0->v (prepended),
    False -> v->0 (appended).  Needed because the design HOLDS the gradient at the 180
    window (it zeros slew, not amplitude), so a split lobe does not naturally reach 0
    there -- we add the ramp the design omitted, at <= slew_max."""
    vn = float(np.linalg.norm(v))
    n = int(np.ceil(vn / (0.9 * slew_max * raster)))      # 0.9 margin
    if n <= 0:
        return np.zeros((0, 3))
    frac = (np.arange(n) / n) if leading else (1.0 - np.arange(1, n + 1) / n)
    return frac[:, None] * v[None, :]


def _resample_lobe(seg, dt, raster, slew_max):
    """Resample a (k,3) gradient segment from grid ``dt`` onto ``raster`` and bracket it
    with slew-limited ramps to 0 at both ends (Pulseq requires arbitrary gradients to
    start and end at 0; the design's lobe does not, at the 180 side)."""
    k = seg.shape[0]
    if k < 2:
        return None
    t_in = np.arange(k) * dt
    n_r = max(2, int(round(t_in[-1] / raster)) + 1)
    t_r = np.arange(n_r) * raster
    core = np.zeros((n_r, 3))
    for ax in range(3):
        core[:, ax] = np.interp(t_r, t_in, seg[:, ax])
    head = _ramp_to_zero(core[0], raster, slew_max, leading=True)
    tail = _ramp_to_zero(core[-1], raster, slew_max, leading=False)
    return np.concatenate([head, core, tail], axis=0)


def design_to_pulseq(design, *, scanner='siemens_prisma', filename=None,
                     flip90=90.0, flip180=180.0, grad_raster_time=10e-6, system=None):
    """Build a scanner-runnable Pulseq spin echo from a :class:`WaveformDesign`.

    Returns the pypulseq ``Sequence`` (written to ``filename`` if given).  ``scanner``
    keys ``dmipy_sim ... PULSEQ_SYSTEMS`` (real Gmax/slew); the design's gradient is
    un-folded to the physical waveform, resampled to ``grad_raster_time``, and split
    around the 180 placed in the reserved TE/2 gap.
    """
    pp = _require_pypulseq()
    sys = system or make_system(scanner, grad_raster_time=grad_raster_time,
                                rf_raster_time=grad_raster_time,
                                block_duration_raster=grad_raster_time,
                                adc_raster_time=grad_raster_time, **_DEAD)
    gamma_hz = float(getattr(sys, 'gamma', GAMMA_HZ))

    eff = np.asarray(design.effective_G(), float)         # (n_t,3) T/m, effective
    dt = float(design.dt)
    n_t = eff.shape[0]
    echo = int(design.echo_idx)
    phys = eff.copy()
    phys[echo:] *= -1.0                                   # un-fold -> physical gradient
    gmax_w = float(np.linalg.norm(phys, axis=1).max()) or 1.0
    gnorm = np.linalg.norm(phys, axis=1)
    # the optimizer zeros the SLEW (not g) in the off-regions, so g is HELD there and
    # feasibility only guarantees |g| <= ~2% Gmax in the RF window -- detect the windows
    # as LOW-gradient (<=5% Gmax) runs, not exact zeros.
    low = gnorm <= 0.05 * gmax_w
    on = ~low
    if not on.any():
        raise ValueError("design has no gradient to export")
    enc = np.where(on)[0]
    i_lead, i_tail = int(enc[0]), int(enc[-1])            # first/last encoding samples
    # the 180 gap = the low-gradient run CONTAINING the echo (expand out from echo_idx)
    if not low[echo]:
        raise ValueError("echo_idx is not in a low-gradient window -- cannot place the 180")
    a = echo
    while a > 0 and low[a - 1]:
        a -= 1
    b = echo
    while b < n_t - 1 and low[b + 1]:
        b += 1                                            # gap = samples [a, b]

    # round all manually-built block durations to the raster (the design grid dt =
    # TE/(n_t-1) is NOT a raster multiple, which otherwise trips Pulseq's raster checks)
    rt = float(grad_raster_time)
    def _ras(x, lo=0.0):
        return max(lo, round(max(x, lo) / rt)) * rt
    rf_min = _ras(sys.rf_dead_time + sys.rf_ringdown_time + 2e-4)
    lead_dur = _ras(i_lead * dt, rf_min)                  # 90 in the lead-in
    gap_dur = _ras((b - a + 1) * dt, rf_min)              # the reserved 180 window
    tail_dur = _ras((n_t - 1 - i_tail) * dt, rt)          # ADC/readout tail

    slew_max = float(PULSEQ_SYSTEMS.get(scanner, {}).get('max_slew', 200.0))   # T/m/s
    pre = _resample_lobe(phys[i_lead:a], dt, rt, slew_max)
    post = _resample_lobe(phys[b + 1:i_tail + 1], dt, rt, slew_max)

    seq = pp.Sequence(system=sys)
    seq.add_block(pp.make_block_pulse(flip_angle=np.deg2rad(flip90), duration=lead_dur,
                                      use='excitation', system=sys))
    if pre is not None:
        seq.add_block(*[pp.make_arbitrary_grad(channel=ch, waveform=pre[:, ci] * gamma_hz,
                                               system=sys)
                        for ci, ch in enumerate('xyz') if np.any(pre[:, ci])])
    seq.add_block(pp.make_block_pulse(flip_angle=np.deg2rad(flip180), duration=gap_dur,
                                      use='refocusing', system=sys))
    if post is not None:
        seq.add_block(*[pp.make_arbitrary_grad(channel=ch, waveform=post[:, ci] * gamma_hz,
                                               system=sys)
                        for ci, ch in enumerate('xyz') if np.any(post[:, ci])])
    # readout: ADC with dwell on the raster (echo at the end = TE)
    n_adc = max(1, int(round(tail_dur / rt)))
    seq.add_block(pp.make_adc(num_samples=n_adc, dwell=rt, system=sys))

    if filename:
        seq.write(filename)
    return seq


def seq_btensor(seq, *, gamma_hz=GAMMA_HZ, n=8000):
    """b-tensor (s/m^2, 3x3) recomputed from an assembled Pulseq ``Sequence`` by
    integrating its realized gradient waveforms with the 180 sign flip folded in."""
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


def pulseq_delivery_report(design, seq, *, scanner='siemens_prisma'):
    """Run the offline acceptance checks: timing, realized peak Gmax/slew vs the system
    limits, and the b-tensor round-trip (assembled .seq vs design)."""
    from dmipy_sim.sequences.pulseq import PULSEQ_SYSTEMS
    pp = _require_pypulseq()
    ok, err = seq.check_timing()
    wave, *_ = seq.waveforms_and_times()
    gamma_hz = GAMMA_HZ
    # realized peak amplitude and slew on the fine raster (mT/m, T/m/s)
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
    lim = PULSEQ_SYSTEMS.get(scanner, {})
    return dict(
        timing_ok=bool(ok), timing_error=err,
        b_design=float(design.b_value), b_seq=b_seq,
        b_rel_err=abs(b_seq - design.b_value) / (design.b_value + 1e-30),
        max_grad_mT=gmax * 1e3, limit_grad_mT=lim.get('max_grad'),
        max_slew=smax, limit_slew=lim.get('max_slew'),
        grad_ok=(gmax * 1e3 <= lim.get('max_grad', 1e9) * 1.01),
        slew_ok=(smax <= lim.get('max_slew', 1e12) * 1.02),
    )
