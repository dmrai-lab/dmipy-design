"""Overnight deliverability battery: for every pulse type x scanner, design a PNS-aware
waveform (constraint #9), export it to a runnable Pulseq spin echo, and run the full
offline acceptance battery -- everything a scanner checks at load time except thermal/GIRF:

  * check_timing (raster / dead-time / contiguity),
  * Gmax / slew within the system box,
  * b-tensor round-trip (design vs assembled .seq),
  * pypulseq SAFE PNS (the authoritative on-export check, IEC 60601-2-33).

Compares a PNS-aware design (pns=True) against a pns=False baseline so the PNS the
constraint removes is visible.  Writes the .seq files + a markdown report incrementally
(so partial results survive an interrupt).  No scanner required.

The in-design SAFE (coarse design grid) reads ~10-15 pp BELOW pypulseq's fine-raster
SAFE, so we design with a margin (pns_target=65 -> ~80% on export) and treat the export
SAFE as authoritative.
"""
import os, time, traceback, warnings; warnings.filterwarnings('ignore')
import numpy as np
from dmipy_design.optimizers import design_waveform, SequenceTiming
from dmipy_design.pulseq_export import design_to_pulseq, pulseq_delivery_report, pulseq_pns_report

OUT = os.path.join(os.path.dirname(__file__), 'deliverability_output')
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, 'deliverability_report.md')

BUDGET = dict(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)   # asymmetric, real
TE = 0.060
SCANNERS = {'prisma': dict(G_max=0.08, name='Prisma', mT=80),
            'connectom': dict(G_max=0.30, name='Connectom', mT=300)}
PULSES = [('LTE', 1.0, None), ('PTE', -0.5, None), ('STE', 0.0, None), ('OGSE', 1.0, 80.0)]
# the SAFE-PNS convolutions make pns=True designs compile-heavy (~7 min at these
# settings), so keep restarts/n_t moderate so the 8 PNS designs finish overnight (~1.5-2 h).
DKW = dict(slew_rate_max=200.0, n_t=200, n_restarts=32, n_outer=14,
           null_M1=False, null_M2=False, maxwell=False, seed=0)

rows = []
def write_report():
    with open(REPORT, 'w') as f:
        f.write("# Deliverability battery (offline acceptance: timing + box + b round-trip + SAFE PNS)\n\n")
        f.write("budget t_excite=3 t_refocus=6 t_readout_pre_echo=14 ms; TE=60 ms; n_restarts=48.\n")
        f.write("PNS = pypulseq SAFE on the exported .seq (100%=limit, ~80%=normal mode). "
                "pns_target=65 in-design (margin for the coarse-grid underestimate).\n\n")
        f.write("| pulse | scanner | mode | b (s/mm2) | timing | maxG/lim | slew/lim | "
                "b_rel_err | SAFE PNS (export) | .seq |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %s | %s | %s | %.0f | %s | %.0f/%.0f | %.0f/%.0f | %.1f%% | %.0f%% %s | %s |\n" % (
                r['pulse'], r['scanner'], r['mode'], r['b'], 'OK' if r['timing'] else 'FAIL',
                r['maxG'], r['limG'], r['slew'], r['limS'], r['b_rel']*100,
                r['pns'], 'OK' if r['pns'] <= 80 else ('1st' if r['pns'] <= 100 else 'OVER'),
                os.path.basename(r['seq'])))

t0 = time.time()
for sk, sc in SCANNERS.items():
    for tag, bd, sf in PULSES:
        for mode, pns in (('baseline', False), ('pns-aware', True)):
            label = '%s/%s/%s' % (tag, sk, mode)
            scanner_key = 'siemens_' + sk                    # 'siemens_prisma' / 'siemens_connectom'
            try:
                print('[%5.0fs] designing %s ...' % (time.time()-t0, label), flush=True)
                d = design_waveform(bd, G_max=sc['G_max'], TE=TE, spectral_freq=sf,
                                    timing=SequenceTiming(**BUDGET), pns=pns, pns_target=65.0, **DKW)
                seq = design_to_pulseq(d, scanner=scanner_key)
                fn = os.path.join(OUT, 'seq_%s_%s_%s.seq' % (tag, sk, mode))
                seq.write(fn)
                rep = pulseq_delivery_report(d, seq, scanner=scanner_key)
                pr = pulseq_pns_report(seq)
                rows.append(dict(pulse=tag, scanner=sc['name'], mode=mode, b=d.b_value/1e6,
                                 timing=rep['timing_ok'], maxG=rep['max_grad_mT'], limG=sc['mT'],
                                 slew=rep['max_slew'], limS=200, b_rel=rep['b_rel_err'],
                                 pns=pr['pns_max_pct'], seq=fn))
                print('   -> b=%.0f feasible=%s timing=%s b_rel=%.1f%% SAFE-PNS=%.0f%%' % (
                    d.b_value/1e6, d.feasible, rep['timing_ok'], rep['b_rel_err']*100, pr['pns_max_pct']), flush=True)
            except Exception as e:
                print('   FAILED %s: %s' % (label, e), flush=True)
                traceback.print_exc()
            write_report()                                   # incremental: survive interrupts

print('\n[%5.0fs] DONE. report: %s' % (time.time()-t0, REPORT), flush=True)
