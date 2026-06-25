"""Cost of the robustness constraints (M1 / M2 / Maxwell) for LTE diffusion encoding.

Regenerates the cost table in ``waveform_designer``'s module docstring AND a figure
showing how the waveform gets richer as constraints are added — the extra
sign-alternating sub-lobes ARE the compensation machinery, bought at the cost of b.

For PURE max-b LTE the optimum is the bang-bang ~PGSE shape (2 lobes straddling the
180); the optimizer just rediscovers Stejskal-Tanner, so there is little to "design".
The robustness constraints are what force non-trivial shape — and that shape is a
continuous b-vs-compensation trade, not a hidden better encoder.

Run:
    /home/rutger/dmipy-core/.venv/bin/python benchmarks/constraint_cost_waveforms.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmipy_design.optimizers import design_waveform, SequenceTiming

G_MAX, SLEW, TE, N_T = 0.08, 200.0, 0.060, 200
TIMING = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)
BASE = dict(G_max=G_MAX, slew_rate_max=SLEW, TE=TE, n_t=N_T, timing=TIMING,
            n_restarts=48, n_outer=16, moment_tol=2e-2)

CONFIGS = [
    ("none (PGSE bang-bang)", dict(null_M1=False, null_M2=False, maxwell=False)),
    ("+ M1 (velocity)",       dict(null_M1=True,  null_M2=False, maxwell=False)),
    ("+ M2 (acceleration)",   dict(null_M1=True,  null_M2=True,  maxwell=False)),
    ("+ Maxwell (full)",      dict(null_M1=True,  null_M2=True,  maxwell=True)),
]


def lobes(g_axis, gmax):
    """Count sign-alternating lobes (|g| > 5% Gmax) on one axis."""
    s = np.sign(np.where(np.abs(g_axis) > 0.05 * gmax, g_axis, 0.0))
    s = s[s != 0]
    return int(1 + np.sum(np.abs(np.diff(s)) > 0)) if len(s) else 0


def main():
    designs, b0 = [], None
    print(f"{'constraints':<24}{'b (s/mm2)':>11}{'factor':>9}   lobes pre/post   M1     M2     mx")
    print("-" * 84)
    for name, cfg in CONFIGS:
        d = design_waveform(1.0, **BASE, **cfg)
        b = d.b_value / 1e6
        if b0 is None:
            b0 = b
        e = d.echo_idx
        npre, npost = lobes(d.G[:e, 0], G_MAX), lobes(d.G[e:, 0], G_MAX)
        print(f"{name:<24}{b:>11.0f}{b0/max(b,1):>8.1f}×     {npre:>2d} / {npost:<2d}        "
              f"{d.m1_index:.3f}  {d.m2_index:.3f}  {d.maxwell_index:.3f}")
        designs.append((name, d, npre, npost))

    # figure: g(t) per axis stacked, 180 marked, lobe counts annotated
    t = np.arange(N_T) * (TE / (N_T - 1)) * 1e3  # ms
    fig, axes = plt.subplots(len(designs), 1, figsize=(8, 9), sharex=True)
    for ax, (name, d, npre, npost) in zip(axes, designs):
        t180 = d.echo_idx * (TE / (N_T - 1)) * 1e3
        for k, lab in enumerate("xyz"):
            ax.plot(t, d.G[:, k] * 1e3, lw=1.2, label=f"g_{lab}")
        ax.axvline(t180, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_ylabel("g (mT/m)")
        ax.set_title(f"{name}   b={d.b_value/1e6:.0f} s/mm²   "
                     f"lobes {npre}/{npost} (pre/post-180)   "
                     f"M1={d.m1_index:.3f} M2={d.m2_index:.3f} mx={d.maxwell_index:.3f}",
                     fontsize=9, loc="left")
        ax.set_ylim(-G_MAX * 1e3 * 1.1, G_MAX * 1e3 * 1.1)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("t (ms)   — dashed line = 180° refocusing pulse")
    fig.suptitle("Robustness constraints force compensation structure (b-vs-comp trade)\n"
                 "3T Prisma, LTE, TE=60 ms — the extra lobes ARE the compensation",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "benchmarks/constraint_cost_waveforms.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
