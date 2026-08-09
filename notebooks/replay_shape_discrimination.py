"""Design a gradient waveform that best discriminates a cylinder from a sphere — by replaying the
Substrate Commons canonical replay-pack dataset.

This is the substrate-informed acquisition-design pivot: instead of re-simulating (or replaying raw
trajectory arrays) for each candidate waveform, we replay pre-computed `.rpk` packs. The signal is a
cheap, differentiable matmul on the pack's DCT-compressed trajectory (`dmipy_sim.replay`), so the
waveform is optimized by gradient descent *through* the stored Monte-Carlo substrate.

Run:
    SUBSTRATE_COMMONS_DATA=/path/to/canonical  python notebooks/replay_shape_discrimination.py
where the dataset holds the canonical packs, e.g.
    <root>/canonical/D0-2.00e-9/cylinder/d005.00um.rpk
    <root>/canonical/D0-2.00e-9/sphere/d005.00um.rpk
(Install the replay engine with `pip install dmipy-design[sim]`.)
"""
import os
import numpy as np

from dmipy_sim.replay import read_rpk, compile_scheme, replay_signal
from dmipy_sim.constants import GAMMA
from dmipy_design.replay_design import design_discriminating_waveform


def main(d_um=5.0, D0=2.0e-9, G_max=0.30):
    root = os.environ.get("SUBSTRATE_COMMONS_DATA", ".")
    base = os.path.join(root, "canonical", f"D0-{D0*1e9:.2f}e-9")
    cyl = read_rpk(os.path.join(base, "cylinder", f"d{d_um:05.2f}um.rpk"))
    sph = read_rpk(os.path.join(base, "sphere", f"d{d_um:05.2f}um.rpk"))

    # Baseline: the best plain PGSE over a b-sweep (gradient perpendicular to the cylinder axis = x).
    dt, n_t = cyl.dt, cyl.n_t
    delta, Delta = 10e-3, min((n_t - 2) * dt, 40e-3)
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    def pgse(b):
        g = np.zeros((1, n_t, 3)); nd = max(1, int(round(delta / dt))); ng = int(round(Delta / dt))
        g[0, :nd, 0] = np.sqrt(b / bu); g[0, ng:ng + nd, 0] = -np.sqrt(b / bu)
        return g
    base_c = max(abs(replay_signal(cyl, compile_scheme(pgse(b), dt, cyl.K))[0]
                     - replay_signal(sph, compile_scheme(pgse(b), dt, sph.K))[0])
                 for b in np.linspace(0.5e9, 8e9, 12))

    # Optimized: freeform waveform maximizing |E_cyl - E_sph|, differentiated through the packs.
    res = design_discriminating_waveform(cyl, sph, direction=(1., 0, 0), G_max=G_max,
                                         n_restarts=6, maxiter=400)

    print(f"cylinder vs sphere @ d={d_um} um")
    print(f"  best PGSE contrast |E_cyl - E_sph| = {base_c:.4f}")
    print(f"  optimized waveform contrast        = {res.contrast:.4f}  "
          f"(E_cyl={res.E_A:.4f}, E_sph={res.E_B:.4f}, b={res.b_value/1e6:.0f} s/mm^2)")
    print(f"  improvement over best PGSE: {res.contrast/base_c:.2f}x")
    return res


if __name__ == "__main__":
    main()
