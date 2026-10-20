"""Pin BLAS/OpenMP to a single thread before NumPy is imported.

The NOW designer is many small SciPy SLSQP solves (tiny matmuls in a tight inner loop).
On a high-core-count machine, NumPy/OpenBLAS otherwise spins up one thread per core for
each tiny operation, and the thread-contention overhead dwarfs the actual work (a ~0.04 s
solve balloons to tens of seconds). Single-threaded BLAS is dramatically faster here.

conftest is imported before the test modules (and thus before NumPy), so these take effect.
"""
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")


# ----------------------------------------------------------------------------- replay-pack fixtures
# The packs the replay-design, LUT, co-design and certificate tests read: walked by dmipy-sim's producer at
# small n and packed by its bank, exactly as a published pack is (never a hand-rolled walk converted to a pack).
import pytest

SLAB_N_WALKERS, SLAB_K, SLAB_T, SLAB_D0 = 600, 40, 0.1, 2e-9


def _slab_pack(L, seed):
    """A reflecting slab of thickness ``L`` (dmipy-sim's :class:`~dmipy_sim.geometry.Box1D`, free along y, z) walked
    over one 100 ms window at ``SLAB_K`` bands and packed, the size-discrimination substrate."""
    import dmipy_sim as ds
    from dmipy_sim.geometry import Box1D
    from dmipy_sim.replay.bank import build_replay_pack
    walk = ds.simulate_trajectories(n_walkers=SLAB_N_WALKERS, diffusivity=SLAB_D0, geometry=Box1D(L), T_max=SLAB_T,
                                    seed=seed, K=SLAB_K, require_gpu=False)
    return build_replay_pack(walk, id=f"test/slab-{seed}", license="CC-BY-4.0", citation="test", K=SLAB_K)


@pytest.fixture(scope="session")
def slab_packs():
    """``{thickness: pack}`` for slabs of 3, 4, 5 and 8 um (seeds 0..3)."""
    pytest.importorskip("dmipy_sim")
    return {L: _slab_pack(L, s) for s, L in enumerate((3e-6, 4e-6, 5e-6, 8e-6))}
