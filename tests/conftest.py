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
