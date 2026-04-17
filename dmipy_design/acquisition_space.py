"""
AcquisitionSpace: defines the design space for optimal experiment design.
"""

from dataclasses import dataclass, field


@dataclass
class AcquisitionSpace:
    """Parameterisation of the acquisition design space.

    Free parameters (with bounds) are the variables dmipy-design optimises.
    Fixed parameters are held constant.  Discrete choices are searched
    exhaustively or via Bayesian optimisation.

    Parameters
    ----------
    n_measurements : int
        Total number of DWI measurements in the designed protocol.
    bvalue_range : (float, float)
        Min and max b-value in s/m².  E.g. (0, 10000e6).
    delta_range : (float, float) or None
        Gradient pulse duration range in seconds.  None = fixed by Delta.
    Delta_range : (float, float) or None
        Gradient separation range in seconds.
    TE_range : (float, float) or None
        Echo time range in seconds.  None = not optimised.
    b_delta_values : list of float or None
        Discrete B-tensor shape values to include in the design space.
        b_delta = 1 → LTE (linear), 0 → STE (spherical), -0.5 → PTE (planar).
        None = PGSE only.
    waveform_types : list of str
        Waveform types to include.  Subset of ['pgse', 'ogse', 'pgste', 'ste', 'pte'].
    n_shells : int or None
        If set, constrain the optimised protocol to exactly this many shells.
    fixed_bvecs : array (N, 3) or None
        If set, gradient directions are fixed; only b-values/timings are optimised.
    """
    n_measurements: int
    bvalue_range: tuple[float, float] = (0.0, 10000e6)
    delta_range: tuple[float, float] | None = None
    Delta_range: tuple[float, float] | None = None
    TE_range: tuple[float, float] | None = None
    b_delta_values: list[float] | None = None
    waveform_types: list[str] = field(default_factory=lambda: ["pgse"])
    n_shells: int | None = None
    fixed_bvecs: object = None   # np.ndarray (N, 3) or None
