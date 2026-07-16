"""
Hardware and time constraints for acquisition design.
"""

from dataclasses import dataclass


@dataclass
class HardwareConstraints:
    """MRI scanner hardware constraints.

    Parameters
    ----------
    G_max : float
        Maximum gradient amplitude in T/m.  Typical values:
        standard 3T: 0.04–0.08 T/m; Connectom 3T: 0.30 T/m.
    slew_rate_max : float
        Maximum slew rate in T/m/s.
    TE_min : float
        Minimum echo time in seconds (hardware/SAR constraint).
    TE_max : float
        Maximum echo time in seconds.
    """
    G_max: float = 0.08          # T/m  (standard 3T Prisma)
    slew_rate_max: float = 200.0  # T/m/s
    TE_min: float = 0.060         # s
    TE_max: float = 0.200         # s

    def gradient_for_b(self, b: float, delta: float, Delta: float) -> float:
        """Compute gradient amplitude required for a given b-value and timing."""
        import numpy as np
        GAMMA = 2.675e8  # rad/s/T
        denom = GAMMA ** 2 * delta ** 2 * (Delta - delta / 3.0)
        if denom <= 0:
            return float("inf")
        return float(np.sqrt(b / denom))

    def is_feasible(self, b: float, delta: float, Delta: float) -> bool:
        """Return True if the (b, delta, Delta) combination is hardware-feasible."""
        G = self.gradient_for_b(b, delta, Delta)
        return G <= self.G_max and delta < Delta


@dataclass
class TimeConstraints:
    """Scan time constraints.

    Parameters
    ----------
    total_scan_time_s : float
        Maximum total scan time in seconds.
    tr : float
        Repetition time in seconds.  Total measurements = total_scan_time / TR.
    """
    total_scan_time_s: float = 600.0  # 10 minutes
    tr: float = 5.0                   # seconds

    @property
    def max_measurements(self) -> int:
        return int(self.total_scan_time_s / self.tr)
