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


# --------------------------------------------------------------- deliverable waveform set
@dataclass
class WaveformProblem:
    """The set ``G_scanner`` of *deliverable* gradient waveforms, as an SLSQP problem.

    This is the feasible set shared by everything in this package that optimises a gradient
    waveform: NOW maximises b over it, and the replay certificate takes a sup of the replay
    error over it. They must be the SAME set -- a certificate taken over a smaller set than the
    designer can reach is unsound, because a user could design a waveform the pack was never
    certified for and get a silently wrong number back.

    The constraints are held EXACTLY by SQP rather than by projection or penalty, so an
    optimum rides ``|G| = G_max`` and ``|dG/dt| = slew_max`` instead of stopping short of them.
    Variables are axis-major over the free (encoding-window) samples, ``x[k*nf + i]``.
    """

    n_t: int
    n_axes: int
    dt: float
    free: "np.ndarray"          # indices of samples the optimiser may move
    sign: "np.ndarray"          # (n_t, 1) spin-echo sign: +1 before the 180, -1 after
    G_max: float
    slew_rate_max: float
    constraints: list
    bounds: list

    @property
    def n_free(self):
        return len(self.free)

    @property
    def n_var(self):
        return self.n_axes * self.n_free

    def gradient(self, x):
        """Waveform ``(n_t, n_axes)`` from the flat variable vector (zero outside the window)."""
        import numpy as np
        g = np.zeros((self.n_t, self.n_axes))
        g[self.free, :] = np.asarray(x).reshape(self.n_axes, self.n_free).T
        return g

    def effective(self, x):
        """The waveform as the SPINS see it -- sign-flipped across the refocusing pulse."""
        return self.sign * self.gradient(x)

    def flatten(self, dJdg):
        """Fold a per-sample gradient ``(n_t, n_axes)`` back to the flat variable layout."""
        return dJdg[self.free, :].T.reshape(-1)


def waveform_problem(*, n_t, n_axes, dt, echo, encoding_mask=None, G_max=0.08,
                     slew_rate_max=200.0, null_M1=True, null_M2=True):
    """Build the shared :class:`WaveformProblem` (amplitude box, slew band, refocusing, moments).

    ``encoding_mask`` marks samples where the gradient may be non-zero; it is eroded by one
    ramp length so the waveform ramps from zero at each window edge rather than stepping.
    Refocusing (``q(TE)=0``) is always imposed -- a diffusion waveform that does not refocus is
    not a diffusion waveform.
    """
    import numpy as np
    import scipy.sparse as sp

    GAMMA = 267.513e6
    enc = (np.ones(n_t, bool) if encoding_mask is None else np.asarray(encoding_mask, bool))
    nr = max(1, int(np.ceil(G_max / (slew_rate_max * dt))))
    er = enc.copy()
    for j in range(1, nr + 1):
        er &= np.roll(enc, j) & np.roll(enc, -j)
    free = np.where(er)[0]
    nf, na = len(free), int(n_axes)
    if nf == 0:
        raise ValueError("the encoding window is shorter than one gradient ramp; nothing is free")
    s = np.where(np.arange(n_t) < int(echo), 1.0, -1.0)[:, None]
    tt = (np.arange(n_t) * dt)[:, None]

    Sel = sp.csr_matrix((np.ones(nf), (free, np.arange(nf))), shape=(n_t, nf))
    D = sp.diags([-np.ones(n_t), np.ones(n_t - 1)], [0, 1], shape=(n_t - 1, n_t))
    A_slew = sp.block_diag([(D @ Sel) / dt] * na).toarray()
    A_ref = sp.block_diag([(GAMMA * dt * s[free, 0]).reshape(1, -1)] * na).toarray()
    cons = [{"type": "ineq", "fun": lambda x: slew_rate_max - A_slew @ x, "jac": lambda x: -A_slew},
            {"type": "ineq", "fun": lambda x: slew_rate_max + A_slew @ x, "jac": lambda x: A_slew},
            {"type": "eq", "fun": (lambda A: lambda x: A @ x)(A_ref),
             "jac": (lambda A: lambda x: A)(A_ref)}]
    for on, w in ((null_M1, tt), (null_M2, tt ** 2)):
        if on:
            A = sp.block_diag([((w * s)[free, 0] * dt).reshape(1, -1)] * na).toarray()
            cons.append({"type": "eq", "fun": (lambda A: lambda x: A @ x)(A),
                         "jac": (lambda A: lambda x: A)(A)})
    return WaveformProblem(n_t=n_t, n_axes=na, dt=dt, free=free, sign=s, G_max=G_max,
                           slew_rate_max=slew_rate_max, constraints=cons,
                           bounds=[(-G_max, G_max)] * (na * nf))
