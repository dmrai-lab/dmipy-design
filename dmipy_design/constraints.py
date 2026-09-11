"""Scan-time constraints. A scanner's gradient and RF limits are data with a citation and live in dmipy-sim's
catalogue (:class:`dmipy_sim.acquisition.scanners.ScannerLimits`); every designer here takes one as ``limits``.
"""

from dataclasses import dataclass


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
    _null_M1: bool = False
    _null_M2: bool = False

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

    def project(self, x, n_iter=8, per_axis=True):
        """Retract ``x`` into the same feasible set, for first-order solvers.

        The set is exposed two ways on purpose. `constraints`/`bounds` hand it to SQP, which
        holds it EXACTLY and is what the NOW designer wants for its ~420-variable problems.
        This hands the same set to a projected first-order method, which is what certification
        needs: a certificate is taken over the whole walk, so n_t is 601-1001 rather than 140,
        SQP's QP subproblem is O(n^3) in the 1800-3000 variables that implies, and it stops
        being usable well before that. Same set, two interfaces -- which is the point, since a
        certificate over a different set than the designer can reach would be unsound.

        Moment nulling is affine, the slew limit a band and the amplitude limit a box, so all
        three are convex and alternating projection converges to the intersection. The slew step
        is a forward rate-limit rather than the exact Euclidean projection onto the band, so this
        is a retraction: every returned waveform is feasible, but it is not the nearest feasible
        point. That is sound for a certificate (the sup is taken over feasible waveforms only)
        and is why the ascent, not the projection, is asked to push back to the limits.
        """
        import numpy as np
        g = self.gradient(x) if np.ndim(x) == 1 else np.array(x, float)
        t = np.arange(self.n_t) * self.dt
        basis = [self.sign[:, 0]]
        for order, on in ((1, self._null_M1), (2, self._null_M2)):
            if on:
                basis.append((t ** order) * self.sign[:, 0])
        Q, _ = np.linalg.qr(np.stack(basis, 1))
        for _ in range(n_iter):
            g = g - Q @ (Q.T @ g)
            for _ in range(3):
                d = np.clip(np.diff(g, axis=0), -self.slew_rate_max * self.dt,
                            self.slew_rate_max * self.dt)
                g = np.concatenate([g[:1], g[:1] + np.cumsum(d, axis=0)], axis=0)
            if per_axis:
                g = np.clip(g, -self.G_max, self.G_max)
            else:
                n = np.linalg.norm(g, axis=1, keepdims=True)
                g = g * np.minimum(1.0, self.G_max / np.maximum(n, 1e-30))
        g = g - Q @ (Q.T @ g)
        mask = np.zeros(self.n_t, bool); mask[self.free] = True
        g[~mask] = 0.0                                  # nothing outside the encoding window
        return self.flatten(g)


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
    from dmipy_sim.constants import GAMMA

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
                           bounds=[(-G_max, G_max)] * (na * nf),
                           _null_M1=bool(null_M1), _null_M2=bool(null_M2))
