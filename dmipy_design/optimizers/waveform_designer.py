"""
Tensor-valued diffusion gradient-waveform designer (NOW-style, JAX/GPU).

Generates a hardware-realizable physical gradient waveform ``g(t)`` with a
finite-180 spin echo built in, achieving a target b-tensor shape
(``b_delta``: LTE = 1, STE = 0, PTE = -0.5, or any value in [-0.5, 1]) while
maximizing the b-value under Prisma-class hardware (G_max, slew, TE).  The
finite 180 is intrinsic: the spin-echo sign flip enters ``q(t)``, so static-spin
refocusing ``q(TE) = 0`` is part of the optimization (an effective-only waveform
carrying no 180 cannot refocus a static field — susceptibility/off-resonance).

Robustness constraints (ON by default; turn off only with justification)
------------------------------------------------------------------------
Shape, refocusing and the hardware box are always enforced — that is the minimum
for a valid waveform.  On top of that, three constraints suppress real
acquisition confounds.  They default ON because leaving a *needed* one off
BIASES the measurement, whereas turning an unneeded one on only costs b-value
(SNR).  So disable one only when its confound is demonstrably absent:

  * ``null_M1`` — velocity compensation, ``∫ t·g_eff dt = 0``.  Off ⇒ the signal
    is sensitive to bulk velocity (cardiac pulsation, CSF/flow, perfusion):
    velocity-dependent dephasing masquerades as diffusion and biases the metrics,
    worst at high b and near vessels/ventricles/cord.  Costs the most b (~4×; the
    classic flow-comp penalty).  Safe to disable only for static samples
    (ex-vivo, fixed tissue, phantoms) or low b.
  * ``null_M2`` — acceleration compensation, ``∫ t²·g_eff dt = 0``.  Second-order
    (pulsatile) motion.  Safe to disable when velocity comp alone is adequate
    (commonly so in vivo); keep on for strongly pulsatile regimes.
  * ``maxwell`` — concomitant-field (Maxwell) compensation, ``∫ s·g·gᵀ dt = 0``
    (Szczepankiewicz 2019).  Concomitant fields ∝ g²/B0 add a spatially-varying
    phase that does not refocus for time-ASYMMETRIC waveforms, biasing the
    b-tensor metrics across the FOV (worse off-isocenter, low B0, strong
    gradients).  It is automatically ~0 for time-symmetric waveforms (cheap/free
    there) and only bites for asymmetric designs (which are used to shorten TE /
    gain SNR).  Safe to disable for symmetric waveforms or near-isocenter,
    high-B0, small-FOV work.

All three indices (``m1_index``, ``m2_index``, ``maxwell_index``) are reported on
the result regardless of which flags were active, so the residual confound is
always visible — even for a constraint left off.

Physics (the metrics double as the constraint functions)
--------------------------------------------------------
Effective dephasing with the 180 sign flip folded in::

    s(t) = +1 for t < t_180,  -1 for t >= t_180
    g_eff = s * g
    q(t) = gamma * cumsum(g_eff) * dt                  (rad/m)
    B    = integral q q^T dt                           (s/m^2, the b-tensor)
    b    = trace(B)
    M_k  = integral t^k g_eff dt                       (gradient moments)
    Maxw = integral s g g^T dt                         (concomitant matrix)

Refocusing (echo) is the zeroth moment condition ``q(TE) = 0`` (= M0 = 0).

Parameterization (both hardware limits structural)
--------------------------------------------------
slew = S_max·tanh(|raw|)/|raw|·raw·rf_off  ->  |dg/dt| ~<= S_max (radial)
g_raw = dt·cumsum(slew)                     ->  g(0) = 0
g    = G_max·tanh(|g_raw|/G_max)/|g_raw|·g_raw  ->  |g| <= G_max
so b can be driven to the amplitude wall without violating the box.  The
amplitude squash perturbs the slew slightly, so a residual slew constraint is
kept.  Equality constraints (refocus, shape, g(TE)=0, RF-window, residual slew,
and any active M1/M2/Maxwell) are driven to zero by an augmented Lagrangian.

Solver
------
JAX augmented Lagrangian: inner jaxopt L-BFGS minimizes ``-b + λ·c + (μ/2)|c|²``
(vmapped over restarts on GPU), outer loop updates ``λ += μ c`` and grows ``μ``.
Decoupling feasibility (multipliers) from the objective avoids penalty-weight
balancing that otherwise collapses b.  Best feasible restart (max b) wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False

GAMMA = 267.513e6  # rad/s/T — matches dmipy_sim.constants.GAMMA

# Fixed constraint order; the AL uses a flag-selected subset, the report shows all.
CONSTRAINT_NAMES = ('refocus', 'shape', 'g(TE)=0', 'RF-window', 'slew',
                    'M1', 'M2', 'maxwell', 'spectral')
_BASE_CONSTRAINTS = (0, 1, 2, 3, 4)   # always active (validity)


def encoding_spectrum(G, dt, echo_idx):
    """Encoding power spectrum |q̃(f)|² of a physical gradient + its summary.

    The rigorous spectral-content quantity (Stepišnik): the diffusion signal is
    ``ln S ≈ −∫ D(ω)·|q̃(ω)|² dω``, so a waveform is characterized — for ANY shape,
    pure or broadband — by this spectrum, not by a nominal "frequency".  Returns
    ``(freqs_Hz, power, centroid_Hz, bandwidth_Hz, rms_Hz)`` (one-sided), letting
    you quantify and propagate the actual spectral content (and its imprecision).
    """
    import numpy as _np
    G = _np.asarray(G, dtype=_np.float64)
    s = _np.where(_np.arange(G.shape[0]) < echo_idx, 1.0, -1.0)[:, None]
    q = GAMMA * _np.cumsum(s * G, axis=0) * dt                      # (n_t,3) rad/m
    P = _np.sum(_np.abs(_np.fft.rfft(q, axis=0)) ** 2, axis=1)      # (nf,) power
    f = _np.fft.rfftfreq(G.shape[0], dt)
    Psum = P.sum() + 1e-30
    centroid = float((f * P).sum() / Psum)
    bandwidth = float(_np.sqrt(((f - centroid) ** 2 * P).sum() / Psum))
    rms = float(_np.sqrt((GAMMA ** 2 * _np.sum(G ** 2)) / (_np.sum(q ** 2) + 1e-30))
                / (2 * _np.pi))
    return f, P, centroid, bandwidth, rms


@dataclass
class SequenceTiming:
    """Physical diffusion spin-echo timing budget that pins the encoding windows.

    The 180 sits at TE/2 (spin-echo refocus condition).  Diffusion encoding is OFF
    during the excitation lead-in, across the 180 (+crushers), and during the
    readout tail; the two remaining windows (pre-/post-180) are generally UNEQUAL
    because ``t_prep+t_excite ≠ t_readout_pre_echo``.  So any pre/post asymmetry of
    the optimized waveform is a *consequence* of this budget, never a free knob::

        [prep+excite] [== pre-180 encode ==] [180] [== post-180 encode ==] [readout→echo]
        0             t_lead                 TE/2∓t_refocus/2          TE−t_ro_pre   TE

    All times in seconds.  Pass to ``design_waveform(..., timing=...)`` and the
    encoding-window masks + 180 position are derived (overriding echo_frac /
    rf_duration).  Build it from a real sequence with ``from_pulseq``, or from a
    readout description with ``from_readout``.
    """
    t_excite: float                  # 90 RF duration; encoding starts after it
    t_refocus: float                 # 180 RF (+crusher) duration; off across it at TE/2
    t_readout_pre_echo: float        # readout start → echo; post-180 encode ends by TE−this
    t_prep: float = 0.0              # optional fat-sat/prep before encoding
    TE: float | None = None          # native echo time (e.g. read from a .seq); masks() default
    symmetric: bool = False          # VANILLA mode: mirror the pre/post-180 windows about the
    #                                  echo (equal durations), dead-timing the surplus of the
    #                                  longer window.  See masks() — this is the conventional
    #                                  "symmetric" waveform you reach by REFUSING the asymmetry.

    @property
    def t_lead(self) -> float:
        """Dead time from t=0 (excitation centre) until encoding may begin."""
        return self.t_prep + self.t_excite

    def min_TE(self) -> float:
        """Smallest TE for which both the pre- and post-180 encoding windows exist."""
        return max(2.0 * (self.t_lead + self.t_refocus / 2.0),
                   2.0 * (self.t_readout_pre_echo + self.t_refocus / 2.0))

    def masks(self, TE=None, n_t=256):
        """Return ``(slew_off_mask (n_t,1) float, echo_idx int)`` for a given TE.

        ``slew_off_mask`` is 1 in the two encoding windows and 0 in the off-regions
        (excitation lead-in, the 180, the readout tail), so the optimizer's gradient
        lives only where the hardware allows it.  The 180 (echo_idx) is at TE/2.

        ``symmetric`` (VANILLA mode): the inner edges are already ±t_refocus/2 from the
        echo, so a symmetric (mirror about the echo) encoding requires equal OUTER
        extents — both windows reach ``W = min(pre_dur, post_dur)`` out from the 180.
        The surplus of whichever real window was longer is forced to 0 → it becomes
        dead time the spins spend transverse (extra T2 loss).  This is the conventional
        symmetric waveform: the cost of REFUSING the budget's natural asymmetry.
        """
        TE = float(TE if TE is not None else self.TE)
        if TE < self.min_TE() - 1e-9:
            raise ValueError(
                f"TE={TE*1e3:.2f} ms is below min_TE={self.min_TE()*1e3:.2f} ms for "
                f"this timing (encoding windows would vanish).")
        dt = TE / (n_t - 1)
        t = np.arange(n_t) * dt
        echo = TE / 2.0
        on = np.ones(n_t, dtype=np.float64)
        on[t < self.t_lead] = 0.0                                   # excitation lead-in
        on[np.abs(t - echo) <= self.t_refocus / 2.0] = 0.0          # 180 (+crusher)
        on[t > TE - self.t_readout_pre_echo] = 0.0                  # readout tail
        if self.symmetric:
            pre_dur = (echo - self.t_refocus / 2.0) - self.t_lead
            post_dur = (TE - self.t_readout_pre_echo) - (echo + self.t_refocus / 2.0)
            W = max(0.0, min(pre_dur, post_dur))                    # mirror extent from 180
            on[t < echo - self.t_refocus / 2.0 - W] = 0.0           # dead-time the longer side
            on[t > echo + self.t_refocus / 2.0 + W] = 0.0
        return on[:, None], int(round(echo / dt))

    @classmethod
    def from_readout(cls, *, t_excite, t_refocus, readout_duration, partial_fourier,
                     t_prep=0.0, TE=None):
        """Build from a readout description.  The echo (k-space centre) sits
        ``(pf−0.5)/pf`` into the readout, so partial Fourier (pf<1) shortens the
        post-180 window — exactly the mechanism that makes the optimum asymmetric."""
        pf = float(partial_fourier)
        if not (0.5 <= pf <= 1.0):
            raise ValueError(f"partial_fourier must be in [0.5, 1.0]; got {pf}")
        return cls(float(t_excite), float(t_refocus),
                   float(readout_duration) * (pf - 0.5) / pf, float(t_prep), TE)

    @classmethod
    def from_pulseq(cls, src):
        """Read the timing budget (and native TE) from a Pulseq ``.seq`` via
        ``dmipy_sim.sequences.pulseq.pulseq_timing`` (first RF=90, second=180, one ADC)."""
        from dmipy_sim.sequences.pulseq import pulseq_timing
        d = pulseq_timing(src)
        return cls(t_excite=d['t_excite'], t_refocus=d['t_refocus'],
                   t_readout_pre_echo=d['t_readout_pre_echo'], TE=d['TE'])


# ===========================================================================
# Differentiable physics — these ARE the success metrics / constraint functions
# ===========================================================================

def _echo_sign(n_t: int, echo_idx: int):
    return jnp.where(jnp.arange(n_t) < echo_idx, 1.0, -1.0)[:, None]   # (n_t,1)


def effective_q(g, dt: float, echo_idx: int):
    """Effective dephasing q(t) (rad/m), with the 180 sign flip folded in."""
    s = _echo_sign(g.shape[0], echo_idx)
    return GAMMA * dt * jnp.cumsum(s * g, axis=0)                       # (n_t,3)


def b_tensor(g, dt: float, echo_idx: int):
    """b-tensor B = ∫ q qᵀ dt  (3,3), matching dmipy_sim._btensor_from_waveform.

    Explicit outer-product sum, not einsum/matmul: the small (3×3) contraction
    triggers an XLA "too small divisible part of the contracting dimension"
    failure inside ``jax.vmap`` on GPU.
    """
    q = effective_q(g, dt, echo_idx)
    return dt * jnp.sum(q[:, :, None] * q[:, None, :], axis=0)


def b_value(B):
    return jnp.trace(B)


def b_delta_of(B):
    """Normalized b-tensor anisotropy: 1 LTE, 0 STE, -0.5 PTE (frame-invariant)."""
    w = jnp.linalg.eigvalsh(B)
    b = jnp.sum(w) + 1e-30
    idx = jnp.argmax(jnp.abs(w - b / 3.0))
    lam_u = w[idx]
    return (lam_u - (b - lam_u) / 2.0) / b


def _target_eigs(b_delta, b):
    lam_par = (b / 3.0) * (1.0 + 2.0 * b_delta)
    lam_perp = (b / 3.0) * (1.0 - b_delta)
    return jnp.sort(jnp.array([lam_par, lam_perp, lam_perp]))


def _shape_penalty(B, b_delta):
    w = jnp.sort(jnp.linalg.eigvalsh(B))
    b = jnp.sum(w) + 1e-30
    return jnp.sum((w - _target_eigs(b_delta, b)) ** 2) / (b ** 2)


# ===========================================================================
# Parameterization + full constraint vector
# ===========================================================================

def _waveform_from_raw(raw, dt, s_max, g_max, slew_off_mask):
    """raw (n_t,3) -> slew- and amplitude-bounded, RF-gated physical g (n_t,3)."""
    rn = jnp.sqrt(jnp.sum(raw ** 2, axis=1, keepdims=True) + 1e-12)
    slew = s_max * (jnp.tanh(rn) / rn) * raw * slew_off_mask            # |slew|<=S_max
    g_raw = dt * jnp.cumsum(slew, axis=0)                              # g(0)=0
    gn = jnp.sqrt(jnp.sum(g_raw ** 2, axis=1, keepdims=True) + 1e-12)
    return g_max * (jnp.tanh(gn / g_max) / gn) * g_raw                 # |g|<=G_max


def _rms_frequency(g, q):
    """RMS encoding frequency f_rms (Hz) — the FFT-free spectral-content measure.

    From Stepišnik's spectral formalism: <ω²> = ∫|q'|² / ∫|q|² with q' = γ·g_eff,
    so f_rms = sqrt(γ²·Σ|g|² / Σ|q|²)/(2π).  ~0 for PGSE (low-freq), ≈ f for an
    OGSE at frequency f.  Differentiable and cheap (no FFT)."""
    w2 = (GAMMA ** 2) * jnp.sum(g ** 2) / (jnp.sum(q ** 2) + 1e-30)
    return jnp.sqrt(w2) / (2.0 * jnp.pi)


def _b_and_constraints(raw, dt, echo_idx, s_max, g_max, b_delta, rf_mask,
                       slew_off_mask, t_arr, TE, f_target=0.0):
    """Return (b, c) where c is the full (9,) vector of equality violations.

    Each entry is a normalized squared violation (0 = satisfied); the AL selects
    an active subset and the report reads them all.  ``f_target`` (Hz) drives the
    RMS encoding frequency (spectral content) when the spectral constraint is
    active; 0 leaves that entry inert.
    """
    g = _waveform_from_raw(raw, dt, s_max, g_max, slew_off_mask)
    B = b_tensor(g, dt, echo_idx)
    b = b_value(B)
    s = _echo_sign(g.shape[0], echo_idx)                  # (n_t,1)
    geff = s * g                                          # effective gradient
    q = GAMMA * dt * jnp.cumsum(geff, axis=0)
    q2 = jnp.sum(q ** 2, axis=1)
    gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=1) + 1e-30)
    slewnorm = jnp.sqrt(jnp.sum((jnp.diff(g, axis=0) / dt) ** 2, axis=1) + 1e-30)

    # gradient moments (velocity/acceleration) and the Maxwell matrix
    M1 = jnp.sum(t_arr[:, None] * geff, axis=0) * dt                  # (3,)
    M2 = jnp.sum((t_arr ** 2)[:, None] * geff, axis=0) * dt           # (3,)
    Mmx = dt * jnp.sum(s[:, :, None] * g[:, :, None] * g[:, None, :], axis=0)  # (3,3)

    f_rms = _rms_frequency(g, q)
    c_spec = jnp.where(f_target > 0.0,
                       (f_rms - f_target) ** 2 / (f_target ** 2 + 1e-30), 0.0)

    c = jnp.array([
        jnp.sum(q[-1] ** 2) / (jnp.max(q2) + 1e-30),                  # 0 refocus (M0)
        _shape_penalty(B, b_delta),                                  # 1 shape
        jnp.sum(g[-1] ** 2) / g_max ** 2,                            # 2 g(TE)=0
        jnp.mean((gnorm * rf_mask) ** 2) / g_max ** 2,               # 3 RF window
        jnp.max(jnp.maximum(slewnorm - s_max, 0.0) ** 2) / s_max ** 2,  # 4 slew
        jnp.sum(M1 ** 2) / (g_max * TE ** 2) ** 2,                   # 5 M1 (velocity)
        jnp.sum(M2 ** 2) / (g_max * TE ** 3) ** 2,                   # 6 M2 (accel)
        jnp.sum(Mmx ** 2) / (g_max ** 2 * TE) ** 2,                  # 7 Maxwell
        c_spec,                                                      # 8 spectral (f_rms→f)
    ])
    return b, c


# ===========================================================================
# Public result + designer
# ===========================================================================

@dataclass
class WaveformDesign:
    """An optimized tensor-valued gradient waveform + its acceptance metrics."""
    G: np.ndarray            # (n_t, 3) physical gradient, T/m
    dt: float
    echo_idx: int
    b_value: float           # s/m²
    b_delta: float
    b_delta_target: float
    max_amplitude: float     # max |G|, T/m
    max_slew: float          # max |dG/dt|, T/m/s
    refocus_residual: float  # |q(TE)| / max|q|
    rf_window_leak: float    # max |G| inside the RF window, T/m
    m1_index: float          # |M1| / (G_max·TE²)   — 0 when velocity-compensated
    m2_index: float          # |M2| / (G_max·TE³)   — 0 when accel-compensated
    maxwell_index: float     # ||M||_F / (G_max²·TE) — 0 when Maxwell-compensated
    spectral_rms_hz: float   # RMS encoding frequency (≈0 PGSE-like, ≈f for OGSE)
    spectral_centroid_hz: float   # centroid of |q̃(f)|² (one-sided)
    spectral_bandwidth_hz: float  # rms spread of |q̃(f)|² — "how imprecise" the frequency is
    spectral_target_hz: float     # requested spectral_freq, or None
    active_constraints: tuple
    feasible: bool
    report: dict

    def effective_G(self) -> np.ndarray:
        """Effective (sign-folded) gradient — what dmipy-sim's b-from-waveform eats."""
        s = np.where(np.arange(self.G.shape[0]) < self.echo_idx, 1.0, -1.0)[:, None]
        return self.G * s

    def to_sim_waveform(self, b_target: float | None = None):
        """Build a dmipy-sim ``Waveform`` (effective gradient, echo at TE) for MC.

        Hands the physical gradient to the dmipy-sim forward / MC pipeline.
        ``b_target`` (s/m²) optionally rescales the amplitude (b ∝ |G|²; the
        b-tensor shape and refocusing are invariant under the rescale).
        """
        from dmipy_sim.waveforms import Waveform
        G = self.effective_G()
        if b_target is not None:
            G = G * np.sqrt(b_target / self.b_value)
        return Waveform(G=G[None].astype(np.float32), dt=self.dt,
                        echo_idx=self.G.shape[0] - 1)


def design_waveform(
    b_delta: float,
    *,
    G_max: float = 0.08,
    slew_rate_max: float = 200.0,
    TE: float = None,
    n_t: int = 256,
    echo_frac: float = 0.5,
    rf_duration: float = 0.004,
    timing: 'SequenceTiming' = None,
    null_M1: bool = True,
    null_M2: bool = True,
    maxwell: bool = True,
    spectral_freq: float = None,
    n_restarts: int = 48,
    seed: int = 0,
    inner_maxiter: int = 300,
    n_outer: int = 16,
    verbose: bool = False,
) -> WaveformDesign:
    """Design a hardware-realizable tensor-valued spin-echo gradient waveform.

    Parameters
    ----------
    b_delta : float
        Target b-tensor shape: 1 = LTE, 0 = STE, -0.5 = PTE.
    G_max, slew_rate_max, TE : float
        Hardware constraints (T/m, T/m/s, s).  Defaults are 3T Prisma.
    null_M1, null_M2, maxwell : bool, default True
        Robustness constraints (velocity / acceleration moment nulling and
        concomitant-field compensation).  ON by default — disabling one only
        when its confound is absent (static sample, symmetric waveform, …); see
        the "Robustness constraints" section of the module docstring for the
        bias each suppresses, the b-cost, and when it is safe to turn off.  The
        fully-constrained problem is the hardest to converge; if a heavily
        constrained design comes back ``feasible=False``, raise ``n_restarts`` /
        ``n_outer``.
    spectral_freq : float or None, default None
        Target RMS encoding frequency (Hz) — the spectrum-paradigm OGSE knob.
        ``None`` leaves the spectral content free (a low-frequency PGSE-like
        waveform for ``b_delta=1``); a value adds a constraint driving the RMS
        encoding frequency to it, yielding an oscillating (OGSE-like) waveform at
        that frequency — for any ``b_delta`` (e.g. a frequency-dependent STE).
        The *realized* spectrum is always reported (``spectral_rms_hz`` /
        ``spectral_centroid_hz`` / ``spectral_bandwidth_hz``): per the spectral
        formalism (``ln S ≈ −∫ D(ω)|q̃(ω)|² dω``) you propagate the actual encoding
        spectrum into modelling rather than assuming a pure single frequency, so
        the bandwidth quantifies how monochromatic the result actually is.
    n_t, echo_frac, rf_duration : see module docstring.
    n_restarts, seed, inner_maxiter, n_outer : solver controls.

    Returns
    -------
    WaveformDesign
        ``.m1_index / .m2_index / .maxwell_index`` are always reported (0 ≈
        compensated), regardless of which flags were active — so toggling a flag
        shows what changed and what it cost in ``.b_value``.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX + jaxopt are required for design_waveform.")

    # Resolve TE: explicit > timing's native TE > default.
    if TE is None:
        TE = timing.TE if (timing is not None and timing.TE is not None) else 0.080
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    if timing is not None:
        # Real sequence budget pins the encoding windows + the 180 (TE/2); echo_frac
        # / rf_duration are ignored and any pre/post asymmetry is derived, not chosen.
        slew_off_np, echo_idx = timing.masks(TE, n_t)        # (n_t,1), int
        rf_mask_np = 1.0 - slew_off_np[:, 0]                  # g≈0 in ALL off-regions
    else:
        echo_idx = int(round(echo_frac * (n_t - 1)))
        rf_mask_np = (np.abs(t - echo_idx * dt) <= 0.5 * rf_duration).astype(np.float64)
    rf_mask = jnp.asarray(rf_mask_np)
    slew_off_mask = jnp.asarray(1.0 - rf_mask_np)[:, None]
    t_arr = jnp.asarray(t)
    b_scale = (GAMMA * G_max) ** 2 * TE ** 3 / 50.0       # ~ achievable LTE b

    # active constraint subset (validity always on; flags add M1/M2/Maxwell/spectral)
    f_target = float(spectral_freq) if spectral_freq else 0.0
    active = list(_BASE_CONSTRAINTS)
    if null_M1:
        active.append(5)
    if null_M2:
        active.append(6)
    if maxwell:
        active.append(7)
    if f_target > 0.0:
        active.append(8)
    active_idx = jnp.asarray(active)
    active_names = tuple(CONSTRAINT_NAMES[i] for i in active)
    n_active = len(active)

    # structured (q-MAS-like) + random warm starts
    key = jax.random.PRNGKey(seed)
    kf, kp, ka, kx, kn = jax.random.split(key, 5)
    tt = (jnp.arange(n_t) / (n_t - 1))[None, :, None]
    freqs = jax.random.randint(kf, (n_restarts, 1, 3), 1, 7)
    phase = jax.random.uniform(kp, (n_restarts, 1, 3), minval=0.0, maxval=2 * np.pi)
    amp = jax.random.uniform(ka, (n_restarts, 1, 3), minval=1.0, maxval=3.0)
    axis_w = jax.random.uniform(kx, (n_restarts, 1, 3), minval=0.2, maxval=1.0)
    raw = (amp * axis_w * jnp.sin(2 * np.pi * freqs * tt + phase)
           + 0.3 * jax.random.normal(kn, (n_restarts, n_t, 3)))

    bc = lambda r: _b_and_constraints(r, dt, echo_idx, slew_rate_max, G_max,
                                      b_delta, rf_mask, slew_off_mask, t_arr, TE,
                                      f_target)

    # --- augmented Lagrangian, per-restart multipliers, vmapped on GPU ---
    # The loss + LBFGS solver are built ONCE (μ, λ flow in as traced args) and the whole
    # outer loop runs inside a single jitted lax.scan, so the vmapped LBFGS graph compiles
    # ONE time.  (The previous version re-`def`-ed al_loss and re-constructed LBFGS inside
    # a Python `for` loop; each fresh closure is a jit-cache miss, so the expensive
    # vmapped-LBFGS graph was XLA-recompiled ~n_outer times -- that recompilation, not the
    # arithmetic, dominated the wall clock for this small problem.)
    def al_loss(r, lam_r, mu_s):
        b, c_all = bc(r)
        c = c_all[active_idx]
        return -b / b_scale + jnp.sum(lam_r * c) + 0.5 * mu_s * jnp.sum(c ** 2)

    solver = LBFGS(fun=al_loss, maxiter=inner_maxiter, tol=1e-8,
                   jit=True, history_size=10)

    def _outer_step(carry, _):
        raw_c, lam_c, mu_c = carry
        raw_n = jax.vmap(lambda r, l: solver.run(r, l, mu_c).params)(raw_c, lam_c)
        cs = jax.vmap(lambda r: bc(r)[1][active_idx])(raw_n)
        lam_n = lam_c + mu_c * cs
        mu_n = jnp.minimum(mu_c * 4.0, 1e6)
        return (raw_n, lam_n, mu_n), jnp.min(jnp.max(cs, axis=1))

    @jax.jit
    def _run_al(raw0, lam0):
        return jax.lax.scan(_outer_step, (raw0, lam0, jnp.float32(10.0)),
                            None, length=n_outer)

    (raw, lam, mu), maxc_hist = _run_al(raw, jnp.zeros((n_restarts, n_active)))
    if verbose:
        for outer, h in enumerate(np.asarray(maxc_hist)):
            print(f"  outer {outer}: max|c| best={float(h):.2e}")

    # --- evaluate all metrics, pick best feasible (max b) ---
    def _metrics(r):
        g = _waveform_from_raw(r, dt, slew_rate_max, G_max, slew_off_mask)
        B = b_tensor(g, dt, echo_idx)
        q = effective_q(g, dt, echo_idx)
        gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=1) + 1e-30)
        slew = jnp.sqrt(jnp.sum((jnp.diff(g, axis=0) / dt) ** 2, axis=1) + 1e-30)
        refoc = jnp.sqrt(jnp.sum(q[-1] ** 2)) / (jnp.sqrt(jnp.max(jnp.sum(q ** 2, axis=1))) + 1e-30)
        _, c_all = bc(r)
        return jnp.array([b_value(B), b_delta_of(B), jnp.max(gnorm), jnp.max(slew),
                          refoc, jnp.max(gnorm * rf_mask),
                          jnp.sqrt(c_all[5]), jnp.sqrt(c_all[6]), jnp.sqrt(c_all[7]),
                          _rms_frequency(g, q)]), g

    metrics, gs = jax.vmap(_metrics)(raw)
    metrics = np.asarray(metrics)
    gs = np.asarray(gs)
    (b_all, bd_all, amp_all, slew_all, refoc_all, win_all,
     m1_all, m2_all, mx_all, frms_all) = metrics.T

    feas = ((np.abs(bd_all - b_delta) < 2e-2) & (amp_all <= G_max * 1.01)
            & (slew_all <= slew_rate_max * 1.02) & (refoc_all < 1e-2)
            & (win_all <= G_max * 0.02))
    if null_M1:
        feas &= (m1_all < 5e-2)
    if null_M2:
        feas &= (m2_all < 5e-2)
    if maxwell:
        feas &= (mx_all < 5e-2)
    if f_target > 0.0:
        feas &= (np.abs(frms_all - f_target) / f_target < 0.15)   # RMS freq near target

    if feas.any():
        cand = np.where(feas)[0]
        best = int(cand[np.argmax(b_all[cand])])
        feasible = True
    else:
        viol = (np.maximum(np.abs(bd_all - b_delta) - 2e-2, 0)
                + np.maximum(amp_all - G_max, 0) / G_max
                + np.maximum(slew_all - slew_rate_max, 0) / slew_rate_max
                + refoc_all + win_all / G_max
                + (m1_all if null_M1 else 0) + (m2_all if null_M2 else 0)
                + (mx_all if maxwell else 0)
                + (np.abs(frms_all - f_target) / f_target if f_target > 0 else 0))
        best = int(np.argmin(viol))
        feasible = False

    G_best = gs[best].astype(np.float64)
    _, _, spec_centroid, spec_bw, spec_rms = encoding_spectrum(G_best, dt, echo_idx)
    return WaveformDesign(
        G=G_best, dt=dt, echo_idx=echo_idx,
        b_value=float(b_all[best]), b_delta=float(bd_all[best]),
        b_delta_target=float(b_delta), max_amplitude=float(amp_all[best]),
        max_slew=float(slew_all[best]), refocus_residual=float(refoc_all[best]),
        rf_window_leak=float(win_all[best]), m1_index=float(m1_all[best]),
        m2_index=float(m2_all[best]), maxwell_index=float(mx_all[best]),
        spectral_rms_hz=spec_rms, spectral_centroid_hz=spec_centroid,
        spectral_bandwidth_hz=spec_bw, spectral_target_hz=(f_target or None),
        active_constraints=active_names, feasible=bool(feasible),
        report={'n_restarts': n_restarts, 'n_feasible': int(feas.sum()),
                'b_scale': float(b_scale), 'dt': dt,
                'b_feasible_max': float(b_all[feas].max()) if feas.any() else None},
    )


def min_te_for_b(b_target, b_delta, *, timing=None, te_lo=None, te_hi=None,
                 te_max=0.25, tol_te=1e-3, n_seeds=1, verbose=False, **design_kwargs):
    """Smallest TE whose max-b design reaches ``b_target`` -- the first-class
    min-TE-at-given-b mode.

    ``design_waveform`` only maximizes b at a FIXED TE.  Because the achievable b is
    monotonically increasing in TE, the minimum TE that reaches a target b is found by
    BISECTING TE around that max-b primitive -- which is exactly what this does (no
    hand-rolled scan).  Each bracket/bisection step is one ``design_waveform`` call (= one
    JIT compile); a batched-TE-in-one-compile version is the natural future speed-up.

    Parameters
    ----------
    b_target : float -- required b-value (s/m^2).
    b_delta : float -- target b-tensor shape, passed straight to ``design_waveform``.
    timing : SequenceTiming or None -- the timing budget; its ``min_TE()`` is the hard
        lower floor for the bracket.
    te_lo, te_hi : float or None -- optional TE bracket (s).  Defaults: ``te_lo`` = the
        timing floor (else 1 ms); ``te_hi`` is grown x1.5 until b_target is reached.
    te_max : float -- cap (s) for the upward search; raises if b_target is unreachable.
    tol_te : float -- stop when the TE bracket is narrower than this (s).
    n_seeds : int -- design each TE over this many restart-RNG seeds (0..n_seeds-1) and
        keep the BEST FEASIBLE (max b).  >1 makes b(TE) robust to the optimizer's seed-to-
        seed variance on tight problems (so the result is reproducible and a fair match to
        an equally seed-robust vanilla design); costs n_seeds designs per TE.
    **design_kwargs -- forwarded to ``design_waveform`` (G_max, slew_rate_max, n_t,
        n_restarts, n_outer, null_M1/M2/maxwell, spectral_freq, ...; any ``seed`` is
        overridden by the n_seeds sweep).

    Returns
    -------
    (design, te) : the feasible :class:`WaveformDesign` at the smallest TE reaching
        ``b_target``, and that TE (s).
    """
    dkw = {k: v for k, v in design_kwargs.items() if k != 'seed'}

    def reached(te):
        best = None
        for s in range(max(1, n_seeds)):
            d = design_waveform(b_delta, TE=te, timing=timing, seed=s, **dkw)
            # prefer feasible designs; among equal feasibility, the higher b
            if best is None or (bool(d.feasible), d.b_value) > (bool(best.feasible),
                                                                best.b_value):
                best = d
        ok = bool(best.feasible) and (best.b_value >= b_target)
        if verbose:
            print(f"  min_te_for_b: TE={te*1e3:6.2f}ms  b={best.b_value/1e6:7.0f}  "
                  f"feasible={best.feasible}  -> {'reaches' if ok else 'short'}")
        return ok, best

    floor = timing.min_TE() if timing is not None else 1e-3
    lo = max(float(te_lo) if te_lo is not None else floor, floor)
    ok_lo, d_lo = reached(lo)
    if ok_lo:
        return d_lo, lo                                   # floor already reaches it
    hi = float(te_hi) if te_hi is not None else lo * 1.6
    ok_hi, d_hi = reached(hi)
    while not ok_hi:                                      # grow the upper bracket
        if hi > te_max:
            raise ValueError(
                f"b_target={b_target/1e6:.0f} s/mm^2 not reached below te_max="
                f"{te_max*1e3:.0f} ms (best b={d_hi.b_value/1e6:.0f}).")
        hi *= 1.5
        ok_hi, d_hi = reached(hi)
    best_d = d_hi                                         # invariant: lo short, hi reaches
    while hi - lo > tol_te:
        mid = 0.5 * (lo + hi)
        ok_mid, d_mid = reached(mid)
        if ok_mid:
            hi, best_d = mid, d_mid
        else:
            lo = mid
    return best_d, hi
