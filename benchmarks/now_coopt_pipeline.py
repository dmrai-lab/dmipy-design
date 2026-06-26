"""NOW -> differentiable-Bloch co-optimization, with RF DELIVERABILITY constraints.

Stage 1 -- NOW (scipy SQP) designs the constraint-OPTIMAL diffusion gradient (max b at the
exact b-tensor shape, riding slew/amplitude, M1/M2 nulled, q(TE)=0).  Closed-form b objective,
no Bloch, no autodiff -> active-set SQP is the right tool and gets the true optimum.

Stage 2 -- a differentiable spin-echo Bloch (JAX) takes that exact gradient as a fixed
warm-start and optimizes the 180 refocusing RF over a realistic (B1+ transmit x off-resonance)
ensemble.  Two variants are compared:
  (2a) UNCONSTRAINED -- optimize raw per-step flips.  Maximizes robustness but the RF is not
       deliverable (abrupt steps -> huge RF slew, hard 0<->pi phase jumps, unbounded SAR/peak).
  (2b) DELIVERABLE -- the RF analogue of NOW's gradient constraints:
         * band-limited envelope (optimize a few low-frequency cosine coefficients) -> bounds
           the RF slew / bandwidth STRUCTURALLY (the RF analogue of slew-rate);
         * peak-B1 penalty  |B1| <= B1_max      (RF analogue of the gradient amplitude box);
         * SAR penalty       integral B1^2 dt <= budget   (RF analogue of the heat constraint);
         * finer raster than the 14-step PoC.
       The limits come from SCANNER (provisional typical-3T values, to be replaced by the cited
       dmipy_sim scanner-constants catalogue once that lands).

Run:  python benchmarks/now_coopt_pipeline.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax, jax.numpy as jnp
from jax import grad, jit
import optax

from dmipy_design.optimizers import design_waveform_now
from dmipy_sim.sequences import scanner_constants as scc

GAMMA = 267.513e6
jax.config.update("jax_enable_x64", True)

# ---- hardware/safety limits pulled from the CITED scanner-constants catalogue ----
# GE SIGNA Premier (3T): the model with cited RF numbers (peak B1 19 µT @75 kg, RF raster 1 µs).
SCANNER_MODEL = "ge_signa_premier_3T"
G_max, slew_max = scc.gradient_limits(SCANNER_MODEL, regime="diffusion")   # T/m, T/m/s (PNS-aware)
SCANNER = dict(
    B1_max=scc.get_limit(SCANNER_MODEL, "rf", "peak_B1_body_coil", si=True),  # 19 µT @75kg, cited
    rf_raster=scc.get_limit("siemens_magnetom_prisma_3T", "rf", "rf_raster_time", si=True),  # 1 µs (IDEA/Pulseq std), cited
    sar_headroom=1.30,   # SAR budget as a multiple of the plain hard-180 power (relative proxy;
)                        # the absolute IEC W/kg limit lives in scc.sar_limit())

# ----------------------------------------------------------------------------
# Stage 1 -- NOW designs the constraint-optimal LTE gradient (finer grid for RF resolution).
# ----------------------------------------------------------------------------
print("Scanner: %s  (G_max=%.0f mT/m, slew=%.0f T/m/s diffusion-derated, peakB1=%.0f uT, "
      "RF raster=%.0f us — all cited)"
      % (SCANNER_MODEL, G_max * 1e3, slew_max, SCANNER["B1_max"] * 1e6, SCANNER["rf_raster"] * 1e6),
      flush=True)
TE, n_t = 0.060, 280
d = design_waveform_now(b_delta=1.0, G_max=G_max, slew_rate_max=slew_max, TE=TE, n_t=n_t,
                        null_M1=True, null_M2=True, maxwell=False, n_restarts=6)
dt = d.dt; echo = d.echo_idx; b_now = d.b_value
geff = np.asarray(d.effective_G())
g_axis = geff[:, int(np.argmax(np.sum(geff ** 2, axis=0)))]
print("Stage 1  NOW gradient:  b=%.0f s/mm^2  feasible=%s  slew=%.0f  refoc=%.0e"
      % (b_now * 1e-6, d.feasible, d.max_slew, d.refocus_residual), flush=True)

# the 180 lives in NOW's contiguous gradient-off refocus window around echo_idx
zero = np.abs(g_axis) < 1e-9 * np.max(np.abs(g_axis))
lo = echo
while lo > 0 and zero[lo - 1]: lo -= 1
hi = echo
while hi < n_t - 1 and zero[hi + 1]: hi += 1
win = np.arange(lo, hi + 1); nwin = win.size
win_dur_ms = nwin * dt * 1e3
print("Stage 2  RF window: %d samples, %.2f ms  (grid raster %.0f us; true DAC raster %.0f us)"
      % (nwin, win_dur_ms, dt * 1e6, SCANNER["rf_raster"] * 1e6), flush=True)

# ensemble: B1+ transmit scale x static off-resonance
b1 = jnp.asarray(np.linspace(0.7, 1.3, 7)); dw = jnp.asarray(np.linspace(-250.0, 250.0, 7) * 2 * np.pi)
B1 = jnp.repeat(b1, dw.size); DW = jnp.tile(dw, b1.size); E = B1.size
D1, D2 = 0.5e-9, 1.5e-9
diff_sep = abs(np.exp(-b_now * D1) - np.exp(-b_now * D2))
win_j = jnp.asarray(win)


def _rotx(M, a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack([M[0], c * M[1] - s * M[2], s * M[1] + c * M[2]])
def _rotz(M, a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack([c * M[0] - s * M[1], s * M[0] + c * M[1], M[2]])


def flips_full(flips_win):
    """Embed the window flips (signed, summing to pi in magnitude) into an (n_t,) array."""
    f = jnp.pi * flips_win / (jnp.sum(jnp.abs(flips_win)) + 1e-9)
    return jnp.zeros(n_t).at[win_j].set(f)


def refoc_fraction(flips_win):
    ff = flips_full(flips_win)
    M = jnp.stack([jnp.zeros(E), jnp.zeros(E), jnp.ones(E)])
    M = _rotx(M, (jnp.pi / 2) * B1)                              # 90_x (B1-scaled)
    def body(i, M):
        M = _rotx(M, ff[i] * B1)                                # 0 outside the window
        return _rotz(M, DW * dt)                                # off-resonance precession
    M = jax.lax.fori_loop(0, n_t, body, M)
    return jnp.abs(jnp.mean(M[0] + 1j * M[1]))


def rf_metrics(flips_win):
    """(peak_B1 T, SAR proxy = integral B1^2 dt, max RF slew T/m... T/s)."""
    f = jnp.pi * flips_win / (jnp.sum(jnp.abs(flips_win)) + 1e-9)
    B1t = f / (GAMMA * dt)                                       # B1 amplitude (T) per window step
    peak = jnp.max(jnp.abs(B1t))
    sar = jnp.sum(B1t ** 2) * dt
    slew = jnp.max(jnp.abs(jnp.diff(B1t)) / dt)
    return peak, sar, slew


# reference SAR of a plain flat (hard) 180 over the window -> sets the deliverable budget
hard = jnp.ones(nwin)
pk_h, sar_h, sl_h = (float(x) for x in rf_metrics(hard))
r_hard = float(refoc_fraction(hard))
SAR_BUDGET = SCANNER["sar_headroom"] * sar_h
print("Stage 2  HARD 180 (flat):     refoc=%.3f  peakB1=%.1fuT  SAR=%.2g  RFslew=%.0f uT/ms"
      % (r_hard, pk_h * 1e6, sar_h, sl_h * 1e3), flush=True)


def run_adam(loss_fn, p0, steps=500, lr=3e-2):
    g = jit(grad(loss_fn)); opt = optax.adam(lr); st = opt.init(p0); p = p0
    for _ in range(steps):
        up, st = opt.update(g(p), st); p = optax.apply_updates(p, up)
    return p


# (2a) UNCONSTRAINED -- raw per-step flips, robustness only
env_unc = run_adam(lambda f: -refoc_fraction(f) * diff_sep, hard)
r_unc = float(refoc_fraction(env_unc)); pk_u, sar_u, sl_u = (float(x) for x in rf_metrics(env_unc))
print("Stage 2a UNCONSTRAINED co-opt: refoc=%.3f  peakB1=%.1fuT  SAR=%.2g (%.1fx)  RFslew=%.0f uT/ms"
      % (r_unc, pk_u * 1e6, sar_u, sar_u / sar_h, sl_u * 1e3), flush=True)

# (2b) DELIVERABLE -- band-limited envelope (cosine basis) + peak-B1 + SAR penalties
N_BASIS = 8
ii = (np.arange(nwin)[:, None] + 0.5) / nwin
Bmat = jnp.asarray(np.cos(np.pi * np.arange(N_BASIS)[None, :] * ii))   # (nwin, N_BASIS) DCT-II

def env_from_coeffs(c):
    return Bmat @ c                                              # band-limited -> RF slew bounded

def loss_deliverable(c):
    f = env_from_coeffs(c)
    peak, sar, _ = rf_metrics(f)
    contrast = refoc_fraction(f) * diff_sep
    pen = 50.0 * jnp.maximum(peak / SCANNER["B1_max"] - 1.0, 0.0) ** 2 \
        + 50.0 * jnp.maximum(sar / SAR_BUDGET - 1.0, 0.0) ** 2
    return -contrast + pen

c0 = jnp.zeros(N_BASIS).at[0].set(1.0)                          # start ~flat
c_opt = run_adam(loss_deliverable, c0, steps=800)
env_del = env_from_coeffs(c_opt)
r_del = float(refoc_fraction(env_del)); pk_d, sar_d, sl_d = (float(x) for x in rf_metrics(env_del))
print("Stage 2b DELIVERABLE co-opt:   refoc=%.3f  peakB1=%.1fuT  SAR=%.2g (%.1fx)  RFslew=%.0f uT/ms"
      % (r_del, pk_d * 1e6, sar_d, sar_d / sar_h, sl_d * 1e3), flush=True)
print("  peakB1<=%.0fuT? %s   SAR<=%.1fx? %s   (band-limited basis bounds RF slew structurally)"
      % (SCANNER["B1_max"] * 1e6, pk_d <= SCANNER["B1_max"] * 1.02,
         SCANNER["sar_headroom"], sar_d <= SAR_BUDGET * 1.02), flush=True)

np.savez("/home/rutger/dmipy-scratch/now_coopt.npz",
         G=np.asarray(d.G), dt=dt, echo=echo, win=win, b_now=b_now,
         hard=np.asarray(hard), env_unc=np.asarray(env_unc), env_del=np.asarray(env_del),
         b1=np.asarray(b1), dw=np.asarray(dw),
         B1_max=SCANNER["B1_max"], sar_budget=SAR_BUDGET, sar_hard=sar_h)
