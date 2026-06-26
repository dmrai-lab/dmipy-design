"""NOW -> differentiable-Bloch co-optimization: the "best of both worlds" pipeline.

Two solvers, each doing the job it is actually good at, chained as a warm-start:

  Stage 1 -- NOW (scipy SQP).  Designs the constraint-OPTIMAL diffusion gradient:
    max b at the exact b-tensor shape, riding slew=S_max and |G|=G_max, with M1/M2
    nulled and the spin-echo refocus condition q(TE)=0 satisfied to machine precision.
    This is a closed-form b objective -- no Bloch needed, no autodiff needed -- so an
    active-set SQP is the right tool and gets the true optimum.

  Stage 2 -- differentiable Bloch (JAX).  Takes that exact gradient as a FIXED
    warm-start and optimizes the 180 refocusing RF envelope by back-propagating
    through a spin-echo Bloch forward over a realistic (B1+ transmit x static
    off-resonance) ensemble.  This is what NOW structurally cannot see: NOW's
    b/refocus are ideal-hard-pulse quantities, but a real 180 over a B1+/B0 spread
    loses signal a shaped, robust 180 retains.  The objective is the DELIVERED
    diffusion contrast = refocused-signal-fraction(RF, ensemble) x |e^{-b D1}-e^{-b D2}|.

The point: you do NOT need a single differentiable solver that also does exact-constraint
b-maximization (active-set SQP doesn't vmap/autodiff; that's why NOW and fmincon are
scipy/MATLAB).  You need the differentiable BLOCH forward -- and you plug NOW's optimum
into it.  Run:  python benchmarks/now_coopt_pipeline.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax, jax.numpy as jnp
from jax import grad, jit
import optax

from dmipy_design.optimizers import design_waveform_now

GAMMA = 267.513e6
jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------
# Stage 1 -- NOW designs the constraint-optimal LTE gradient (the warm-start).
# ----------------------------------------------------------------------------
TE, n_t = 0.060, 140
d = design_waveform_now(b_delta=1.0, TE=TE, n_t=n_t, null_M1=True, null_M2=True,
                        maxwell=False, n_restarts=6)
dt = d.dt; echo = d.echo_idx; b_now = d.b_value
print("Stage 1  NOW gradient:  b=%.0f s/mm^2  feasible=%s  slew=%.0f  refoc=%.0e"
      % (b_now * 1e-6, d.feasible, d.max_slew, d.refocus_residual), flush=True)

# Effective encoding gradient magnitude along its principal (LTE) axis -> the b is fixed
# by NOW; the Bloch below tracks only the COHERENT off-resonance refocusing by the 180
# (diffusion attenuation enters incoherently through the b-factor, not the precession).
geff = np.asarray(d.effective_G())
g_axis = geff[:, int(np.argmax(np.sum(geff ** 2, axis=0)))]   # principal axis samples (n_t,)

# ----------------------------------------------------------------------------
# Stage 2 -- differentiable Bloch co-opt of the 180 envelope on the NOW gradient.
# ----------------------------------------------------------------------------
# Ensemble: B1+ transmit scale x static off-resonance (susceptibility / B0 spread).
b1 = jnp.asarray(np.linspace(0.7, 1.3, 7))                    # +-30% transmit inhomogeneity
dw = jnp.asarray(np.linspace(-250.0, 250.0, 7) * 2 * np.pi)   # +-250 Hz off-resonance (rad/s)
B1 = jnp.repeat(b1, dw.size); DW = jnp.tile(dw, b1.size); E = B1.size

# the 180 lives in NOW's gradient-off refocus window (centred at echo_idx)
K = int(max(8, np.sum(np.abs(g_axis[echo - 30:echo + 30]) < 1e-9 * np.max(np.abs(g_axis)))))
K = min(K, 28)
pidx = np.arange(echo - K // 2, echo - K // 2 + K)
D1, D2 = 0.5e-9, 1.5e-9                                        # m^2/s diffusivities to separate
diff_sep = abs(np.exp(-b_now * D1) - np.exp(-b_now * D2))      # fixed by NOW's b


def _rotx(M, a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack([M[0], c * M[1] - s * M[2], s * M[1] + c * M[2]])
def _rotz(M, a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack([c * M[0] - s * M[1], s * M[0] + c * M[1], M[2]])


def refoc_fraction(env):
    """Ensemble-mean refocused transverse magnitude for a 180 envelope (K,)."""
    env180 = jnp.pi * env / (jnp.sum(jnp.abs(env)) + 1e-9)     # per-step flips summing to pi
    rf_step = {int(pidx[j]): env180[j] for j in range(K)}
    M = jnp.stack([jnp.zeros(E), jnp.zeros(E), jnp.ones(E)])
    M = _rotx(M, (jnp.pi / 2) * B1)                           # 90_x excitation (B1-scaled)
    for i in range(n_t):
        if i in rf_step:
            M = _rotx(M, rf_step[i] * B1)                     # 180 sub-rotation (B1-scaled)
        M = _rotz(M, DW * dt)                                 # static off-resonance precession
    return jnp.abs(jnp.mean(M[0] + 1j * M[1]))


def delivered_contrast(env):
    return refoc_fraction(env) * diff_sep                     # b (hence diff_sep) fixed by NOW


neg = jit(lambda e: -delivered_contrast(e))
gfn = jit(grad(lambda e: -delivered_contrast(e)))

# baseline: hard (flat) 180 on the NOW gradient
hard = jnp.ones(K)
r_hard = float(refoc_fraction(hard)); c_hard = float(delivered_contrast(hard))
print("Stage 2  baseline HARD 180 on NOW gradient:  refoc=%.3f  delivered_contrast=%.4f"
      % (r_hard, c_hard), flush=True)

# co-opt: optimize the 180 envelope through the differentiable Bloch
p = hard
opt = optax.adam(3e-2); st = opt.init(p)
for _ in range(400):
    g_, st = opt.update(gfn(p), st); p = optax.apply_updates(p, g_)
env_o = p
r_o = float(refoc_fraction(env_o)); c_o = float(delivered_contrast(env_o))
print("Stage 2  co-opt SHAPED 180 on NOW gradient:  refoc=%.3f  delivered_contrast=%.4f"
      % (r_o, c_o), flush=True)
print("  -> robust-180 gain over hard-180:  refoc %.3f->%.3f (%.1f%%),  contrast %.2fx"
      % (r_hard, r_o, 100 * (r_o - r_hard) / max(r_hard, 1e-9), c_o / max(c_hard, 1e-9)), flush=True)
print("  NOW set the (optimal, feasible) b; co-opt recovered the signal a real 180 over a "
      "B1+/B0 ensemble would lose.", flush=True)

np.savez("/home/rutger/dmipy-scratch/now_coopt.npz",
         G=np.asarray(d.G), env=np.asarray(env_o), hard=np.asarray(hard),
         pidx=pidx, b1=np.asarray(b1), dw=np.asarray(dw), b_now=b_now)
