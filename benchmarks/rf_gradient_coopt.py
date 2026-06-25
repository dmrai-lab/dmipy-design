"""RF + sequence CO-OPTIMIZATION prototype (differentiable Bloch).

Demonstrates the capability NOW / our SQP-AL designer lack: jointly optimize the diffusion
GRADIENT and the 180 refocusing RF (its B1 envelope) by back-propagating through a
differentiable spin-echo Bloch forward over a (B1+, off-resonance) ensemble.

Sequence: 90_x excite -> pre gradient lobe -> finite shaped 180 (B1 envelope, the co-opt
variable) -> post gradient lobe -> echo.  Ensemble = transmit-inhomogeneity B1 scales x
static off-resonance (susceptibility/B0 spread).  Objective = diffusion CONTRAST between two
diffusivities, contrast = refoc_fraction(RF, ensemble) * |exp(-b D1) - exp(-b D2)|, with the
b-value emerging from the gradient and the refocused fraction emerging from the RF -- so a
B1-robust 180 retains signal a hard 180 loses.  Co-optimize (gradient scale, 180 envelope)
jointly; compare to the fixed hard-180 baseline.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax, jax.numpy as jnp
from jax import grad, jit
GAMMA = 267.513e6

# ── ensemble: B1+ transmit scale x off-resonance ──
b1 = jnp.asarray(np.linspace(0.7, 1.3, 7))
dw = jnp.asarray(np.linspace(-250.0, 250.0, 7) * 2 * np.pi)     # rad/s (±250 Hz B0 spread)
B1 = jnp.repeat(b1, dw.size); DW = jnp.tile(dw, b1.size)        # (E,)
E = B1.size

n_t, TE = 160, 0.060
dt = TE / (n_t - 1); echo = n_t // 2
K = 24                                                          # 180 envelope length (samples)
pidx = np.arange(echo - K // 2, echo - K // 2 + K)             # 180 window indices
D1, D2 = 0.5e-9, 1.5e-9                                         # m^2/s diffusivities to separate
# gradient SHAPE: matched bipolar lobes pre/post (sign-flipped by the 180), area sets b.
gshape = np.zeros(n_t)
gshape[3:echo - K // 2 - 3] = 1.0                              # pre lobe
gshape[echo + K // 2 + 3:n_t - 3] = -1.0                       # post lobe (refocused by 180)
gshape = jnp.asarray(gshape)

def _rotx(M, ang):                                             # rotate about x by ang (E,)
    c, s = jnp.cos(ang), jnp.sin(ang)
    return jnp.stack([M[0], c * M[1] - s * M[2], s * M[1] + c * M[2]])
def _rotz(M, ang):
    c, s = jnp.cos(ang), jnp.sin(ang)
    return jnp.stack([c * M[0] - s * M[1], s * M[0] + c * M[1], M[2]])

def forward(env, g0):
    """Return (refoc_fraction, b_value) for envelope env (K,) and gradient amp g0 (T/m)."""
    env180 = jnp.pi * env / (jnp.sum(jnp.abs(env)) + 1e-9)      # per-step flips summing to pi
    rf_step = {int(pidx[j]): env180[j] for j in range(K)}
    M = jnp.stack([jnp.zeros(E), jnp.zeros(E), jnp.ones(E)])
    M = _rotx(M, (jnp.pi / 2) * B1)                            # 90_x excitation (B1-scaled)
    q = 0.0
    for i in range(n_t):
        if i in rf_step:
            M = _rotx(M, rf_step[i] * B1)                      # 180 sub-rotation (B1-scaled)
        gi = g0 * gshape[i]
        q = q + GAMMA * gi * dt                                # k-space (for b)
        M = _rotz(M, DW * dt)                                  # off-resonance precession
    refoc = jnp.abs(jnp.mean(M[0] + 1j * M[1]))                # ensemble-mean transverse mag
    b = jnp.sum((GAMMA * jnp.cumsum(jnp.where(jnp.arange(n_t) < echo, 1.0, -1.0) * g0 * gshape)
                 * dt) ** 2) * dt
    return refoc, b

def contrast(params):
    env, g0 = params[:K], params[K]
    refoc, b = forward(env, g0)
    return refoc * jnp.abs(jnp.exp(-b * D1) - jnp.exp(-b * D2))
neg = jit(lambda p: -contrast(p))
g = jit(grad(lambda p: -contrast(p)))

# baseline: fixed HARD 180 (flat envelope), optimize only g0
hard_env = jnp.ones(K)
def neg_g0(g0):
    return -contrast(jnp.concatenate([hard_env, jnp.array([g0])]))
# coarse 1-D search for the best g0 with the hard pulse
g0s = np.linspace(0.01, 0.08, 40)
cs = [contrast(jnp.concatenate([hard_env, jnp.array([x])])) for x in g0s]
g0_hard = float(g0s[int(np.argmax(cs))]); c_hard = float(np.max(cs))
rf_hard, b_hard = forward(hard_env, g0_hard)
print("baseline  HARD 180 + best g0: contrast=%.4f  refoc=%.3f  b=%.0f  g0=%.3f"
      % (c_hard, float(rf_hard), float(b_hard) / 1e6, g0_hard), flush=True)

# CO-OPT: jointly optimize (180 envelope, g0) by Adam through the differentiable Bloch
import optax
p = jnp.concatenate([hard_env, jnp.array([g0_hard])])
opt = optax.adam(3e-2); st = opt.init(p)
for it in range(400):
    gr = g(p); up, st = opt.update(gr, st); p = optax.apply_updates(p, up)
    p = p.at[K].set(jnp.clip(p[K], 0.005, 0.085))             # keep g0 in range
env_o, g0_o = p[:K], float(p[K])
rf_o, b_o = forward(env_o, g0_o)
c_o = float(contrast(p))
print("co-opt    JOINT 180-envelope + g0:  contrast=%.4f  refoc=%.3f  b=%.0f  g0=%.3f"
      % (c_o, float(rf_o), float(b_o) / 1e6, g0_o), flush=True)
print("  -> co-opt / hard contrast gain = %.2fx ; refoc %.3f->%.3f" % (c_o / c_hard, float(rf_hard), float(rf_o)), flush=True)
np.savez("/home/rutger/dmipy-scratch/coopt.npz", env=np.asarray(env_o), hard=np.asarray(hard_env),
         pidx=pidx, b1=np.asarray(b1), dw=np.asarray(dw))
