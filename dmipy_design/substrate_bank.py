"""
SubstrateBank — registry of biologically grounded substrates for MC bias
estimation in column generation OED.

Each substrate entry holds pre-computed walker trajectories and the nominal
analytical model parameters. The replay engine (apply_waveform_jax) evaluates
any waveform G(t) against stored trajectories without re-running Monte Carlo.

The bias term B(u) measures the normalised squared difference between the
MC-replayed signal and the analytical model prediction, aggregated (weighted)
over all substrates:

    B(u) = Σ_s  w_s · mean_m[(S_mc - S_an)²]  / (mean_m[S_mc²] + ε)

B(u) is differentiable w.r.t. G via JAX autodiff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


@dataclass
class SubstrateEntry:
    """A single biologically grounded substrate configuration.

    Attributes
    ----------
    substrate_id : str
        Unique identifier (e.g. 'wm_cc_small').
    trajectories : jnp.ndarray, shape (n_walkers, n_t, 3)
        Walker positions in metres, float32 on device.
    dt_traj : float
        Trajectory time step in seconds.
    theta_nominal : jnp.ndarray, shape (P,)
        Nominal analytical model parameters for this substrate.
    biological_weight : float
        Relevance weight in the aggregated bias term.
    walker_weights : jnp.ndarray, shape (n_walkers,) or None
        Optional importance weights (e.g. from trajectory compression).
        If None, uniform mean is used.
    meta : dict
        Free-form metadata (geometry type, diffusivities, etc.).
    """
    substrate_id: str
    trajectories: "jnp.ndarray"
    dt_traj: float
    theta_nominal: "jnp.ndarray"
    biological_weight: float
    walker_weights: "jnp.ndarray | None"
    meta: dict = field(default_factory=dict)


class SubstrateBank:
    """Collection of substrate entries for MC bias computation.

    Parameters
    ----------
    entries : list of SubstrateEntry
        All substrate configurations.

    Examples
    --------
    # From manifest YAML + NPZ files on disk:
    bank = SubstrateBank.from_manifest('/path/to/bank_manifest.yaml', max_walkers=50_000)

    # From a synthetic Brownian motion bank (for tests):
    from dmipy_design.substrate_bank_synthetic import make_synthetic_bank
    bank = make_synthetic_bank(n_walkers=2000, n_t=201)
    """

    def __init__(self, entries: list):
        self.entries = entries

    @classmethod
    def from_manifest(cls, manifest_path: str, max_walkers: int = 50_000) -> "SubstrateBank":
        """Load all substrates listed in a bank manifest YAML.

        The manifest YAML schema (see benchmarks/substrate_bank/bank_manifest.yaml)
        lists substrate IDs, trajectory NPZ paths, biological weights, and
        theta_nominal vectors. Each NPZ file must contain a key 'trajectories'
        with shape (n_walkers_full, n_t, 3).

        Parameters
        ----------
        manifest_path : str
            Path to bank_manifest.yaml.
        max_walkers : int
            Maximum number of walkers to load per substrate. If the NPZ
            contains more, a random subsample is drawn (reproducible by seeding).

        Returns
        -------
        SubstrateBank
        """
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is required for SubstrateBank.from_manifest.")

        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required for SubstrateBank.from_manifest.") from e

        manifest_path = Path(manifest_path)
        manifest_dir  = manifest_path.parent

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        entries = []
        rng = np.random.default_rng(0)

        for sub in manifest.get('substrates', []):
            npz_rel  = sub['trajectories_path']
            npz_path = manifest_dir / npz_rel
            if not npz_path.exists():
                print(f"  [skip] {sub['id']}: {npz_path} not found", flush=True)
                continue
            data     = np.load(npz_path)
            traj_np  = data['trajectories'].astype(np.float32)  # (N_full, n_t, 3)

            n_full = traj_np.shape[0]
            if n_full > max_walkers:
                idx = rng.choice(n_full, size=max_walkers, replace=False)
                idx = np.sort(idx)
                traj_np = traj_np[idx]

            traj_jax   = jnp.array(traj_np)
            theta_jax  = jnp.array(np.array(sub['theta_nominal'], dtype=np.float32))
            ww_np      = data.get('walker_weights', None)
            if ww_np is not None:
                walker_weights = jnp.array(ww_np.astype(np.float32))
                if n_full > max_walkers:
                    walker_weights = walker_weights[idx]
            else:
                walker_weights = None

            entry = SubstrateEntry(
                substrate_id=sub['id'],
                trajectories=traj_jax,
                dt_traj=float(sub['dt_s']),
                theta_nominal=theta_jax,
                biological_weight=float(sub.get('biological_weight', 1.0)),
                walker_weights=walker_weights,
                meta={k: v for k, v in sub.items()
                      if k not in ('trajectories_path', 'theta_nominal', 'dt_s',
                                   'biological_weight', 'id')},
            )
            entries.append(entry)

        return cls(entries)

    def compute_bias_jax(
        self,
        G: "jnp.ndarray",
        dt_wf: float,
        forward_fn: Callable,
        scheme,
        sigma: float = 1.0,
    ) -> "jnp.ndarray":
        """Compute the aggregated MC bias term for a given waveform G.

        For each substrate entry:
        1. Replay MC: S_mc = apply_waveform_jax(G, entry.trajectories, ...)
        2. Evaluate analytical model: S_an = forward_fn(entry.theta_nominal, scheme)
        3. Normalised squared error: err = mean[(S_mc - S_an)^2] / (mean[S_mc^2] + 1e-6)
        4. Weight by entry.biological_weight

        Returns the weighted average over all substrates, normalised by total weight.

        Parameters
        ----------
        G : jnp.ndarray, shape (n_meas, n_t_wf, 3)
            Gradient waveform array in T/m. JAX-traced (differentiable).
        dt_wf : float
            Waveform time step in seconds.
        forward_fn : callable (theta_nominal, scheme) -> jnp.ndarray, shape (n_meas,)
            Analytical model forward function.
        scheme : JaxScheme
            The JaxScheme corresponding to the G waveform (for the analytical call).
        sigma : float
            Noise sigma (currently unused, reserved for future SNR-weighted bias).

        Returns
        -------
        bias : jnp.ndarray, scalar
            Differentiable bias term B(u), dimensionless.
        """
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is required for compute_bias_jax.")

        from dmipy_sim.trajectories import apply_waveform_jax  # lazy import

        weighted_err_sum  = jnp.zeros((), dtype=jnp.float32)
        total_weight = 0.0

        for entry in self.entries:
            # MC signal from trajectory replay
            S_mc = apply_waveform_jax(
                G, entry.trajectories, entry.dt_traj, dt_wf, entry.walker_weights
            ).astype(jnp.float32)   # (n_meas,)

            # Analytical model signal
            S_an = forward_fn(entry.theta_nominal, scheme).astype(jnp.float32)  # (n_meas,)

            # Normalised squared error (dimensionless)
            err = jnp.mean((S_mc - S_an) ** 2) / (jnp.mean(S_mc ** 2) + 1e-6)

            weighted_err_sum = weighted_err_sum + entry.biological_weight * err
            total_weight += entry.biological_weight

        if total_weight == 0.0:
            return jnp.zeros((), dtype=jnp.float32)

        return weighted_err_sum / total_weight
