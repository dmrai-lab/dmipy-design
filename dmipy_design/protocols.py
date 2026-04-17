"""
Protocol: a designed acquisition protocol, ready to convert to dmipy-core format.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Protocol:
    """A fully specified dMRI acquisition protocol.

    Attributes
    ----------
    bvalues : ndarray, shape (N,)       s/m²
    bvecs   : ndarray, shape (N, 3)     unit gradient vectors
    delta   : ndarray, shape (N,)       gradient pulse duration (s)
    Delta   : ndarray, shape (N,)       gradient separation (s)
    TE      : ndarray, shape (N,) or None
    b_delta : ndarray, shape (N,) or None   B-tensor shape (-0.5, 0, 1)
    crlb    : float or None             CRLB objective achieved
    objective : str or None            'A', 'D', or 'E'
    """
    bvalues: np.ndarray
    bvecs: np.ndarray
    delta: np.ndarray
    Delta: np.ndarray
    TE: np.ndarray | None = None
    b_delta: np.ndarray | None = None
    crlb: float | None = None
    objective: str | None = None

    def to_dmipy_scheme(self):
        """Convert to a dmipy-core ``PGSEAcquisitionScheme``.

        Returns
        -------
        scheme : PGSEAcquisitionScheme
            Ready to pass to ``MultiCompartmentModel.fit(scheme, data)``.
        """
        from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
        return acquisition_scheme_from_bvalues(
            self.bvalues,
            self.bvecs,
            self.delta,
            self.Delta,
            **({"TE": self.TE} if self.TE is not None else {}),
        )

    def save(self, path: str) -> None:
        """Save protocol to a compressed .npz file."""
        arrays = dict(
            bvalues=self.bvalues,
            bvecs=self.bvecs,
            delta=self.delta,
            Delta=self.Delta,
        )
        if self.TE is not None:
            arrays["TE"] = self.TE
        if self.b_delta is not None:
            arrays["b_delta"] = self.b_delta
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "Protocol":
        """Load protocol from a .npz file."""
        d = np.load(path)
        return cls(
            bvalues=d["bvalues"],
            bvecs=d["bvecs"],
            delta=d["delta"],
            Delta=d["Delta"],
            TE=d.get("TE"),
            b_delta=d.get("b_delta"),
        )
