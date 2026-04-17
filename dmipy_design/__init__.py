"""
dmipy-design: optimal acquisition design for diffusion MRI via JAX autodiff
CRLB and B-tensor encoding.
"""

from .acquisition_space import AcquisitionSpace
from .constraints import HardwareConstraints, TimeConstraints
from .objectives import a_optimal, d_optimal, e_optimal, parameter_selective_crlb
from .fim import compute_fim, compute_fim_averaged
from .jax_scheme_encoder import JaxScheme, encode_pgse
from .protocols import Protocol

__all__ = [
    "AcquisitionSpace",
    "HardwareConstraints",
    "TimeConstraints",
    "a_optimal",
    "d_optimal",
    "e_optimal",
    "parameter_selective_crlb",
    "compute_fim",
    "compute_fim_averaged",
    "JaxScheme",
    "encode_pgse",
    "Protocol",
]
