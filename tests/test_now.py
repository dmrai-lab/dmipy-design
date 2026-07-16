"""NOW design oracle — validity of the designed gradient waveforms (LTE/PTE/STE/OGSE)."""
import pytest

from dmipy_design.optimizers import design_waveform_now, NowDesign


@pytest.mark.parametrize("name,b_delta", [("LTE", 1.0), ("PTE", -0.5), ("STE", 0.0)])
def test_now_designs_are_feasible(name, b_delta):
    d = design_waveform_now(b_delta, G_max=0.08, slew_rate_max=200.0, TE=0.08,
                            n_t=72, n_restarts=6, seed=0)
    assert isinstance(d, NowDesign)
    assert d.feasible, f"{name} design not feasible"
    assert d.b_value > 0
    assert d.refocus_residual < 1e-2                 # q(TE)=0 (spin-echo refocus)
    assert d.max_amplitude <= 0.08 * 1.02            # amplitude box
    assert d.max_slew <= 200.0 * 1.02                # slew limit
    assert d.m1_index < 5e-2 and d.m2_index < 5e-2   # M1/M2 nulled (default on)
    if d.n_axes >= 2:
        assert d.shape_residual < 5e-2               # requested b-tensor shape achieved


def test_ogse_spectral_freq_drives_encoding_frequency():
    """spectral_freq pins the RMS encoding frequency (OGSE-like), reported in spectral_rms."""
    d = design_waveform_now(1.0, TE=0.08, n_t=128, spectral_freq=80.0,
                            n_restarts=6, seed=0)
    assert d.feasible
    assert abs(d.spectral_rms - 80.0) / 80.0 < 0.1


def test_no_spectral_constraint_is_pgse_like_low_frequency():
    d = design_waveform_now(1.0, TE=0.08, n_t=72, n_restarts=3, seed=0)
    assert d.feasible
    assert d.spectral_rms < 40.0                     # a single-lobe PGSE-like waveform
