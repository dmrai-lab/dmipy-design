"""NOW design oracle — validity of the designed gradient waveforms (LTE/PTE/STE/OGSE)."""
import numpy as np
import pytest

from dmipy_design import ScannerLimits
from dmipy_design.optimizers import design_waveform_now, NowDesign

LIM = ScannerLimits.of("siemens_prisma")            # 80 mT/m, 200 T/m/s, from dmipy-sim's cited catalogue


@pytest.mark.parametrize("name,b_delta", [("LTE", 1.0), ("PTE", -0.5), ("STE", 0.0)])
def test_now_designs_are_feasible(name, b_delta):
    d = design_waveform_now(b_delta, limits=LIM, TE=0.08, n_t=72, n_restarts=6, seed=0)
    assert isinstance(d, NowDesign)
    assert d.feasible, f"{name} design not feasible"
    assert d.b_value > 0
    assert d.refocus_residual < 1e-2                 # q(TE)=0 (spin-echo refocus)
    assert d.max_amplitude <= LIM.G_max * 1.02       # amplitude box
    assert d.max_slew <= LIM.slew_max * 1.02         # slew limit
    assert d.m1_index < 5e-2 and d.m2_index < 5e-2   # M1/M2 nulled (default on)
    if d.n_axes >= 2:
        assert d.shape_residual < 5e-2               # requested b-tensor shape achieved


def test_ogse_spectral_freq_drives_encoding_frequency():
    """spectral_freq pins the RMS encoding frequency (OGSE-like), reported in spectral_rms."""
    d = design_waveform_now(1.0, limits=LIM, TE=0.08, n_t=128, spectral_freq=80.0,
                            n_restarts=6, seed=0)
    assert d.feasible
    assert abs(d.spectral_rms - 80.0) / 80.0 < 0.1


def test_no_spectral_constraint_is_pgse_like_low_frequency():
    d = design_waveform_now(1.0, limits=LIM, TE=0.08, n_t=72, n_restarts=3, seed=0)
    assert d.feasible
    assert d.spectral_rms < 40.0                     # a single-lobe PGSE-like waveform


def test_the_design_is_dmipy_sims_acquisition_object():
    """to_sequence() is the ScannerSequence dmipy-sim simulates: the physical gradient with the 180 at TE/2,
    the budget's finite pulses, the b the designer reported, refocused, and rescalable to a target b."""
    d = design_waveform_now(1.0, limits=LIM, TE=0.08, n_t=72, n_restarts=3, seed=0)
    seq = d.to_sequence()
    assert seq.timing is d.timing
    assert [e.duration_s for e in seq.rf] == [d.timing.t_excite, d.timing.t_refocus]
    assert seq.rf.refocus_time == pytest.approx(d.TE / 2.0, abs=seq.dt)
    np.testing.assert_allclose(seq.b(), [d.b_value], rtol=1e-6)
    assert seq.refocusing_residual < 1e-6
    np.testing.assert_allclose(np.asarray(seq.G)[0], d.G, rtol=1e-6, atol=1e-9)  # the physical gradient, as designed (float32)
    np.testing.assert_allclose(d.effective_G(), np.asarray(seq.G_eff)[0])
    np.testing.assert_allclose(d.to_sequence(b_target=2e8).b(), [2e8], rtol=1e-6)
    assert d.limits is LIM


def test_the_scanner_is_named_not_numbered():
    """The limits come from the catalogue by name, class or an explicit envelope; a designer carries no number."""
    a = design_waveform_now(1.0, limits="connectom", TE=0.05, n_t=64, n_restarts=2, seed=0)
    b = design_waveform_now(1.0, limits=(0.3, 200.0), TE=0.05, n_t=64, n_restarts=2, seed=0)
    assert a.limits.G_max == pytest.approx(0.3) and a.b_value > 0 and b.limits.kind == "envelope"
    with pytest.raises(TypeError):
        design_waveform_now(1.0, TE=0.05)                                        # limits is not optional
# --------------------------------------------------------------------------------------
# b_delta validation: an unrealisable shape must raise, never come back as another shape
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("b_delta", [0.75, 0.5, 0.25, -0.25, 0.9, -1.0, 2.0])
def test_unrealisable_b_delta_is_rejected(b_delta):
    """The shape constraint pins two or three b-tensor axes equal, so only LTE/STE/PTE exist.

    Before this check an intermediate b_delta returned an ISOTROPIC design carrying the requested value
    in NowDesign.b_delta with a shape residual of ~1e-30 -- self-consistent for the constraint actually
    applied, and silently not what was asked for.
    """
    with pytest.raises(ValueError, match="not realisable"):
        design_waveform_now(b_delta, limits=LIM, TE=0.06, n_t=48, n_restarts=1)


@pytest.mark.parametrize("b_delta", [1.0, 0.0, -0.5])
def test_supported_b_delta_is_accepted(b_delta):
    from dmipy_design.optimizers.now import _validate_b_delta
    assert _validate_b_delta(b_delta) == b_delta


def test_supported_shapes_are_actually_achieved():
    """Each supported b_delta comes back as that shape, measured from the design's own b-tensor."""
    import numpy as np
    from dmipy_design.optimizers.now import SUPPORTED_SHAPES

    for target, name, _rank in SUPPORTED_SHAPES:
        d = design_waveform_now(target, limits=LIM, TE=0.08, n_t=72, n_restarts=6, seed=0)
        G = np.asarray(d.effective_G())
        if G.ndim == 3:
            G = G[0]
        # effective_G() already carries the post-180 sign flip; applying the mask again would undo it
        # and hand back the physical gradient, whose b-tensor is a different shape entirely.
        q = np.cumsum(G, axis=0) * d.dt
        B = (q[:, :, None] * q[:, None, :]).sum(0) * d.dt
        lam = np.linalg.eigvalsh(B)
        tr = float(lam.sum())
        l1 = lam[np.argsort(-np.abs(lam - tr / 3.0))[0]]
        achieved = (l1 - (tr - l1) / 2.0) / tr
        assert abs(achieved - target) < 0.05, f"{name}: asked {target:+.2f}, got {achieved:+.3f}"


def test_min_te_does_not_swallow_the_shape_error():
    """min_te_for_b treats ValueError as 'TE too short' while growing its bracket, so the shape error
    has to be raised before that loop or it resurfaces as an unreachable-b_target failure."""
    from dmipy_design import min_te_for_b
    with pytest.raises(ValueError, match="not realisable"):
        min_te_for_b(1e9, 0.5, limits=LIM, n_seeds=1)


def test_stimulated_echo_rejects_unrealisable_shape():
    from dmipy_design import design_stimulated_echo
    with pytest.raises(ValueError, match="not realisable"):
        design_stimulated_echo(0.5, limits=LIM, TM=0.05, TE=0.06, n_t=48, n_restarts=1)
