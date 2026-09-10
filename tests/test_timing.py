"""The budget is dmipy-sim's SequenceTiming; the designer reads it on a grid -- the encoding windows, with
the derived (not free-knob) pre/post-180 asymmetry, and the stimulated echo's matched periods."""
import numpy as np
import pytest

from dmipy_design.optimizers import (SequenceTiming, DEFAULT_TIMING, encoding_mask, stimulated_echo_mask,
                                     encoding_spectrum)


def test_the_budget_is_dmipy_sims():
    from dmipy_sim.acquisition.timing import SequenceTiming as SimTiming
    assert SequenceTiming is SimTiming
    assert DEFAULT_TIMING.min_TE() > 0


def test_encoding_windows_are_asymmetric():
    """A real timing budget pins the encoding windows; an unequal lead-in vs readout-pre-echo makes the
    pre/post-180 windows asymmetric -- a derived consequence, not a knob -- with the gradient masked off in
    lead-in / 180 / readout."""
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3, readout_duration=30e-3, partial_fourier=0.75)
    assert abs(st.t_readout_pre_echo - 10e-3) < 1e-9          # 30ms·(0.25/0.75)
    TE, n_t = 0.080, 400
    mask, echo = encoding_mask(st, TE, n_t)
    on = mask[:, 0]
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    assert echo == round((TE / 2) / dt)                       # 180 at TE/2
    assert on[t < st.t_lead].sum() == 0                       # excitation lead-in off
    assert on[np.abs(t - TE / 2) <= st.t_refocus / 2].sum() == 0   # 180 off
    assert on[t > TE - st.t_readout_pre_echo].sum() == 0      # readout tail off
    pre, post = on[t < TE / 2].sum(), on[t > TE / 2].sum()
    assert pre > post * 1.1                                   # asymmetric, derived


def test_symmetric_is_the_vanilla_waveform():
    """symmetric=True mirrors the encoding windows about the 180 (equal pre/post durations) and dead-times the
    surplus of the longer window: the same TE and 180 position, less total encoding -- the cost of refusing
    the budget's natural asymmetry."""
    b = dict(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=10e-3)
    TE, n_t = 0.050, 300
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    on_a, e_a = encoding_mask(SequenceTiming(**b), TE, n_t)
    on_s, e_s = encoding_mask(SequenceTiming(**b), TE, n_t, symmetric=True)
    assert e_a == e_s == round((TE / 2) / dt)                 # 180 stays at TE/2
    pre_a, post_a = on_a[t < TE / 2, 0].sum(), on_a[t > TE / 2, 0].sum()
    pre_s, post_s = on_s[t < TE / 2, 0].sum(), on_s[t > TE / 2, 0].sum()
    assert pre_a > post_a * 1.1                               # default is asymmetric
    assert pre_s == post_s                                    # vanilla is symmetric
    assert post_s == post_a                                   # both equal the shorter window
    assert on_s.sum() < on_a.sum()                            # vanilla encodes less (dead time)


def test_min_te_guard():
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3, readout_duration=30e-3, partial_fourier=0.75)
    with pytest.raises(ValueError):                           # below min_TE the windows vanish
        encoding_mask(st, st.min_TE() - 1e-3, 200)


def test_stimulated_echo_periods_are_matched_around_the_mixing_time():
    st = SequenceTiming(t_excite=2e-3, t_refocus=4e-3, t_readout_pre_echo=2e-3)
    TM, TE, n_t = 50e-3, 0.12, 400
    on, recall, store = stimulated_echo_mask(st, TM, TE, n_t)
    on = on[:, 0]
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    assert 0 < store < recall < n_t and (recall - store) * dt == pytest.approx(TM, abs=dt)
    assert on[t < st.t_lead].sum() == 0 and on[-1] == 0.0                 # lead-in off, readout sample free
    inside = (t > store * dt - st.t_excite / 2 + dt) & (t < recall * dt + st.t_excite / 2 - dt)
    assert on[inside].sum() == 0                                            # store .. recall: the gradient is off
    pre, post = on[t < store * dt].sum(), on[t > recall * dt].sum()
    assert abs(pre - post) <= 1                                             # tau1 = tau3, to a sample
    assert (2 * store * dt + TM) == pytest.approx(TE, abs=2 * dt)           # TE = 2 t_store + TM
    with pytest.raises(ValueError, match="no room"):
        stimulated_echo_mask(st, TM, TM + 2 * st.t_excite, n_t)


def test_encoding_spectrum_reads_a_sequence():
    from dmipy_sim.sequences import ogse, pgse
    lo = pgse([[1, 0, 0]], 8e-3, 24e-3, bvalues=[1e9], n_t=400, slew_rate=np.inf)
    hi = ogse([[1, 0, 0]], 100.0, 20e-3, shape="cosine", bvalues=[1e9], n_t=400, slew_rate=np.inf)
    f_lo = encoding_spectrum(lo)[4]
    f_hi = encoding_spectrum(hi)[4]
    assert f_lo < 40.0 < f_hi and abs(f_hi - 100.0) / 100.0 < 0.15
