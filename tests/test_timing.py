"""SequenceTiming — encoding windows derived from a real timing budget, and the
derived (not free-knob) pre/post-180 asymmetry."""
import numpy as np
import pytest

from dmipy_design.optimizers import SequenceTiming


def test_sequence_timing_windows_are_asymmetric():
    """A real timing budget pins the encoding windows; an unequal lead-in vs
    readout-pre-echo makes the pre/post-180 windows asymmetric — a derived
    consequence, not a knob — with the gradient masked off in lead-in/180/readout."""
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3,
                                     readout_duration=30e-3, partial_fourier=0.75)
    assert abs(st.t_readout_pre_echo - 10e-3) < 1e-9          # 30ms·(0.25/0.75)
    TE, n_t = 0.080, 400
    mask, echo = st.masks(TE, n_t)
    on = mask[:, 0]
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    assert echo == round((TE / 2) / dt)                       # 180 at TE/2
    assert on[t < st.t_lead].sum() == 0                       # excitation lead-in off
    assert on[np.abs(t - TE / 2) <= st.t_refocus / 2].sum() == 0   # 180 off
    assert on[t > TE - st.t_readout_pre_echo].sum() == 0      # readout tail off
    pre, post = on[t < TE / 2].sum(), on[t > TE / 2].sum()
    assert pre > post * 1.1                                   # asymmetric, derived


def test_sequence_timing_symmetric_is_the_vanilla_waveform():
    """The VANILLA waveform: symmetric=True mirrors the encoding windows about the
    180 (equal pre/post durations) and dead-times the surplus of the longer window.
    Same TE and 180 position, but less total encoding -> the cost of refusing the
    budget's natural asymmetry (extra transverse time = T2 loss)."""
    b = dict(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=10e-3)
    TE, n_t = 0.050, 300
    dt = TE / (n_t - 1)
    t = np.arange(n_t) * dt
    on_a, e_a = SequenceTiming(**b).masks(TE, n_t)
    on_s, e_s = SequenceTiming(**b, symmetric=True).masks(TE, n_t)
    assert e_a == e_s == round((TE / 2) / dt)                 # 180 stays at TE/2
    pre_a, post_a = on_a[t < TE / 2, 0].sum(), on_a[t > TE / 2, 0].sum()
    pre_s, post_s = on_s[t < TE / 2, 0].sum(), on_s[t > TE / 2, 0].sum()
    assert pre_a > post_a * 1.1                               # default is asymmetric
    assert pre_s == post_s                                    # vanilla is symmetric
    assert post_s == post_a                                   # both equal the shorter window
    assert on_s.sum() < on_a.sum()                            # vanilla encodes less (dead time)


def test_sequence_timing_min_te_guard():
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3,
                                     readout_duration=30e-3, partial_fourier=0.75)
    with pytest.raises(ValueError):                           # below min_TE windows vanish
        st.masks(st.min_TE() - 1e-3, 200)
