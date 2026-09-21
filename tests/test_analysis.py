import math

import numpy as np
import pytest

from wavesimu import analysis as A
from wavesimu.theory import wavenumber


def test_zero_crossing_regular(synthetic_series):
    t, eta, H, T = synthetic_series
    zc = A.zero_crossing(t, eta)
    assert zc.n_waves == 39
    assert zc.h_mean == pytest.approx(H, rel=1e-3)
    assert zc.h_13 == pytest.approx(H, rel=1e-3)
    assert zc.t_mean == pytest.approx(T, rel=1e-3)
    assert zc.crest_max == pytest.approx(H / 2, rel=1e-3)


def test_zero_crossing_removes_mean(synthetic_series):
    t, eta, H, T = synthetic_series
    assert A.zero_crossing(t, eta + 0.5).h_mean == pytest.approx(H, rel=1e-3)


def test_zero_crossing_short_series():
    assert A.zero_crossing([0, 1], [0, 0]).n_waves == 0
    assert A.zero_crossing(np.arange(3), np.array([-1, 1, -1])).n_waves == 0


def test_spectrum_regular(synthetic_series):
    t, eta, H, T = synthetic_series
    sp = A.spectrum(t, eta)
    # pour une houle monochromatique Hm0 = sqrt(2) H et Tp = T
    assert sp.hm0 == pytest.approx(math.sqrt(2) * H, rel=0.02)
    assert sp.tp == pytest.approx(T, rel=0.05)
    assert sp.m0 == pytest.approx(H * H / 8, rel=0.02)


def test_spectrum_irregular_hm0():
    rng = np.random.default_rng(0)
    t = np.arange(0, 400, 0.05)
    eta = np.zeros_like(t)
    freqs = np.linspace(0.4, 1.4, 60)
    amps = 0.01 * np.exp(-((freqs - 0.7) / 0.15) ** 2)
    for f, a in zip(freqs, amps):
        eta += a * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    hm0 = 4 * math.sqrt(np.sum(amps**2) / 2)
    sp = A.spectrum(t, eta)
    assert sp.hm0 == pytest.approx(hm0, rel=0.05)
    assert 1.2 < sp.tp < 1.7


def test_reflection_goda_recovers_known_kr():
    H, T, d = 0.1, 1.5, 0.5
    t = np.arange(0, 60, 0.02)
    k = wavenumber(T, d)
    dl = 0.3 * math.pi / k
    w = 2 * math.pi / T

    def eta(x, kr, phase=0.7):
        return 0.5 * H * np.cos(k * x - w * t) + 0.5 * kr * H * np.cos(k * x + w * t + phase)

    for kr in (0.0, 0.3, 0.8):
        r = A.reflection_goda(t, eta(2.0, kr), eta(2.0 + dl, kr), dl, d)
        assert r.kr == pytest.approx(kr, abs=2e-3)
        assert r.h_incident == pytest.approx(H, rel=1e-3)
    with pytest.raises(ValueError):
        A.reflection_goda(t, eta(1, 0), eta(1, 0), 0, d)


def test_reflection_short_record_with_period_truncation():
    H, T, d = 0.1, 1.5, 0.5
    t = np.arange(0, 8.3, 0.02)  # 5,5 périodes : fuites spectrales sans troncature
    k = wavenumber(T, d)
    dl = 0.3 * math.pi / k
    w = 2 * math.pi / T

    def eta(x):
        return 0.5 * H * np.cos(k * x - w * t) + 0.5 * 0.2 * H * np.cos(k * x + w * t + 0.7)

    r = A.reflection_goda(t, eta(2.0), eta(2.0 + dl), dl, d, period=T, fmin=0.5 / T, fmax=3.0 / T)
    assert r.kr == pytest.approx(0.2, abs=0.02)


def test_reflection_no_valid_frequency():
    t = np.arange(0, 10, 0.02)
    r = A.reflection_goda(t, np.sin(t), np.sin(t), 100.0, 0.5)
    assert math.isnan(r.kr) and r.n_freq == 0


def test_analyze_gauge_and_compare(synthetic_series):
    t, eta, H, T = synthetic_series
    rep = A.analyze_gauge("WG1", t, eta, 10.0, 50.0)
    assert rep.t_start >= 10.0 and rep.t_end <= 50.0
    d = rep.as_dict()
    assert d["zc_h_mean"] == pytest.approx(H, rel=1e-3)
    cmp = A.compare_to_target(rep, H, T)
    assert cmp["height_ratio"] == pytest.approx(1.0, rel=1e-3)
    assert cmp["period_ratio"] == pytest.approx(1.0, rel=1e-3)
    cmp_irr = A.compare_to_target(rep, H, T, irregular=True)
    assert cmp_irr["height_measured"] == pytest.approx(math.sqrt(2) * H, rel=0.02)
    with pytest.raises(ValueError):
        A.analyze_gauge("x", t, eta, 100, 200)


def test_suggested_window():
    t0, t1 = A.suggested_window(30.0, 1.5, 2.8, 4.0, 1.88, 1.0)
    assert t1 == 30.0
    assert 4.0 / 1.88 < t0 < 15.0
    t0, _ = A.suggested_window(5.0, 1.5, 2.8, 4.0, 1.88, 1.0)
    assert t0 == 2.5


def test_window():
    t = np.arange(10.0)
    tw, ew = A.window(t, t * 2, 2, 5)
    assert list(tw) == [2, 3, 4, 5] and list(ew) == [4, 6, 8, 10]
