import math

import pytest

from wavesimu import theory as th


def test_dispersion_deep_and_shallow_limits():
    # eau profonde : L -> g T^2 / 2pi
    assert th.wavelength(1.0, 50.0) == pytest.approx(th.deep_water_wavelength(1.0), rel=1e-6)
    # eau peu profonde : c -> sqrt(g h)
    assert th.celerity(30.0, 0.2) == pytest.approx(math.sqrt(9.81 * 0.2), rel=2e-3)


def test_dispersion_relation_satisfied():
    T, h = 1.5, 0.5
    k = th.wavenumber(T, h)
    w = 2 * math.pi / T
    assert w * w == pytest.approx(9.81 * k * math.tanh(k * h), rel=1e-10)


def test_group_velocity_bounds():
    c = th.celerity(1.5, 0.5)
    cg = th.group_velocity(1.5, 0.5)
    assert 0.5 * c < cg < c


def test_piston_transfer_limits():
    # eau peu profonde : H/S -> kh ; eau profonde : H/S -> 2
    assert th.piston_transfer(0.01, 1.0) == pytest.approx(0.01, rel=1e-3)
    assert th.piston_transfer(10.0, 1.0) == pytest.approx(2.0, rel=1e-6)


def test_flap_transfer_deep_limit():
    # eau profonde : H/S -> 2 (1 - 1/kh) pour un volet articulé au fond
    assert th.flap_transfer(10.0, 1.0) == pytest.approx(2.0 * (1 - 0.1), rel=1e-4)
    assert th.flap_transfer(200.0, 1.0) == pytest.approx(2.0, rel=6e-3)


def test_strokes_consistent():
    H, T, h = 0.1, 1.5, 0.5
    k = th.wavenumber(T, h)
    assert th.piston_stroke(H, T, h) * th.piston_transfer(k, h) == pytest.approx(H)
    assert th.flap_stroke(H, T, h) > th.piston_stroke(H, T, h)


def test_regime_and_summary():
    assert th.regime(1.0, 5.0) == "deep"
    assert th.regime(20.0, 0.5) == "shallow"
    s = th.summarize(0.1, 1.5, 0.5)
    assert s.regime == "intermediate"
    assert s.steepness == pytest.approx(0.1 / s.wavelength)
    assert s.breaking_height > 0.1
    assert set(s.as_dict()) >= {"wavelength", "piston_stroke", "ursell"}


def test_invalid_inputs():
    with pytest.raises(ValueError):
        th.wavenumber(0, 1)


def test_estimates(regular_case):
    p = th.estimate_particles(regular_case)
    assert p["fluid"] == pytest.approx(12.0 * 0.5 / 0.01**2, rel=0.01)
    assert p["total"] > p["fluid"] > 0
    c = th.estimate_cost(regular_case)
    assert c["dt_estimate"] > 0 and c["steps_estimate"] > 1000
