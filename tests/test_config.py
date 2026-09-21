import pytest

from wavesimu.config import Case, ConfigError, Gauge, case_from_dict, load_case, resolve_damping, save_case


def test_default_case_is_valid():
    Case().validate()


def test_roundtrip_yaml(tmp_path, regular_case):
    p = save_case(regular_case, tmp_path / "c.yaml")
    loaded = load_case(p)
    assert loaded.name == regular_case.name
    assert [g.name for g in loaded.gauges] == [g.name for g in regular_case.gauges]
    assert loaded.wavemaker.waves.height == regular_case.wavemaker.waves.height
    assert loaded.wavemaker.awas.enabled is True


def test_roundtrip_json(tmp_path, regular_case):
    p = save_case(regular_case, tmp_path / "c.json")
    assert load_case(p).flume.length == regular_case.flume.length


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="clé inconnue"):
        case_from_dict({"name": "x", "flume": {"lenght": 3}})


@pytest.mark.parametrize(
    "patch, msg",
    [
        ({"dp": 0}, "dp"),
        ({"dimension": "4d"}, "dimension"),
        ({"flume": {"depth": 2, "height": 1}}, "flume.depth"),
        ({"wavemaker": {"kind": "plunger"}}, "wavemaker.kind"),
        ({"wavemaker": {"kind": "flap", "awas": {"enabled": True}}}, "AWAS"),
        ({"wavemaker": {"waves": {"height": 0.8}}}, "profondeur"),
        ({"gauges": [{"name": "A", "x": 1}, {"name": "A", "x": 2}]}, "double"),
        ({"gauges": [{"name": "A", "x": 99}]}, "hors du canal"),
        ({"dimension": "3d"}, "flume.width"),
        ({"structures": [{"kind": "cylinder"}]}, "2D"),
        ({"damping": {"start": 9, "end": 3}}, "damping"),
    ],
)
def test_validation_errors(patch, msg):
    with pytest.raises(ConfigError, match=msg):
        case_from_dict({"name": "t", **patch})


def test_missing_required_field():
    with pytest.raises(ConfigError):
        case_from_dict({"gauges": [{"x": 1.0}]})


def test_nested_lists_built_as_dataclasses():
    c = case_from_dict({"gauges": [{"name": "G", "x": 1.0}], "flume": {"beach": {"start": 5.0}}})
    assert isinstance(c.gauges[0], Gauge)
    assert c.flume.beach.height_at(7.0) == pytest.approx(0.2)


def test_resolve_damping_defaults():
    c = Case()
    d = resolve_damping(c, wavelength=2.0)
    assert d["end"] == c.flume.length
    assert d["start"] == pytest.approx(c.flume.length - 3.0)
    c.damping.enabled = False
    assert resolve_damping(c, 2.0) is None
    c.damping.enabled = True
    c.damping.start, c.damping.end = 6.0, 9.0
    assert resolve_damping(c, 2.0) == {"start": 6.0, "end": 9.0}


def test_load_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="introuvable"):
        load_case(tmp_path / "nope.yaml")
