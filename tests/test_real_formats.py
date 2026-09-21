"""Lecture des formats réels produits par DualSPHysics v5.4 (extraits dans tests/data)."""

from pathlib import Path

import numpy as np
import pytest

from wavesimu.postprocess import (
    find_gauge_files,
    load_elevations,
    parse_run_out,
    read_gauge_csv,
    read_measuretool_csv,
    relative_to_rest,
)

DATA = Path(__file__).parent / "data"


def test_parse_real_run_out():
    ri = parse_run_out(DATA / "Run.out")
    assert ri.finished
    assert ri.runtime == pytest.approx(37.257847)
    assert ri.particles == 3915
    assert ri.n_parts == 9
    assert ri.last_time == pytest.approx(4.000095)
    assert ri.parts[3, 2] == 1127  # TotalSteps avec séparateur de milliers
    assert ri.parts[-1, 4] == pytest.approx(9.69)  # Time/Sec
    assert ri.info["Steps of simulation"] == "22,679"


def test_read_real_gauge_csv():
    es = read_gauge_csv(DATA / "GaugesSWL_WG1.csv")
    assert es.names == ["WG1"]
    assert es.time[0] == 0.0 and es.time[1] == pytest.approx(0.050042)
    # colonne swlz (niveau absolu), pas swlx (position)
    assert es["WG1"][0] == pytest.approx(0.293779)
    assert np.all(es["WG1"] < 0.4)


def test_read_real_measuretool_csv():
    es = read_measuretool_csv(DATA / "Elevation_Elevation.csv", ["WG1", "WG2", "WG3"])
    assert es.names == ["WG1", "WG2", "WG3"]
    assert es.positions == {"WG1": (1.0, 0.0), "WG2": (1.5, 0.0), "WG3": (2.5, 0.0)}
    assert es["WG3"][0] == pytest.approx(0.293779)
    assert len(es.time) == 8


def test_relative_to_rest():
    es = read_measuretool_csv(DATA / "Elevation_Elevation.csv")
    rel = relative_to_rest(es, swl=0.3)
    for name in rel.names:
        assert rel[name][0] == 0.0
    # une série déjà centrée n'est pas modifiée
    already = relative_to_rest(rel, swl=0.3)
    assert np.allclose(already[rel.names[0]], rel[rel.names[0]])


def test_load_real_layout(tmp_path):
    out = tmp_path / "small_out"
    out.mkdir()
    for f in ("GaugesSWL_WG1.csv", "GaugesSWL_WG2.csv"):
        (out / f).write_text((DATA / f).read_text())
    (out / "GaugesSWL_AwasMkb10.csv").write_text((DATA / "GaugesSWL_WG1.csv").read_text())
    assert [p.name for p in find_gauge_files(out)] == ["GaugesSWL_WG1.csv", "GaugesSWL_WG2.csv"]
    es = load_elevations(tmp_path, "small", swl=0.3)
    assert set(es.names) == {"WG1", "WG2"}
    assert es["WG1"][0] == 0.0
    # MeasureTool prioritaire quand présent
    (out / "measuretool").mkdir()
    (out / "measuretool" / "Elevation_Elevation.csv").write_text((DATA / "Elevation_Elevation.csv").read_text())
    es = load_elevations(tmp_path, "small", ["A", "B", "C"], swl=0.3)
    assert es.names == ["A", "B", "C"] and es["A"][0] == 0.0
