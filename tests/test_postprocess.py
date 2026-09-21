import textwrap

import numpy as np
import pytest

from wavesimu.postprocess import find_gauge_files, load_elevations, parse_run_out, read_gauge_csv, read_measuretool_csv


RUN_OUT = """
[Initialising simulation ...]
Particles: 12345 (fluid=10000, bound=2345)
CaseName: flume
Part      PartTime      TotalSteps    Steps    Time/Sec   Finish time
Part_0000        0.000000         0        0       0.00   21-09-2026 19:30:00
Part_0001        0.050000       133      133      10.86   21-09-2026 19:31:00
Part_0002        0.100000       270      137      21.50   21-09-2026 19:31:00
*** WARNING: some warning here
Simulation Runtime: 41.7 sec.
"""


def test_parse_run_out(tmp_path):
    p = tmp_path / "Run.out"
    p.write_text(RUN_OUT)
    ri = parse_run_out(p)
    assert ri.n_parts == 3
    assert ri.last_time == pytest.approx(0.1)
    assert ri.runtime == pytest.approx(41.7)
    assert ri.finished
    assert ri.particles == 12345
    assert ri.info["CaseName"] == "flume"
    assert len(ri.warnings) == 1


def test_parse_run_out_unfinished(tmp_path):
    p = tmp_path / "Run.out"
    p.write_text("Part_0000  0.0 0 0 0.0\nPart_0001  0.05 10 10 1.0\n")
    ri = parse_run_out(p)
    assert not ri.finished and ri.n_parts == 2


MT_CSV = textwrap.dedent("""\
    Pos.x [m];2;4
    Pos.y [m];0;0
    Pos.z [m];0;0
    Part;Time [s];Elevation_0 [m];Elevation_1 [m]
    0;0.0;0.001;0.002
    0;0.05;0.011;0.012
    1;0.10;0.021;0.022
    """)


def test_read_measuretool_csv(tmp_path):
    p = tmp_path / "Elevation_Elevation.csv"
    p.write_text(MT_CSV)
    es = read_measuretool_csv(p, ["WG1", "WG2"])
    assert es.names == ["WG1", "WG2"]
    assert es.time.tolist() == [0.0, 0.05, 0.10]
    assert es["WG2"].tolist() == pytest.approx([0.002, 0.012, 0.022])
    assert es.positions["WG2"] == (4.0, 0.0)
    es2 = read_measuretool_csv(p)
    assert es2.names == ["Elev_0", "Elev_1"]


def test_read_measuretool_comma_decimal(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("Part;Time [s];Elevation_0 [m]\n0;0,0;0,5\n0;0,1;0,6\n")
    es = read_measuretool_csv(p)
    assert es["Elev_0"].tolist() == pytest.approx([0.5, 0.6])


def test_read_gauge_csv_and_load(tmp_path):
    out = tmp_path / "c_out"
    out.mkdir()
    for name, base in (("WG1", 0.5), ("WG2", 0.5)):
        (out / f"GaugesSwl_{name}.csv").write_text(
            "Part;Time [s];Swl_z [m]\n0;0.0;%f\n0;0.1;%f\n0;0.2;%f\n" % (base, base + 0.01, base - 0.01)
        )
    files = find_gauge_files(out)
    assert len(files) == 2
    es = read_gauge_csv(files[0])
    assert es.names == ["WG1"]
    es = load_elevations(tmp_path, "c", swl=0.5)
    assert set(es.names) == {"WG1", "WG2"}
    assert es["WG1"].tolist() == pytest.approx([0.0, 0.01, -0.01])


def test_load_elevations_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_elevations(tmp_path, "nothing")


def test_load_prefers_measuretool(tmp_path):
    out = tmp_path / "c_out"
    (out / "measuretool").mkdir(parents=True)
    (out / "measuretool" / "Elevation_Elevation.csv").write_text(MT_CSV)
    (out / "GaugesSwl_WG9.csv").write_text("Part;Time [s];Swl_z [m]\n0;0;0.5\n")
    es = load_elevations(tmp_path, "c", ["A", "B"])
    assert es.names == ["A", "B"]
