import xml.etree.ElementTree as ET

import pytest

from wavesimu.casegen import CaseGenerator, gauge_points_file, generate_case_xml, structure_mk
from wavesimu.config import Beach, Case, Gauge, Structure, Waves
from wavesimu.templates import TEMPLATES, get_template


def _root(case):
    return ET.fromstring(generate_case_xml(case))


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_all_templates_produce_valid_xml(name):
    root = _root(get_template(name))
    assert root.tag == "case"
    assert root.find("casedef/constantsdef/gravity").get("z") == "-9.81"
    assert root.find("execution/parameters") is not None
    assert root.find("casedef/geometry/commands/mainlist/fillbox") is not None


def test_regular_piston(regular_case):
    root = _root(regular_case)
    p = root.find("execution/special/wavepaddles/piston")
    assert p is not None
    assert p.find("waveheight").get("value") == "0.1"
    assert p.find("waveperiod").get("value") == "1.5"
    assert p.find("depth").get("value") == "0.5"
    assert p.find("mkbound").get("value") == str(regular_case.wavemaker.mk)
    assert p.find("awas_zsurf/swl").get("value") == "0.5"
    assert root.find("casedef/motion/objreal").get("ref") == str(regular_case.wavemaker.mk)


def test_irregular_uses_spectrum_paddle():
    root = _root(get_template("irregular"))
    p = root.find("execution/special/wavepaddles/piston_spectrum")
    assert p is not None
    assert p.find("spectrum").get("value") == "jonswap"
    assert p.find("waves").get("value") == "128"
    assert root.find("execution/special/wavepaddles/piston") is None


def test_flap_paddle():
    root = _root(get_template("flap"))
    p = root.find("execution/special/wavepaddles/flap")
    assert p is not None
    assert p.find("flapaxis0") is not None and p.find("variabledraft") is not None


def test_gauges_and_damping(regular_case):
    root = _root(regular_case)
    swl = root.findall("execution/special/gauges/swl")
    assert [g.get("name") for g in swl] == [g.name for g in regular_case.gauges]
    assert swl[0].find("point0").get("x") == "2"
    dz = root.find("execution/special/damping/dampingzone")
    assert float(dz.find("limitmin").get("x")) == pytest.approx(8.5)
    assert float(dz.find("limitmax").get("x")) == pytest.approx(12.0)


def test_no_gauges_no_gauges_block():
    root = _root(Case(damping=__import__("wavesimu.config", fromlist=["Damping"]).Damping(enabled=False)))
    assert root.find("execution/special/gauges") is None
    assert root.find("execution/special/damping") is None


def test_2d_walls_have_no_left_face_and_bottom_extends_behind_paddle(regular_case):
    root = _root(regular_case)
    boxes = root.findall("casedef/geometry/commands/mainlist/drawbox")
    walls = boxes[0]
    assert walls.find("boxfill").text == "bottom | right"
    assert float(walls.find("point").get("x")) < -regular_case.wavemaker.thickness
    # en 2D les formes ont une épaisseur en y (GenCase coupe dans le plan y=0)
    assert float(walls.find("point").get("y")) == pytest.approx(-0.1)
    assert float(walls.find("size").get("y")) == pytest.approx(0.2)
    fill = root.find("casedef/geometry/commands/mainlist/fillbox")
    assert float(fill.find("size").get("y")) == pytest.approx(0.2)
    assert root.find("casedef/geometry/definition/pointmin").get("y") == "0"


def test_3d_walls():
    root = _root(get_template("tank3d"))
    walls = root.findall("casedef/geometry/commands/mainlist/drawbox")[0]
    assert "front" in walls.find("boxfill").text
    assert root.find("casedef/geometry/commands/mainlist/drawcylinder") is not None


def test_beach_prism_2d():
    c = Case()
    c.flume.beach = Beach(start=6.0, slope=0.1)
    root = _root(c)
    prism = root.find("casedef/geometry/commands/mainlist/drawprism")
    pts = prism.findall("point")
    assert len(pts) == 6
    assert float(pts[2].get("z")) == pytest.approx(0.4)
    assert float(pts[0].get("y")) == pytest.approx(-0.1)


def test_structures_mk_and_floatings():
    c = get_template("floating")
    assert structure_mk(c, 0) == 20
    root = _root(c)
    fl = root.find("casedef/floatings/floating")
    assert fl.get("mkbound") == "20"
    assert fl.get("rhopbody") == "500"
    c2 = Case(structures=[Structure(mk=33)])
    assert structure_mk(c2, 0) == 33


def test_parameters_mapping():
    c = Case()
    c.physics.kernel = "cubic"
    c.physics.step_algorithm = "verlet"
    c.physics.boundary = "mdbc"
    c.time.end = 7.5
    root = _root(c)
    params = {p.get("key"): p.get("value") for p in root.findall("execution/parameters/parameter")}
    assert params["Kernel"] == "1"
    assert params["StepAlgorithm"] == "1"
    assert params["Boundary"] == "2"
    assert params["TimeMax"] == "7.5"
    assert params["TimeOut"] == "0.05"


def test_write_and_gauge_points(tmp_path, regular_case):
    path = CaseGenerator(regular_case).write(tmp_path)
    assert path.name == "flume_regular_Def.xml"
    assert path.read_text().startswith('<?xml version="1.0"')
    gp = gauge_points_file(regular_case, tmp_path / "g.txt")
    lines = gp.read_text().splitlines()
    assert lines[0] == "POINTS"
    n_per = int(regular_case.flume.height / (0.5 * regular_case.dp)) + 1
    assert len(lines) == 1 + n_per * len(regular_case.gauges)
    assert lines[1].split()[0] == "2.000000"


def test_invalid_case_rejected_by_generator():
    with pytest.raises(Exception):
        generate_case_xml(Case(dp=-1))
