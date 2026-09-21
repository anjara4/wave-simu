import json
import os
from pathlib import Path

import pytest

from wavesimu.cli import main
from wavesimu.config import save_case
from wavesimu.runner import RunError, RunOptions, Runner
from wavesimu.templates import get_template
from wavesimu.tools import Toolchain, find_tool


def test_toolchain_discovery_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DUALSPHYSICS_HOME", str(tmp_path / "nowhere"))
    monkeypatch.setenv("PATH", str(tmp_path))
    tc = Toolchain.discover()
    assert not tc.is_complete()
    assert "introuvable" in tc.report()
    with pytest.raises(FileNotFoundError):
        tc.require("gencase")


def test_toolchain_discovery_fake(fake_dsph):
    tc = Toolchain.discover()
    assert tc.is_complete()
    assert tc.get("gencase").name == "GenCase_linux64"
    assert "CPU" in tc.get("dualsphysics_cpu").name
    assert "CPU" not in tc.get("dualsphysics_gpu").name
    assert str(fake_dsph / "bin" / "linux") in tc.environment()["LD_LIBRARY_PATH"]


def test_tool_override_env(fake_dsph, tmp_path, monkeypatch):
    alt = tmp_path / "MyGenCase"
    alt.write_text("#!/bin/sh\n")
    monkeypatch.setenv("DSPH_GENCASE", str(alt))
    assert find_tool("gencase") == alt
    with pytest.raises(KeyError):
        find_tool("unknown")


def test_dry_run_writes_inputs(tmp_path, regular_case, monkeypatch):
    monkeypatch.setenv("DUALSPHYSICS_HOME", str(tmp_path / "none"))
    r = Runner(regular_case, tmp_path / "run", options=RunOptions(dry_run=True), log=lambda m: None)
    results = r.run()
    assert all(s.skipped for s in results)
    assert [s.name for s in results] == ["gencase", "solve", "partvtk", "isosurface", "measuretool"]
    lay = r.layout
    assert lay.case_def.exists() and lay.gauges_file.exists() and (lay.root / "run.sh").exists()
    script = (lay.root / "run.sh").read_text()
    assert '"$DSPH_BIN"/GenCase_linux64 flume_regular_Def flume_regular_out/flume_regular -save:all' in script
    manifest = json.loads(lay.manifest.read_text())
    assert manifest["case"] == "flume_regular"


def test_full_pipeline_with_fake_binaries(fake_dsph, tmp_path):
    case = get_template("structure")
    r = Runner(case, tmp_path / "run", options=RunOptions(threads=2), log=lambda m: None)
    results = r.run()
    names = [s.name for s in results]
    assert names == ["gencase", "solve", "partvtk", "isosurface", "measuretool", "forces_reef"]
    assert all(s.ok and not s.skipped for s in results)
    assert r.layout.run_out.exists()
    assert "-ompthreads:2" in Path(results[1].log).read_text()
    assert (r.layout.measure / "Elevation_Elevation.csv").exists()


def test_gpu_command(fake_dsph, tmp_path, regular_case):
    r = Runner(regular_case, tmp_path / "run", options=RunOptions(gpu=True, dry_run=True), log=lambda m: None)
    cmd = r.cmd_solve()
    assert cmd[-1] == "-gpu" and "CPU" not in Path(cmd[0]).name


def test_failed_step_raises(fake_dsph, tmp_path, regular_case):
    bad = fake_dsph / "bin" / "linux" / "GenCase_linux64"
    bad.write_text("#!/usr/bin/env bash\necho boom\nexit 7\n")
    r = Runner(regular_case, tmp_path / "run", log=lambda m: None)
    with pytest.raises(RunError, match="code 7"):
        r.run()
    assert r.layout.manifest.exists()


# ------------------------------------------------------------------ CLI


def test_cli_init_check_theory(tmp_path, capsys):
    out = tmp_path / "c.yaml"
    assert main(["init", str(out), "-t", "irregular", "-n", "mycase"]) == 0
    assert out.exists()
    assert main(["init", str(out)]) == 1  # existe déjà
    assert main(["init", str(out), "--force"]) == 0
    assert main(["init", "--list"]) == 0
    assert main(["check", str(out)]) == 0
    assert main(["check", str(out), "--json"]) == 0
    assert main(["theory", "-H", "0.1", "-T", "1.5", "-d", "0.5"]) == 0
    assert main(["theory", "-H", "0.1", "-T", "1.5", "-d", "0.5", "--json"]) == 0
    assert "wavelength" in capsys.readouterr().out


def test_cli_check_warnings(tmp_path, capsys):
    c = get_template("regular")
    c.dp = 0.05  # 2 particules par hauteur de vague
    c.time.end = 3.0
    p = save_case(c, tmp_path / "coarse.yaml")
    assert main(["check", str(p), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert any("particules par hauteur" in w for w in rep["warnings"])
    assert any("durée courte" in w for w in rep["warnings"])


def test_cli_invalid_case(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\ndp: -1\n")
    assert main(["check", str(p)]) == 1
    assert main(["run", str(p), "-o", str(tmp_path)]) == 1


def test_cli_run_without_binaries(tmp_path, monkeypatch):
    monkeypatch.setenv("DUALSPHYSICS_HOME", str(tmp_path / "none"))
    monkeypatch.setenv("PATH", str(tmp_path))
    p = save_case(get_template("regular"), tmp_path / "c.yaml")
    assert main(["run", str(p), "-o", str(tmp_path / "runs")]) == 2
    assert main(["run", str(p), "-o", str(tmp_path / "runs"), "--dry-run"]) == 0
    assert (tmp_path / "runs" / "flume_regular_Def.xml").exists()
    assert main(["gencase", str(p), "-o", str(tmp_path / "g"), "--layout"]) == 0
    assert (tmp_path / "g" / "flume_regular_layout.png").exists()
    assert main(["doctor"]) == 1


def test_cli_full_run_and_analyze(fake_dsph, tmp_path, capsys):
    p = save_case(get_template("regular"), tmp_path / "c.yaml")
    wd = tmp_path / "runs"
    assert main(["run", str(p), "-o", str(wd)]) == 0
    assert main(["doctor"]) == 0
    capsys.readouterr()
    assert main(["analyze", str(wd), "--json", "--save", "--plot"]) == 0
    rep = json.loads(capsys.readouterr().out.split("figures dans")[0])
    assert rep["run"]["finished"] and rep["run"]["runtime_s"] == pytest.approx(82.5)
    names = [g["name"] for g in rep["gauges"]]
    assert names == ["WG1", "WG2", "WG3", "WG4"]
    for g in rep["gauges"]:
        # houle incidente 0.10 m + réfléchie 0.02 m : enveloppe locale dans [0.08, 0.12]
        assert 0.078 <= g["zc_h_mean"] <= 0.122
        assert g["zc_t_mean"] == pytest.approx(1.5, rel=0.02)
    assert rep["reflection"]["kr"] == pytest.approx(0.2, abs=0.02)
    assert (wd / "analysis.json").exists()
    assert (wd / "figures" / "elevations.png").exists()
    assert (wd / "figures" / "spectra.png").exists()
    assert (wd / "figures" / "heights.png").exists()
    # sortie texte + fenêtre explicite
    assert main(["analyze", str(wd), "--t-start", "8", "--t-end", "15"]) == 0
    out = capsys.readouterr().out
    assert "réflexion" in out and "WG1" in out
    # post seul
    assert main(["post", str(wd)]) == 0
    assert main(["post", str(wd), "--dry-run"]) == 0


def test_cli_analyze_without_results(tmp_path):
    p = save_case(get_template("regular"), tmp_path / "c.yaml")
    assert main(["analyze", str(tmp_path)]) == 1
    assert main(["analyze", str(tmp_path / "empty")]) == 1
