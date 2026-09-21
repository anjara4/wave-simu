import os
import stat
import textwrap
from pathlib import Path

import numpy as np
import pytest

from wavesimu.templates import get_template


@pytest.fixture
def regular_case():
    return get_template("regular")


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -e\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_dsph(tmp_path, monkeypatch):
    """Faux exécutables DualSPHysics qui imitent les sorties de la chaîne."""
    home = tmp_path / "DualSPHysics_v5.4"
    b = home / "bin" / "linux"
    _script(b / "GenCase_linux64", """
        echo "GenCase fake: $@"
        out="$2"; mkdir -p "$(dirname "$out")"; echo fake > "${out}.bi4"; echo fake > "${out}.xml"
    """)
    _script(b / "DualSPHysics5.4CPU_linux64", """
        echo "DualSPHysics fake: $@"
        out="$2"; mkdir -p "$out/data"
        {
          echo "Particles: 1234"
          echo "Part      PartTime      TotalSteps    Steps    Time/Sec   Finish time"
          for i in $(seq 0 40); do printf 'Part_%04d  %f  %d  %d  %f  x\\n' $i $(echo "$i*0.05" | bc -l) $((i*100)) 100 $((i*2)); done
          echo "Simulation Runtime: 82.5 sec."
        } > "$out/Run.out"
        touch "$out/data/Part_0000.bi4"
    """)
    _script(b / "DualSPHysics5.4_linux64", 'echo gpu "$@"')
    _script(b / "PartVTK_linux64", 'echo partvtk "$@"')
    _script(b / "IsoSurface_linux64", 'echo iso "$@"')
    _script(b / "ComputeForces_linux64", 'echo forces "$@"')
    _script(b / "FloatingInfo_linux64", 'echo floating "$@"')
    # MeasureTool : écrit un CSV d'élévation synthétique (houle H=0.1 T=1.5 + réflexion)
    py = tmp_path / "fake_measuretool.py"
    py.write_text(textwrap.dedent("""
        import sys, math
        args = sys.argv[1:]
        pts = args[args.index('-points') + 1]
        out = args[args.index('-savecsv') + 1]
        xs = []
        for line in open(pts):
            parts = line.split()
            if len(parts) == 3:
                x = float(parts[0])
                if x not in xs:
                    xs.append(x)
        import os
        os.makedirs(os.path.dirname(out), exist_ok=True)
        H, T, h = 0.1, 1.5, 0.5
        k = 2.2233
        w = 2 * math.pi / T
        with open(out + '_Elevation.csv', 'w') as f:
            f.write('Pos.x [m];' + ';'.join(str(x) for x in xs) + '\\n')
            f.write('Pos.y [m];' + ';'.join('0' for x in xs) + '\\n')
            f.write('Part;Time [s];' + ';'.join('Elevation_%d [m]' % i for i in range(len(xs))) + '\\n')
            n = 0
            t = 0.0
            while t <= 15.0 + 1e-9:
                vals = []
                for x in xs:
                    ramp = min(1.0, max(0.0, (t - x / 1.884) / T))
                    eta = ramp * (0.5 * H * math.cos(k * x - w * t) + 0.5 * 0.02 * math.cos(k * x + w * t))
                    vals.append('%.6f' % eta)
                f.write('%d;%.4f;%s\\n' % (n // 10, t, ';'.join(vals)))
                t += 0.02
                n += 1
    """))
    _script(b / "MeasureTool_linux64", f'python3 "{py}" "$@"')
    monkeypatch.setenv("DUALSPHYSICS_HOME", str(home))
    for k in list(os.environ):
        if k.startswith("DSPH_"):
            monkeypatch.delenv(k)
    return home


@pytest.fixture
def synthetic_series():
    t = np.arange(0, 60, 0.02)
    H, T = 0.1, 1.5
    return t, 0.5 * H * np.cos(2 * np.pi * t / T), H, T
