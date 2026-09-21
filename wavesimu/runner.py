"""Exécution de la chaîne DualSPHysics.

Étapes (chacune activable séparément) :

1. **gencase** : ``GenCase <cas>_Def <out>/<cas> -save:all``
2. **solve**   : ``DualSPHysics <out>/<cas> <out> -dirdataout data -svres [-cpu|-gpu]``
3. **post**    : PartVTK (particules fluide), IsoSurface (surface libre),
   MeasureTool (élévation aux sondes), ComputeForces / FloatingInfo pour
   les structures.

Le mode ``dry_run`` écrit les fichiers d'entrée et affiche les commandes
sans les exécuter, ce qui permet de préparer un cas sur une machine sans
DualSPHysics et de le lancer ailleurs (script ``run.sh`` généré).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from .casegen import CaseGenerator, gauge_points_file, structure_mk
from .config import Case, save_case
from .tools import Toolchain, platform_tag

#: Noms par défaut utilisés dans les commandes quand l'outil n'est pas résolu
#: (mode dry-run) : ``$DSPH_BIN/<nom>`` est alors substitué dans ``run.sh``.
DEFAULT_TOOL_NAMES = {
    "gencase": "GenCase_{plat}64",
    "dualsphysics_cpu": "DualSPHysics5.4CPU_{plat}64",
    "dualsphysics_gpu": "DualSPHysics5.4_{plat}64",
    "partvtk": "PartVTK_{plat}64",
    "measuretool": "MeasureTool_{plat}64",
    "isosurface": "IsoSurface_{plat}64",
    "computeforces": "ComputeForces_{plat}64",
    "floatinginfo": "FloatingInfo_{plat}64",
}

Logger = Callable[[str], None]


class RunError(RuntimeError):
    """Échec d'une étape de la chaîne de calcul."""


@dataclass
class StepResult:
    name: str
    command: List[str]
    returncode: Optional[int] = None
    duration: float = 0.0
    log: Optional[str] = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.skipped or self.returncode == 0


@dataclass
class RunLayout:
    """Arborescence d'un répertoire de simulation."""

    root: Path
    name: str

    @property
    def case_def(self) -> Path:
        return self.root / f"{self.name}_Def.xml"

    @property
    def out(self) -> Path:
        return self.root / f"{self.name}_out"

    @property
    def case_prefix(self) -> Path:
        return self.out / self.name

    @property
    def data(self) -> Path:
        return self.out / "data"

    @property
    def particles(self) -> Path:
        return self.out / "particles"

    @property
    def surface(self) -> Path:
        return self.out / "surface"

    @property
    def measure(self) -> Path:
        return self.out / "measuretool"

    @property
    def forces(self) -> Path:
        return self.out / "forces"

    @property
    def gauges_file(self) -> Path:
        return self.root / f"{self.name}_gauges.txt"

    @property
    def run_out(self) -> Path:
        return self.out / "Run.out"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifest(self) -> Path:
        return self.root / "wavesimu_run.json"


@dataclass
class RunOptions:
    gpu: bool = False
    threads: int = 0  #: 0 = tous les coeurs
    dry_run: bool = False
    gencase: bool = True
    solve: bool = True
    post: bool = True
    partvtk: bool = True
    isosurface: bool = True
    measure: bool = True
    forces: bool = True
    extra_solver_args: Sequence[str] = field(default_factory=list)


class Runner:
    """Pilote la simulation d'un :class:`Case` dans un répertoire."""

    def __init__(
        self,
        case: Case,
        workdir: Union[str, Path],
        toolchain: Optional[Toolchain] = None,
        options: Optional[RunOptions] = None,
        log: Optional[Logger] = None,
    ):
        self.case = case.validate()
        self.layout = RunLayout(Path(workdir).resolve(), case.name)
        self.toolchain = toolchain if toolchain is not None else Toolchain.discover()
        self.options = options or RunOptions()
        self.log: Logger = log or (lambda msg: print(msg, file=sys.stderr))
        self.results: List[StepResult] = []

    # --------------------------------------------------------------- setup
    def prepare(self) -> Path:
        """Écrit le XML, le fichier de sondes, la config et un script ``run.sh``."""
        lay = self.layout
        lay.root.mkdir(parents=True, exist_ok=True)
        lay.logs.mkdir(exist_ok=True)
        CaseGenerator(self.case).write(lay.root)
        if self.case.gauges:
            gauge_points_file(self.case, lay.gauges_file)
        save_case(self.case, lay.root / f"{self.case.name}.yaml")
        self._write_script()
        self.log(f"[prepare] cas écrit dans {lay.root} ({lay.case_def.name})")
        return lay.case_def

    # ------------------------------------------------------------ commands
    def _tool(self, name: str) -> str:
        p = self.toolchain.get(name)
        if p is None:
            if self.options.dry_run:
                return "$DSPH_BIN/" + DEFAULT_TOOL_NAMES[name].format(plat=platform_tag())
            self.toolchain.require(name)
        return str(p)

    def cmd_gencase(self) -> List[str]:
        lay = self.layout
        return [self._tool("gencase"), str(lay.case_def.with_suffix("")), str(lay.case_prefix), "-save:all"]

    def cmd_solve(self) -> List[str]:
        lay = self.layout
        o = self.options
        tool = "dualsphysics_gpu" if o.gpu else "dualsphysics_cpu"
        cmd = [self._tool(tool), str(lay.case_prefix), str(lay.out), "-dirdataout", "data", "-svres"]
        cmd.append("-gpu" if o.gpu else "-cpu")
        if not o.gpu and o.threads > 0:
            cmd.append(f"-ompthreads:{o.threads}")
        cmd += list(o.extra_solver_args)
        return cmd

    def cmd_partvtk(self) -> List[str]:
        lay = self.layout
        return [self._tool("partvtk"), "-dirin", str(lay.data), "-savevtk", str(lay.particles / "PartFluid"), "-onlytype:-all,+fluid", "-vars:+idp,+vel,+rhop,+press"]

    def cmd_isosurface(self) -> List[str]:
        lay = self.layout
        return [self._tool("isosurface"), "-dirin", str(lay.data), "-saveiso", str(lay.surface / "Surface")]

    def cmd_measuretool(self) -> List[str]:
        lay = self.layout
        return [
            self._tool("measuretool"),
            "-dirin", str(lay.data),
            "-points", str(lay.gauges_file),
            "-onlytype:-all,+fluid",
            "-elevation",
            "-savecsv", str(lay.measure / "Elevation"),
        ]

    def cmd_forces(self, index: int) -> List[str]:
        lay = self.layout
        s = self.case.structures[index]
        mk = structure_mk(self.case, index)
        if s.kind == "floating_box":
            return [self._tool("floatinginfo"), "-dirin", str(lay.data), f"-onlymk:{mk}", "-savemotion", "-savedata", str(lay.forces / f"Floating_{s.name}")]
        return [self._tool("computeforces"), "-dirin", str(lay.data), f"-onlymk:{mk}", "-viscoart:0.1", "-savecsv", str(lay.forces / f"Forces_{s.name}")]

    def plan(self) -> List[StepResult]:
        """Liste des étapes (nom + commande) sans les exécuter."""
        o = self.options
        steps: List[StepResult] = []
        if o.gencase:
            steps.append(StepResult("gencase", self.cmd_gencase()))
        if o.solve:
            steps.append(StepResult("solve", self.cmd_solve()))
        if o.post:
            if o.partvtk:
                steps.append(StepResult("partvtk", self.cmd_partvtk()))
            if o.isosurface:
                steps.append(StepResult("isosurface", self.cmd_isosurface()))
            if o.measure and self.case.gauges:
                steps.append(StepResult("measuretool", self.cmd_measuretool()))
            if o.forces:
                for i, s in enumerate(self.case.structures):
                    if s.compute_forces:
                        steps.append(StepResult(f"forces_{s.name}", self.cmd_forces(i)))
        return steps

    # ------------------------------------------------------------- execute
    def _run_step(self, step: StepResult) -> StepResult:
        lay = self.layout
        cmdline = " ".join(shlex.quote(c) for c in step.command)
        if self.options.dry_run:
            self.log(f"[{step.name}] (dry-run) {cmdline}")
            step.skipped = True
            return step
        for d in (lay.out, lay.particles, lay.surface, lay.measure, lay.forces, lay.logs):
            d.mkdir(parents=True, exist_ok=True)
        logfile = lay.logs / f"{step.name}.log"
        self.log(f"[{step.name}] {cmdline}")
        t0 = time.time()
        with open(logfile, "w", encoding="utf-8") as fh:
            fh.write(f"$ {cmdline}\n\n")
            fh.flush()
            proc = subprocess.run(step.command, stdout=fh, stderr=subprocess.STDOUT, env=self.toolchain.environment(), cwd=str(lay.root))
        step.returncode = proc.returncode
        step.duration = time.time() - t0
        step.log = str(logfile)
        if proc.returncode != 0:
            tail = _tail(logfile)
            raise RunError(f"étape '{step.name}' échouée (code {proc.returncode}). Journal : {logfile}\n{tail}")
        self.log(f"[{step.name}] terminé en {step.duration:.1f} s")
        return step

    def run(self) -> List[StepResult]:
        """Prépare puis exécute toutes les étapes demandées."""
        self.prepare()
        self.results = []
        try:
            for step in self.plan():
                self.results.append(self._run_step(step))
        finally:
            self._write_manifest()
        return self.results

    # -------------------------------------------------------------- outputs
    def _write_script(self) -> Path:
        """Script shell reproduisant la chaîne (utile pour un cluster)."""
        lay = self.layout
        lines = ["#!/usr/bin/env bash", "# Généré par wavesimu — chaîne DualSPHysics", "set -euo pipefail",
                 'cd "$(dirname "$0")"',
                 'DSPH_BIN="${DSPH_BIN:-${DUALSPHYSICS_HOME:-.}/bin/linux}"',
                 'export LD_LIBRARY_PATH="$DSPH_BIN:${LD_LIBRARY_PATH:-}"', ""]
        for step in self.plan():
            cmd = [self._relative(c) for c in step.command]
            lines.append(f"echo '== {step.name} =='")
            lines.append(" ".join(_quote(c) for c in cmd))
            lines.append("")
        script = lay.root / "run.sh"
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            script.chmod(0o755)
        except OSError:
            pass
        return script

    def _relative(self, arg: str) -> str:
        root = self.layout.root.resolve()
        try:
            p = Path(arg)
            if p.is_absolute():
                return str(p.resolve().relative_to(root))
        except (ValueError, OSError):
            pass
        return arg

    def _write_manifest(self) -> None:
        lay = self.layout
        data = {
            "case": self.case.name,
            "workdir": str(lay.root),
            "options": asdict(self.options),
            "toolchain": {k: (str(v) if v else None) for k, v in self.toolchain.tools.items()},
            "steps": [asdict(r) for r in self.results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        lay.manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _quote(arg: str) -> str:
    """Quote pour le shell, en laissant la variable ``$DSPH_BIN`` s'étendre."""
    if arg.startswith("$DSPH_BIN/"):
        return '"$DSPH_BIN"/' + shlex.quote(arg[len("$DSPH_BIN/"):])
    return shlex.quote(arg)


def _tail(path: Path, n: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def run_case(case: Case, workdir: Union[str, Path], **kwargs) -> List[StepResult]:
    """Raccourci : ``Runner(case, workdir, options=RunOptions(**kwargs)).run()``."""
    opts = RunOptions(**kwargs)
    return Runner(case, workdir, options=opts).run()


__all__ = ["RunError", "RunLayout", "RunOptions", "Runner", "StepResult", "run_case"]
