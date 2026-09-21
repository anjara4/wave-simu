"""Localisation des exécutables DualSPHysics.

La racine de l'installation est cherchée, dans l'ordre :

1. l'argument ``home`` ;
2. la variable d'environnement ``DUALSPHYSICS_HOME`` ;
3. quelques emplacements usuels (``~/DualSPHysics*``, ``/opt/DualSPHysics*``).

Les binaires sont attendus dans ``<home>/bin/linux`` (ou ``bin/windows``)
avec les noms de la distribution officielle (``GenCase_linux64``,
``DualSPHysics5.4CPU_linux64``, ``MeasureTool_linux64``…). Un nom peut
aussi être fourni explicitement via ``DSPH_<TOOL>`` (ex. ``DSPH_GENCASE``).
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

#: Motifs de noms (glob) par outil, du plus spécifique au plus générique.
TOOL_PATTERNS: Dict[str, List[str]] = {
    "gencase": ["GenCase_{plat}64", "GenCase4_{plat}64", "GenCase_{plat}64.exe", "GenCase"],
    "dualsphysics_cpu": ["DualSPHysics5.*CPU_{plat}64", "DualSPHysics5.*CPU_{plat}64.exe", "DualSPHysics*CPU*"],
    "dualsphysics_gpu": ["DualSPHysics5.*_{plat}64", "DualSPHysics5.*_{plat}64.exe", "DualSPHysics5.*"],
    "partvtk": ["PartVTK_{plat}64", "PartVTK_{plat}64.exe", "PartVTK*"],
    "measuretool": ["MeasureTool_{plat}64", "MeasureTool_{plat}64.exe", "MeasureTool*"],
    "isosurface": ["IsoSurface_{plat}64", "IsoSurface_{plat}64.exe", "IsoSurface*"],
    "computeforces": ["ComputeForces_{plat}64", "ComputeForces_{plat}64.exe", "ComputeForces*"],
    "floatinginfo": ["FloatingInfo_{plat}64", "FloatingInfo_{plat}64.exe", "FloatingInfo*"],
    "boundaryvtk": ["BoundaryVTK_{plat}64", "BoundaryVTK_{plat}64.exe", "BoundaryVTK*"],
}

#: Outils indispensables pour lancer une simulation.
REQUIRED_TOOLS = ("gencase", "dualsphysics_cpu")


def platform_tag() -> str:
    return "win" if platform.system().lower().startswith("win") else "linux"


def default_homes() -> List[Path]:
    """Emplacements candidats pour l'installation DualSPHysics."""
    homes: List[Path] = []
    env = os.environ.get("DUALSPHYSICS_HOME")
    if env:
        homes.append(Path(env).expanduser())
    for pattern in ("~/DualSPHysics*", "/opt/DualSPHysics*", "/usr/local/DualSPHysics*", "C:/DualSPHysics*"):
        for match in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True):
            homes.append(Path(match))
    return homes


def bin_dirs(home: Optional[Path]) -> List[Path]:
    """Répertoires où chercher les binaires pour une racine donnée."""
    tag = "windows" if platform_tag() == "win" else "linux"
    dirs: List[Path] = []
    if home is not None:
        dirs += [home / "bin" / tag, home / "bin", home]
    return [d for d in dirs if d.is_dir()]


def _is_gpu_candidate(path: Path) -> bool:
    return "CPU" not in path.name.upper()


def find_tool(name: str, home: Optional[Path] = None, extra_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    """Cherche l'exécutable ``name`` (clé de :data:`TOOL_PATTERNS`)."""
    if name not in TOOL_PATTERNS:
        raise KeyError(f"outil inconnu : {name}")
    override = os.environ.get(f"DSPH_{name.upper()}")
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p
        found = shutil.which(override)
        if found:
            return Path(found)
    plat = platform_tag()
    dirs = list(extra_dirs or [])
    homes = [home] if home is not None else default_homes()
    for h in homes:
        dirs += bin_dirs(h)
    for d in dirs:
        for pattern in TOOL_PATTERNS[name]:
            for match in sorted(glob.glob(str(d / pattern.format(plat=plat)))):
                m = Path(match)
                if not m.is_file():
                    continue
                if name == "dualsphysics_gpu" and not _is_gpu_candidate(m):
                    continue
                if name == "dualsphysics_cpu" and "CPU" not in m.name.upper():
                    continue
                return m
    # dernier recours : le PATH
    for pattern in TOOL_PATTERNS[name]:
        if "*" in pattern:
            continue
        found = shutil.which(pattern.format(plat=plat))
        if found:
            return Path(found)
    return None


@dataclass
class Toolchain:
    """Ensemble des exécutables DualSPHysics résolus."""

    home: Optional[Path] = None
    tools: Dict[str, Optional[Path]] = field(default_factory=dict)

    @classmethod
    def discover(cls, home: Optional[os.PathLike] = None) -> "Toolchain":
        h = Path(home).expanduser() if home is not None else None
        if h is None:
            for cand in default_homes():
                if bin_dirs(cand):
                    h = cand
                    break
        tools = {name: find_tool(name, h) for name in TOOL_PATTERNS}
        return cls(home=h, tools=tools)

    def get(self, name: str) -> Optional[Path]:
        return self.tools.get(name)

    def require(self, name: str) -> Path:
        p = self.tools.get(name)
        if p is None:
            raise FileNotFoundError(
                f"exécutable DualSPHysics '{name}' introuvable. Définissez DUALSPHYSICS_HOME "
                f"(racine contenant bin/linux) ou DSPH_{name.upper()}=<chemin>."
            )
        return p

    def has(self, name: str) -> bool:
        return self.tools.get(name) is not None

    def is_complete(self) -> bool:
        return all(self.has(t) for t in REQUIRED_TOOLS)

    def environment(self) -> Dict[str, str]:
        """Variables d'environnement pour l'exécution (bibliothèques partagées)."""
        env = dict(os.environ)
        libdirs = [str(p.parent) for p in self.tools.values() if p is not None]
        if libdirs and platform_tag() != "win":
            current = env.get("LD_LIBRARY_PATH", "")
            uniq: List[str] = []
            for d in libdirs + ([current] if current else []):
                for part in d.split(os.pathsep):
                    if part and part not in uniq:
                        uniq.append(part)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(uniq)
        return env

    def report(self) -> str:
        lines = [f"DualSPHysics home : {self.home or '(non trouvé)'}"]
        for name in TOOL_PATTERNS:
            p = self.tools.get(name)
            mark = "OK " if p else "-- "
            req = " (requis)" if name in REQUIRED_TOOLS and not p else ""
            lines.append(f"  {mark}{name:<17} {p or 'introuvable'}{req}")
        return "\n".join(lines)


__all__ = ["REQUIRED_TOOLS", "TOOL_PATTERNS", "Toolchain", "default_homes", "find_tool", "platform_tag"]
