"""Lecture des sorties DualSPHysics.

* :func:`parse_run_out` — journal ``Run.out`` (avancement, temps de calcul) ;
* :func:`read_measuretool_csv` — CSV d'élévation produit par MeasureTool ;
* :func:`read_gauge_csv` — CSV d'une sonde ``<gauges><swl>`` interne ;
* :func:`load_elevations` — rassemble toutes les séries d'élévation d'un run.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .runner import RunLayout


@dataclass
class RunInfo:
    """Informations extraites de ``Run.out``."""

    parts: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    info: Dict[str, str] = field(default_factory=dict)
    finished: bool = False
    runtime: Optional[float] = None
    particles: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def last_time(self) -> float:
        return float(self.parts[-1, 1]) if len(self.parts) else 0.0

    @property
    def n_parts(self) -> int:
        return len(self.parts)


# Ligne de la table de progression, v5.4 :
#   00001   0.050042   279   279   3,915   540   8.24  21-09-2026 19:58:31
# ou versions antérieures : Part_0001  0.050000  133  133  10.86  ...
_PART_RE = re.compile(
    r"^\s*(?:Part_)?(\d{4,6})\s+([\d.]+)\s+([\d,]+)\s+([\d,]+)(?:\s+[\d,]+\s+[\d,]+)?\s+([\d.]+)(?:\s|$)"
)
# "Simulation Runtime...............: 37.25 sec." / "Total particles: 3,915 (...)"
_KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _()+/-]{2,60}?)\.*:\s+(.+?)\s*$")


def parse_run_out(path: Union[str, Path]) -> RunInfo:
    """Analyse ``Run.out`` (tolérant aux variations de version)."""
    path = Path(path)
    ri = RunInfo()
    rows: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _PART_RE.match(line)
        if m:
            rows.append([float(m.group(i).replace(",", "")) for i in range(1, 6)])
            continue
        if "*** WARNING" in line or line.strip().startswith("WARNING"):
            ri.warnings.append(line.strip())
        m = _KV_RE.match(line)
        if m:
            key, value = m.group(1).strip(), m.group(2).strip()
            ri.info.setdefault(key, value)
            low = key.lower()
            if low.startswith("simulation runtime"):
                num = re.search(r"[\d.]+", value)
                if num:
                    ri.runtime = float(num.group())
            if low in ("particles", "total particles", "particles (fluid+bound)", "particles of simulation (initial)"):
                num = re.search(r"\d+", value.replace(",", ""))
                if num:
                    ri.particles = int(num.group())
        if "End of simulation" in line or "Simulation finished" in line:
            ri.finished = True
    ri.parts = np.array(rows) if rows else np.zeros((0, 5))
    if ri.runtime is not None:
        ri.finished = True
    return ri


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


@dataclass
class ElevationSeries:
    """Séries temporelles d'élévation pour plusieurs sondes."""

    time: np.ndarray
    elevation: Dict[str, np.ndarray]
    positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    source: Optional[Path] = None

    @property
    def names(self) -> List[str]:
        return list(self.elevation)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.elevation[name]


def _sniff_delimiter(text: str) -> str:
    head = text[:2000]
    return ";" if head.count(";") >= head.count(",") else ","


def _to_float(s: str) -> float:
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _read_csv_table(path: Path) -> Tuple[List[str], np.ndarray, Dict[str, List[str]]]:
    """Lit un CSV DualSPHysics : lignes de métadonnées, puis en-tête, puis données."""
    text = path.read_text(encoding="utf-8", errors="replace")
    delim = _sniff_delimiter(text)
    reader = list(csv.reader(text.splitlines(), delimiter=delim))
    header: Optional[List[str]] = None
    meta: Dict[str, List[str]] = {}
    rows: List[List[float]] = []
    for row in reader:
        if not row or all(not c.strip() for c in row):
            continue
        cells = [c.strip() for c in row]
        if header is None:
            joined = " ".join(cells).lower()
            if ("time" in joined) and not _is_numeric_row(cells):
                header = cells
            else:
                key_idx = next((i for i, c in enumerate(cells) if c and not _is_number(c)), 0)
                key = cells[key_idx].rstrip(":").strip()
                meta[_norm_meta_key(key)] = cells[key_idx + 1 :]
            continue
        if _is_numeric_row(cells):
            rows.append([_to_float(c) for c in cells])
    if header is None:
        raise ValueError(f"{path} : en-tête introuvable")
    data = np.array(rows, dtype=float) if rows else np.zeros((0, len(header)))
    return header, data, meta


def _is_number(c: str) -> bool:
    try:
        float(c.replace(",", "."))
        return True
    except ValueError:
        return False


def _norm_meta_key(key: str) -> str:
    """'Pos.x [m]', 'PosX [m]:' -> 'posx'."""
    k = key.lower().split("[")[0]
    return re.sub(r"[^a-z]", "", k)


def _is_numeric_row(cells: List[str]) -> bool:
    vals = [c for c in cells if c]
    if not vals:
        return False
    ok = 0
    for c in vals:
        try:
            float(c.replace(",", "."))
            ok += 1
        except ValueError:
            return False
    return ok >= 1


def _time_column(header: List[str]) -> int:
    for i, h in enumerate(header):
        if h.lower().startswith("time"):
            return i
    raise ValueError("colonne temps introuvable")


def read_measuretool_csv(path: Union[str, Path], names: Optional[List[str]] = None) -> ElevationSeries:
    """Lit un CSV ``*_Elevation.csv`` de MeasureTool.

    Les colonnes d'élévation sont, dans l'ordre, associées aux ``names``
    fournis (sinon nommées ``Elev_0``, ``Elev_1``…).
    """
    path = Path(path)
    header, data, meta = _read_csv_table(path)
    it = _time_column(header)
    cols = [i for i, h in enumerate(header) if i > it and "elev" in h.lower()]
    if not cols:  # toutes les colonnes après le temps
        cols = list(range(it + 1, len(header)))
    time = data[:, it] if len(data) else np.zeros(0)
    elev: Dict[str, np.ndarray] = {}
    positions: Dict[str, Tuple[float, float]] = {}
    px = meta.get("posx")
    py = meta.get("posy")
    for j, col in enumerate(cols):
        name = names[j] if names and j < len(names) else f"Elev_{j}"
        elev[name] = data[:, col] if len(data) else np.zeros(0)
        if px and j < len(px):
            positions[name] = (_to_float(px[j]), _to_float(py[j]) if py and j < len(py) else 0.0)
    return ElevationSeries(time=time, elevation=elev, positions=positions, source=path)


def read_gauge_csv(path: Union[str, Path], name: Optional[str] = None) -> ElevationSeries:
    """Lit le CSV d'une sonde interne ``<gauges><swl>`` (une colonne de niveau)."""
    path = Path(path)
    header, data, _ = _read_csv_table(path)
    it = _time_column(header)
    col = None
    lowered = [h.lower() for h in header]
    for key in ("swlz", "elev", "pos.z", "posz", "z"):
        for i, hl in enumerate(lowered):
            if i != it and hl.startswith(key):
                col = i
                break
        if col is not None:
            break
    if col is None:
        col = it + 1
    name = name or _guess_gauge_name(path)
    time = data[:, it] if len(data) else np.zeros(0)
    values = data[:, col] if len(data) else np.zeros(0)
    return ElevationSeries(time=time, elevation={name: values}, source=path)


def _guess_gauge_name(path: Path) -> str:
    stem = path.stem
    for prefix in ("gaugesswl_", "gaugeswl_", "gauges_", "gauge_"):
        if stem.lower().startswith(prefix):
            return stem[len(prefix):]
    return stem


def find_gauge_files(out_dir: Union[str, Path]) -> List[Path]:
    out_dir = Path(out_dir)
    found = set()
    for p in out_dir.rglob("*.csv"):
        name = p.name.lower()
        if not name.startswith("gauge") or "swl" not in name:
            continue
        if "awas" in name or "measuretool" in str(p).lower():
            continue
        found.add(p)
    return sorted(found)


def relative_to_rest(es: ElevationSeries, swl: Optional[float] = None) -> ElevationSeries:
    """Convertit des niveaux absolus (z de la surface) en élévation.

    Le niveau de repos retranché est le premier échantillon valide de chaque
    sonde (état initial au repos), ou ``swl`` si aucun échantillon n'est
    exploitable. Les valeurs déjà centrées (|niveau initial| petit devant
    ``swl``) sont laissées telles quelles.
    """
    out: Dict[str, np.ndarray] = {}
    for name, v in es.elevation.items():
        v = np.asarray(v, dtype=float)
        finite = v[np.isfinite(v)]
        if len(finite) == 0:
            out[name] = v
            continue
        rest = float(finite[0])
        if swl is not None and abs(rest) < 0.25 * swl:
            rest = 0.0  # déjà relatif
        out[name] = v - rest
    return ElevationSeries(time=es.time, elevation=out, positions=es.positions, source=es.source)


def load_elevations(workdir: Union[str, Path], case_name: str, gauge_names: Optional[List[str]] = None,
                    swl: Optional[float] = None, relative: bool = True) -> ElevationSeries:
    """Charge les élévations d'un run, MeasureTool en priorité, sinon sondes internes.

    Les deux sources fournissent le niveau absolu de la surface ; avec
    ``relative=True`` il est converti en élévation par rapport au niveau
    initial (voir :func:`relative_to_rest`).
    """
    lay = RunLayout(Path(workdir), case_name)
    candidates = sorted(lay.measure.glob("*Elevation*.csv")) if lay.measure.is_dir() else []
    if candidates:
        es = read_measuretool_csv(candidates[0], gauge_names)
        return relative_to_rest(es, swl) if relative else es
    files = find_gauge_files(lay.out)
    if not files:
        raise FileNotFoundError(f"aucune sortie de sonde trouvée dans {lay.out}")
    series: Dict[str, np.ndarray] = {}
    time = None
    for f in files:
        es = read_gauge_csv(f)
        for k, v in es.elevation.items():
            if time is None or len(es.time) < len(time):
                time = es.time
            series[k] = v
    n = min(len(v) for v in series.values())
    series = {k: v[:n] for k, v in series.items()}
    es = ElevationSeries(time=time[:n], elevation=series, source=files[0])
    return relative_to_rest(es, swl) if relative else es


__all__ = [
    "ElevationSeries",
    "RunInfo",
    "find_gauge_files",
    "load_elevations",
    "parse_run_out",
    "read_gauge_csv",
    "read_measuretool_csv",
    "relative_to_rest",
]
