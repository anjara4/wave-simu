"""Description d'un cas de simulation de vagues.

Le cas est décrit dans un fichier YAML (ou JSON) qui est chargé dans des
dataclasses typées puis validé. Toutes les longueurs sont en mètres, les temps
en secondes, les angles en degrés.

Convention d'axes (celle de DualSPHysics) : ``x`` est la direction de
propagation de la houle, ``z`` la verticale ascendante, ``y`` la largeur du
canal (nulle en 2D). Le fond du canal est à ``z = 0`` et le batteur est au
voisinage de ``x = 0``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


class ConfigError(ValueError):
    """Erreur de configuration (valeur manquante ou incohérente)."""


# ---------------------------------------------------------------------------
# Sous-sections
# ---------------------------------------------------------------------------


@dataclass
class Beach:
    """Plage inclinée en fin de canal (absorbe une partie de la houle).

    La plage commence à ``x = start`` au niveau du fond et monte avec la pente
    ``slope`` (tangente, sans dimension) jusqu'à l'extrémité du canal.
    """

    start: float
    slope: float = 0.1

    def height_at(self, x: float) -> float:
        return max(0.0, (x - self.start) * self.slope)


@dataclass
class Flume:
    """Géométrie du canal à houle."""

    length: float = 10.0
    depth: float = 0.5  #: niveau d'eau au repos (SWL)
    height: float = 1.0  #: hauteur des parois
    width: float = 0.0  #: largeur (0 en 2D)
    beach: Optional[Beach] = None


@dataclass
class Waves:
    """Paramètres de la houle générée.

    ``kind`` vaut ``"regular"`` (houle monochromatique) ou ``"irregular"``
    (spectre JONSWAP ou Pierson-Moskowitz). Pour une houle irrégulière,
    ``height`` est la hauteur significative ``Hs`` et ``period`` la période
    de pic ``Tp``.
    """

    kind: str = "regular"
    height: float = 0.1
    period: float = 1.5
    order: int = 2  #: ordre de génération (1 ou 2)
    phase: float = 0.0  #: phase initiale, en unités de pi
    ramp: float = 1.0  #: nombre de périodes (régulier) ou secondes (irrégulier) de rampe
    start: float = 0.0
    duration: float = 0.0  #: 0 = jusqu'à la fin de la simulation
    # Houle irrégulière
    spectrum: str = "jonswap"
    peak_coef: float = 3.3
    n_waves: int = 128
    seed: int = 2
    discretization: str = "stretched"


@dataclass
class Awas:
    """Absorption active (AWAS) sur le batteur piston."""

    enabled: bool = False
    start: Optional[float] = None
    gauge_x: float = 5.0  #: distance sonde AWAS / batteur en unités de dp
    limit_acc: float = 2.0
    correction: bool = True
    coef_stroke: float = 1.8
    coef_period: float = 1.0
    power_func: float = 3.0


@dataclass
class Wavemaker:
    """Batteur à houle.

    ``kind`` : ``"piston"`` ou ``"flap"``. Pour un volet, ``hinge`` donne la
    position de l'axe par rapport au fond (``0`` = axe au fond, ``<0`` =
    au-dessus du fond, ``>0`` = en dessous).
    """

    kind: str = "piston"
    thickness: float = 0.05
    x: float = 0.0
    hinge: float = 0.0
    mk: int = 10
    waves: Waves = field(default_factory=Waves)
    awas: Awas = field(default_factory=Awas)
    save_motion: bool = True


@dataclass
class Damping:
    """Zone d'amortissement numérique en fin de canal."""

    enabled: bool = True
    start: Optional[float] = None  #: défaut : 2 longueurs d'onde avant la fin
    end: Optional[float] = None  #: défaut : extrémité du canal
    redumax: float = 10.0
    overlimit: float = 1.0
    factor: Sequence[float] = (1.0, 1.0, 1.0)


@dataclass
class Gauge:
    """Sonde de niveau d'eau (wave gauge) à la position ``(x, y)``."""

    name: str
    x: float
    y: float = 0.0


@dataclass
class Structure:
    """Obstacle ou corps flottant dans le canal.

    ``kind`` : ``"box"`` (obstacle fixe), ``"floating_box"`` (corps flottant),
    ``"cylinder"`` (cylindre vertical fixe, 3D uniquement).
    """

    kind: str = "box"
    name: str = "structure"
    origin: Sequence[float] = (5.0, 0.0, 0.0)
    size: Sequence[float] = (0.5, 0.0, 0.3)
    mk: Optional[int] = None
    density: float = 500.0  #: corps flottant : masse volumique (kg/m3)
    radius: float = 0.1  #: cylindre
    compute_forces: bool = True


@dataclass
class Physics:
    """Paramètres numériques SPH."""

    rhop0: float = 1000.0
    gravity: float = 9.81
    gamma: float = 7.0
    coefsound: float = 20.0
    coefh: float = 1.0
    cfl: float = 0.2
    kernel: str = "wendland"  #: ``cubic`` ou ``wendland``
    step_algorithm: str = "symplectic"  #: ``verlet`` ou ``symplectic``
    visco_treatment: str = "artificial"  #: ``artificial`` ou ``laminar``
    viscosity: float = 0.01
    visco_bound_factor: float = 1.0
    density_diffusion: int = 2  #: 0 none, 1 Molteni, 2 Fourtakas, 3 Fourtakas full
    density_diffusion_value: float = 0.1
    shifting: int = 0
    shift_coef: float = -2.0
    shift_tfs: float = 0.0
    rigid_algorithm: int = 1
    boundary: str = "dbc"  #: ``dbc`` ou ``mdbc``


@dataclass
class Timing:
    """Durée de simulation et fréquence de sortie."""

    end: float = 10.0
    output_dt: float = 0.05
    gauge_dt: Optional[float] = None  #: défaut : output_dt


@dataclass
class Case:
    """Cas complet de simulation de vagues."""

    name: str = "wave_case"
    dimension: str = "2d"
    dp: float = 0.01
    flume: Flume = field(default_factory=Flume)
    wavemaker: Wavemaker = field(default_factory=Wavemaker)
    damping: Damping = field(default_factory=Damping)
    gauges: List[Gauge] = field(default_factory=list)
    structures: List[Structure] = field(default_factory=list)
    physics: Physics = field(default_factory=Physics)
    time: Timing = field(default_factory=Timing)
    description: str = ""

    # ------------------------------------------------------------------ util
    @property
    def is_2d(self) -> bool:
        return self.dimension.lower() == "2d"

    @property
    def swl(self) -> float:
        return self.flume.depth

    def validate(self) -> "Case":
        """Vérifie la cohérence du cas et lève :class:`ConfigError` sinon."""
        errors: List[str] = []
        if self.dimension.lower() not in ("2d", "3d"):
            errors.append(f"dimension doit être '2d' ou '3d' (reçu {self.dimension!r})")
        if self.dp <= 0:
            errors.append("dp doit être > 0")
        fl = self.flume
        if fl.length <= 0 or fl.depth <= 0 or fl.height <= 0:
            errors.append("flume.length, flume.depth et flume.height doivent être > 0")
        if fl.depth >= fl.height:
            errors.append("flume.depth doit être < flume.height (sinon débordement)")
        if not self.is_2d and fl.width <= 0:
            errors.append("flume.width doit être > 0 en 3D")
        if self.is_2d and fl.width != 0:
            errors.append("flume.width doit être 0 en 2D")
        if fl.beach is not None:
            if not (0 < fl.beach.start < fl.length):
                errors.append("flume.beach.start doit être dans ]0, length[")
            if fl.beach.slope <= 0:
                errors.append("flume.beach.slope doit être > 0")
        wm = self.wavemaker
        if wm.kind not in ("piston", "flap"):
            errors.append(f"wavemaker.kind doit être 'piston' ou 'flap' (reçu {wm.kind!r})")
        if wm.thickness <= 0:
            errors.append("wavemaker.thickness doit être > 0")
        w = wm.waves
        if w.kind not in ("regular", "irregular"):
            errors.append(f"waves.kind doit être 'regular' ou 'irregular' (reçu {w.kind!r})")
        if w.height <= 0 or w.period <= 0:
            errors.append("waves.height et waves.period doivent être > 0")
        if w.order not in (1, 2):
            errors.append("waves.order doit valoir 1 ou 2")
        if w.kind == "irregular" and w.spectrum not in ("jonswap", "pierson-moskowitz"):
            errors.append("waves.spectrum doit être 'jonswap' ou 'pierson-moskowitz'")
        if w.height > fl.depth:
            errors.append("waves.height ne peut dépasser la profondeur d'eau")
        if wm.awas.enabled and wm.kind != "piston":
            errors.append("l'AWAS n'est disponible que pour un batteur piston")
        if self.damping.enabled and self.damping.start is not None and self.damping.end is not None:
            if not (0 <= self.damping.start < self.damping.end <= fl.length):
                errors.append("damping.start/end doivent vérifier 0 <= start < end <= flume.length")
        names = set()
        for g in self.gauges:
            if g.name in names:
                errors.append(f"sonde en double : {g.name}")
            names.add(g.name)
            if not (0 < g.x < fl.length):
                errors.append(f"sonde {g.name} : x={g.x} hors du canal")
        for s in self.structures:
            if s.kind not in ("box", "floating_box", "cylinder"):
                errors.append(f"structure {s.name} : kind inconnu {s.kind!r}")
            if len(s.origin) != 3 or len(s.size) != 3:
                errors.append(f"structure {s.name} : origin et size doivent avoir 3 composantes")
            if s.kind == "cylinder" and self.is_2d:
                errors.append(f"structure {s.name} : cylindre impossible en 2D")
            if s.kind == "floating_box" and s.density <= 0:
                errors.append(f"structure {s.name} : density doit être > 0")
        ph = self.physics
        if ph.kernel not in ("cubic", "wendland"):
            errors.append("physics.kernel doit être 'cubic' ou 'wendland'")
        if ph.step_algorithm not in ("verlet", "symplectic"):
            errors.append("physics.step_algorithm doit être 'verlet' ou 'symplectic'")
        if ph.visco_treatment not in ("artificial", "laminar"):
            errors.append("physics.visco_treatment doit être 'artificial' ou 'laminar'")
        if ph.boundary not in ("dbc", "mdbc"):
            errors.append("physics.boundary doit être 'dbc' ou 'mdbc'")
        if self.time.end <= 0 or self.time.output_dt <= 0:
            errors.append("time.end et time.output_dt doivent être > 0")
        if errors:
            raise ConfigError("Configuration invalide :\n  - " + "\n  - ".join(errors))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(asdict(self))


# ---------------------------------------------------------------------------
# (dé)sérialisation
# ---------------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _build(cls, data: Any, path: str = ""):
    """Construit récursivement une dataclass ``cls`` depuis un dict."""
    if data is None:
        return None
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__} : un mapping est attendu, reçu {type(data).__name__}")
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"{path or cls.__name__} : clé inconnue '{key}' (attendu : {', '.join(known)})")
        f = known[key]
        sub = f"{path}.{key}" if path else key
        kwargs[key] = _convert(f.type, value, sub)
    try:
        return cls(**kwargs)
    except TypeError as exc:  # champ obligatoire manquant
        raise ConfigError(f"{path or cls.__name__} : {exc}") from exc


_NESTED = {
    "Beach": Beach,
    "Flume": Flume,
    "Waves": Waves,
    "Awas": Awas,
    "Wavemaker": Wavemaker,
    "Damping": Damping,
    "Gauge": Gauge,
    "Structure": Structure,
    "Physics": Physics,
    "Timing": Timing,
}


def _convert(type_hint: Any, value: Any, path: str) -> Any:
    hint = str(type_hint)
    for name, cls in _NESTED.items():
        if hint in (name, f"Optional[{name}]", f"typing.Optional[{name}]"):
            return _build(cls, value, path)
        if hint in (f"List[{name}]", f"typing.List[{name}]"):
            if not isinstance(value, list):
                raise ConfigError(f"{path} : une liste est attendue")
            return [_build(cls, v, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def case_from_dict(data: Dict[str, Any]) -> Case:
    """Construit et valide un :class:`Case` depuis un dictionnaire."""
    if not isinstance(data, dict):
        raise ConfigError("le document de configuration doit être un mapping")
    return _build(Case, data).validate()


def load_case(path: Union[str, Path]) -> Case:
    """Charge un cas depuis un fichier YAML ou JSON."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"fichier introuvable : {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if data is None:
        raise ConfigError(f"fichier vide : {path}")
    return case_from_dict(data)


def save_case(case: Case, path: Union[str, Path]) -> Path:
    """Écrit un cas en YAML (ou JSON selon l'extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = case.to_dict()
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    return path


def resolve_damping(case: Case, wavelength: Optional[float] = None) -> Optional[Dict[str, float]]:
    """Retourne les bornes effectives de la zone d'amortissement (ou ``None``).

    Si ``start`` n'est pas fourni, la zone couvre 1,5 longueur d'onde en fin
    de canal (au moins 1 m), sans dépasser 40 % de la longueur du canal.
    """
    d = case.damping
    if not d.enabled:
        return None
    end = d.end if d.end is not None else case.flume.length
    if d.start is not None:
        start = d.start
    else:
        span = max(1.0, 1.5 * wavelength) if wavelength else 0.25 * case.flume.length
        span = min(span, 0.4 * case.flume.length)
        start = end - span
    return {"start": start, "end": end}


__all__ = [
    "Awas",
    "Beach",
    "Case",
    "ConfigError",
    "Damping",
    "Flume",
    "Gauge",
    "Physics",
    "Structure",
    "Timing",
    "Wavemaker",
    "Waves",
    "case_from_dict",
    "load_case",
    "resolve_damping",
    "save_case",
]
