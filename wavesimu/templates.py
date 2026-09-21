"""Cas d'exemple prêts à l'emploi (``wavesimu init --template``)."""

from __future__ import annotations

from typing import Dict

from .config import Awas, Beach, Case, Damping, Flume, Gauge, Physics, Structure, Timing, Wavemaker, Waves


def regular_flume() -> Case:
    """Canal 2D, piston, houle régulière H=0.1 m T=1.5 s, profondeur 0.5 m."""
    return Case(
        name="flume_regular",
        description="Canal 2D avec batteur piston, houle régulière (H=0.10 m, T=1.5 s, h=0.5 m)",
        dimension="2d",
        dp=0.01,
        flume=Flume(length=12.0, depth=0.5, height=1.0),
        wavemaker=Wavemaker(kind="piston", waves=Waves(kind="regular", height=0.1, period=1.5, order=2, ramp=1), awas=Awas(enabled=True)),
        damping=Damping(enabled=True, start=8.5),
        gauges=[Gauge("WG1", 2.0), Gauge("WG2", 4.0), Gauge("WG3", 4.5), Gauge("WG4", 6.0)],
        time=Timing(end=15.0, output_dt=0.05),
    )


def irregular_flume() -> Case:
    """Canal 2D, piston, houle irrégulière JONSWAP Hs=0.08 m Tp=1.6 s."""
    c = regular_flume()
    c.name = "flume_irregular"
    c.description = "Canal 2D, batteur piston, houle irrégulière JONSWAP (Hs=0.08 m, Tp=1.6 s)"
    c.wavemaker.waves = Waves(kind="irregular", height=0.08, period=1.6, order=2, ramp=2.0, spectrum="jonswap", peak_coef=3.3, n_waves=128, seed=2)
    c.wavemaker.awas = Awas(enabled=False)
    c.time = Timing(end=60.0, output_dt=0.05)
    return c


def flap_flume() -> Case:
    """Canal 2D avec batteur volet articulé au fond."""
    c = regular_flume()
    c.name = "flume_flap"
    c.description = "Canal 2D, batteur volet (flap), houle régulière"
    c.wavemaker = Wavemaker(kind="flap", thickness=0.05, waves=Waves(kind="regular", height=0.1, period=1.5, order=1, ramp=1))
    return c


def beach_flume() -> Case:
    """Canal 2D avec plage inclinée (pente 1:10) et déferlement."""
    c = regular_flume()
    c.name = "flume_beach"
    c.description = "Canal 2D, piston, plage inclinée 1:10 à partir de x=6 m (run-up, déferlement)"
    c.flume = Flume(length=12.0, depth=0.5, height=1.0, beach=Beach(start=6.0, slope=0.1))
    c.damping = Damping(enabled=False)
    c.gauges = [Gauge("WG1", 2.0), Gauge("WG2", 4.0), Gauge("WG3", 7.0), Gauge("WG4", 9.0)]
    return c


def structure_flume() -> Case:
    """Canal 2D avec obstacle immergé (récif artificiel) et calcul des forces."""
    c = regular_flume()
    c.name = "flume_structure"
    c.description = "Canal 2D, piston, obstacle rectangulaire immergé à x=6 m avec calcul des forces"
    c.structures = [Structure(kind="box", name="reef", origin=(6.0, 0.0, 0.0), size=(0.6, 0.0, 0.3), compute_forces=True)]
    c.gauges = [Gauge("WG1", 2.0), Gauge("WG2", 4.0), Gauge("WG3", 4.5), Gauge("WG4", 5.5), Gauge("WG5", 8.0)]
    return c


def floating_flume() -> Case:
    """Canal 2D avec caisson flottant (masse volumique 500 kg/m3)."""
    c = regular_flume()
    c.name = "flume_floating"
    c.description = "Canal 2D, piston, caisson flottant à x=6 m (mouvement et forces)"
    c.structures = [Structure(kind="floating_box", name="caisson", origin=(6.0, 0.0, 0.4), size=(0.4, 0.0, 0.2), density=500.0)]
    c.gauges = [Gauge("WG1", 2.0), Gauge("WG2", 4.0), Gauge("WG3", 8.0)]
    c.physics = Physics(rigid_algorithm=1)
    return c


def tank_3d() -> Case:
    """Bassin 3D avec cylindre vertical (dp grossier pour un premier essai)."""
    c = regular_flume()
    c.name = "tank_3d"
    c.description = "Bassin 3D 8 x 1 m, piston, cylindre vertical à x=4 m"
    c.dimension = "3d"
    c.dp = 0.02
    c.flume = Flume(length=8.0, width=1.0, depth=0.5, height=0.9)
    c.structures = [Structure(kind="cylinder", name="pile", origin=(4.0, 0.5, 0.0), size=(0.0, 0.0, 0.9), radius=0.1)]
    c.gauges = [Gauge("WG1", 2.0, 0.5), Gauge("WG2", 3.5, 0.5), Gauge("WG3", 6.0, 0.5)]
    c.time = Timing(end=8.0, output_dt=0.05)
    return c


TEMPLATES: Dict[str, "callable"] = {
    "regular": regular_flume,
    "irregular": irregular_flume,
    "flap": flap_flume,
    "beach": beach_flume,
    "structure": structure_flume,
    "floating": floating_flume,
    "tank3d": tank_3d,
}


def get_template(name: str) -> Case:
    if name not in TEMPLATES:
        raise KeyError(f"modèle inconnu '{name}' (disponibles : {', '.join(TEMPLATES)})")
    return TEMPLATES[name]().validate()


__all__ = ["TEMPLATES", "get_template"]
