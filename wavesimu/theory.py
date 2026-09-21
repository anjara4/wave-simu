"""Théorie linéaire de la houle et fonctions de transfert des batteurs.

Ces formules servent à dimensionner un cas (longueur d'onde, course du
batteur, résolution ``dp`` recommandée, taille du domaine…) et à vérifier
les résultats de simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

G = 9.81


def wavenumber(period: float, depth: float, g: float = G, tol: float = 1e-12) -> float:
    """Nombre d'onde ``k`` solution de la relation de dispersion linéaire
    ``omega^2 = g k tanh(k h)`` (résolution par Newton)."""
    if period <= 0 or depth <= 0:
        raise ValueError("period et depth doivent être > 0")
    omega = 2.0 * math.pi / period
    # Approximation initiale (Guo 2002)
    k = omega * omega / g
    k = k / math.tanh((omega * math.sqrt(depth / g)) ** 1.5) ** (2.0 / 3.0)
    for _ in range(100):
        kh = k * depth
        th = math.tanh(kh)
        f = g * k * th - omega * omega
        df = g * th + g * kh * (1.0 - th * th)
        dk = f / df
        k -= dk
        if abs(dk) < tol * max(k, 1.0):
            break
    return k


def wavelength(period: float, depth: float, g: float = G) -> float:
    """Longueur d'onde ``L = 2 pi / k``."""
    return 2.0 * math.pi / wavenumber(period, depth, g)


def deep_water_wavelength(period: float, g: float = G) -> float:
    return g * period * period / (2.0 * math.pi)


def celerity(period: float, depth: float, g: float = G) -> float:
    """Vitesse de phase ``c = L / T``."""
    return wavelength(period, depth, g) / period


def group_velocity(period: float, depth: float, g: float = G) -> float:
    """Vitesse de groupe ``cg = n c``."""
    k = wavenumber(period, depth, g)
    kh = k * depth
    n = 0.5 * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    return n * celerity(period, depth, g)


def piston_transfer(k: float, depth: float) -> float:
    """Fonction de transfert de Biésel pour un piston : ``H / S``."""
    kh = k * depth
    return 2.0 * (math.cosh(2.0 * kh) - 1.0) / (math.sinh(2.0 * kh) + 2.0 * kh)


def flap_transfer(k: float, depth: float) -> float:
    """Fonction de transfert d'un volet articulé au fond : ``H / S`` où ``S``
    est la course en surface."""
    kh = k * depth
    return (
        4.0
        * math.sinh(kh)
        / (math.sinh(2.0 * kh) + 2.0 * kh)
        * (math.sinh(kh) + (1.0 - math.cosh(kh)) / kh)
    )


def piston_stroke(height: float, period: float, depth: float, g: float = G) -> float:
    """Course totale ``S`` (crête à crête) d'un piston pour générer ``H``."""
    return height / piston_transfer(wavenumber(period, depth, g), depth)


def flap_stroke(height: float, period: float, depth: float, g: float = G) -> float:
    """Course totale en surface d'un volet pour générer ``H``."""
    return height / flap_transfer(wavenumber(period, depth, g), depth)


def ursell(height: float, period: float, depth: float, g: float = G) -> float:
    """Nombre d'Ursell ``H L^2 / h^3``."""
    L = wavelength(period, depth, g)
    return height * L * L / depth**3


def breaking_limit(period: float, depth: float, g: float = G) -> float:
    """Hauteur limite de déferlement (Miche) : ``H = 0.142 L tanh(kh)``."""
    k = wavenumber(period, depth, g)
    return 0.142 * (2.0 * math.pi / k) * math.tanh(k * depth)


def regime(period: float, depth: float, g: float = G) -> str:
    """Classe la houle : ``deep``, ``intermediate`` ou ``shallow``."""
    r = depth / wavelength(period, depth, g)
    if r > 0.5:
        return "deep"
    if r < 0.05:
        return "shallow"
    return "intermediate"


def surface_elevation(t, x, height: float, period: float, depth: float, g: float = G, phase: float = 0.0):
    """Élévation de surface linéaire ``eta = H/2 cos(k x - w t + phi)``.

    Accepte des scalaires ou des tableaux numpy.
    """
    import numpy as np

    k = wavenumber(period, depth, g)
    w = 2.0 * math.pi / period
    return 0.5 * height * np.cos(k * np.asarray(x) - w * np.asarray(t) + phase)


@dataclass
class WaveSummary:
    """Résumé des grandeurs théoriques d'une houle régulière."""

    height: float
    period: float
    depth: float
    wavenumber: float
    wavelength: float
    celerity: float
    group_velocity: float
    steepness: float
    ursell: float
    breaking_height: float
    regime: str
    piston_stroke: float
    flap_stroke: float

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def summarize(height: float, period: float, depth: float, g: float = G) -> WaveSummary:
    k = wavenumber(period, depth, g)
    L = 2.0 * math.pi / k
    return WaveSummary(
        height=height,
        period=period,
        depth=depth,
        wavenumber=k,
        wavelength=L,
        celerity=L / period,
        group_velocity=group_velocity(period, depth, g),
        steepness=height / L,
        ursell=height * L * L / depth**3,
        breaking_height=breaking_limit(period, depth, g),
        regime=regime(period, depth, g),
        piston_stroke=height / piston_transfer(k, depth),
        flap_stroke=height / flap_transfer(k, depth),
    )


def recommended_dp(height: float, particles_per_height: int = 10) -> float:
    """Résolution ``dp`` recommandée : au moins ``particles_per_height``
    particules par hauteur de vague (10 est le minimum usuel)."""
    return height / particles_per_height


def estimate_particles(case) -> Dict[str, int]:
    """Estime le nombre de particules fluide et frontière d'un cas."""
    fl = case.flume
    dp = case.dp
    nx = fl.length / dp
    nz_f = fl.depth / dp
    nz_b = fl.height / dp
    if case.is_2d:
        fluid = nx * nz_f
        bound = 3 * (nx + 2 * nz_b)  # ~3 couches de particules DBC
    else:
        ny = fl.width / dp
        fluid = nx * ny * nz_f
        bound = 3 * (nx * ny + 2 * nz_b * (nx + ny))
    if fl.beach is not None:
        removed = 0.5 * (fl.length - fl.beach.start) * min(fl.depth, fl.beach.height_at(fl.length)) / dp**2
        if not case.is_2d:
            removed *= fl.width / dp
        fluid = max(0.0, fluid - removed)
    return {"fluid": int(fluid), "boundary": int(bound), "total": int(fluid + bound)}


def estimate_cost(case) -> Dict[str, float]:
    """Estimation grossière du coût (pas de temps, nombre de pas)."""
    ph = case.physics
    h = ph.coefh * math.sqrt(2.0 if case.is_2d else 3.0) * case.dp
    # vitesse du son : coefsound * sqrt(g * depth) (vitesse système par défaut = rupture de barrage)
    cs = ph.coefsound * math.sqrt(ph.gravity * case.flume.depth)
    dt = ph.cfl * h / cs
    steps = case.time.end / dt
    return {"h": h, "speedsound": cs, "dt_estimate": dt, "steps_estimate": steps}


__all__ = [
    "G",
    "WaveSummary",
    "breaking_limit",
    "celerity",
    "deep_water_wavelength",
    "estimate_cost",
    "estimate_particles",
    "flap_stroke",
    "flap_transfer",
    "group_velocity",
    "piston_stroke",
    "piston_transfer",
    "recommended_dp",
    "regime",
    "summarize",
    "surface_elevation",
    "ursell",
    "wavelength",
    "wavenumber",
]
