"""Figures matplotlib (dépendance optionnelle).

Toutes les fonctions retournent la figure créée et acceptent ``path`` pour
l'enregistrer directement. Le backend ``Agg`` est utilisé pour permettre un
usage sans affichage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib est requis pour les figures : pip install 'wavesimu[plot]'") from exc
    return plt


def _save(fig, path: Optional[Union[str, Path]]):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_elevations(time, series: Dict[str, np.ndarray], target_height: Optional[float] = None,
                    path: Optional[Union[str, Path]] = None, title: str = "Élévation de surface libre"):
    """Séries temporelles d'élévation, une sous-figure par sonde."""
    plt = _plt()
    names = list(series)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.2 * len(names) + 1), sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        ax.plot(time, series[name], lw=0.9, color="#1f77b4")
        if target_height:
            ax.axhline(target_height / 2, color="#d62728", ls="--", lw=0.8)
            ax.axhline(-target_height / 2, color="#d62728", ls="--", lw=0.8)
        ax.set_ylabel(f"{name}\nη [m]")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_spectra(spectra: Dict[str, "object"], target_period: Optional[float] = None,
                 path: Optional[Union[str, Path]] = None, fmax: Optional[float] = None):
    """Spectres de variance ``S(f)`` de plusieurs sondes."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, sp in spectra.items():
        if len(sp.freq) == 0:
            continue
        ax.plot(sp.freq, sp.density, lw=1.0, label=name)
    if target_period:
        ax.axvline(1.0 / target_period, color="#d62728", ls="--", lw=0.8, label=f"1/T = {1/target_period:.2f} Hz")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("S(f) [m²/Hz]")
    if fmax:
        ax.set_xlim(0, fmax)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_wave_heights(reports: Sequence["object"], target_height: Optional[float] = None,
                      path: Optional[Union[str, Path]] = None):
    """Hauteurs (moyenne, H1/3, Hmax, Hm0) par sonde."""
    plt = _plt()
    names = [r.name for r in reports]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names) + 3), 4))
    w = 0.2
    ax.bar(x - 1.5 * w, [r.stats.h_mean for r in reports], w, label="H moy.")
    ax.bar(x - 0.5 * w, [r.stats.h_13 for r in reports], w, label="H1/3")
    ax.bar(x + 0.5 * w, [r.stats.h_max for r in reports], w, label="Hmax")
    ax.bar(x + 1.5 * w, [r.spectral.hm0 for r in reports], w, label="Hm0")
    if target_height:
        ax.axhline(target_height, color="#d62728", ls="--", lw=1, label="cible")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("H [m]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_run_progress(parts: np.ndarray, path: Optional[Union[str, Path]] = None):
    """Avancement de la simulation depuis ``Run.out`` (temps simulé vs temps CPU)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(parts):
        ax.plot(parts[:, 4], parts[:, 1], marker=".", lw=0.8)
    ax.set_xlabel("temps de calcul [s]")
    ax.set_ylabel("temps simulé [s]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_case_layout(case, path: Optional[Union[str, Path]] = None):
    """Schéma (x, z) du canal : parois, eau, batteur, plage, structures, sondes, amortissement."""
    from .casegen import CaseGenerator

    plt = _plt()
    from matplotlib.patches import Polygon, Rectangle

    gen = CaseGenerator(case)
    fl = case.flume
    fig, ax = plt.subplots(figsize=(11, 3.5))
    x0 = gen.back_x()
    ax.add_patch(Rectangle((case.wavemaker.x, 0), fl.length - case.wavemaker.x, fl.depth, color="#9ecae1", label="eau"))
    if fl.beach is not None:
        b = fl.beach
        ax.add_patch(Polygon([(b.start, 0), (fl.length, 0), (fl.length, min(b.height_at(fl.length), fl.height))], color="#c7a27c", label="plage"))
    ax.plot([x0, fl.length, fl.length], [0, 0, fl.height], color="k", lw=2)
    wm = case.wavemaker
    ax.add_patch(Rectangle((wm.x - wm.thickness, 0), wm.thickness, fl.height, color="#e6550d", label=f"batteur ({wm.kind})"))
    if gen.damping:
        ax.axvspan(gen.damping["start"], gen.damping["end"], color="#bdbdbd", alpha=0.4, label="amortissement")
    for i, s in enumerate(case.structures):
        ox, _, oz = s.origin
        sx, _, sz = s.size
        if s.kind == "cylinder":
            sx, sz = 2 * s.radius, s.size[2]
            ox -= s.radius
        ax.add_patch(Rectangle((ox, oz), sx, sz, color="#756bb1", alpha=0.8, label=f"{s.name} ({s.kind})"))
    for g in case.gauges:
        ax.axvline(g.x, color="#31a354", ls=":", lw=1)
        ax.text(g.x, fl.height * 0.95, g.name, ha="center", va="top", fontsize=8, color="#31a354")
    ax.set_xlim(x0 - 0.1, fl.length + 0.1)
    ax.set_ylim(-0.05, fl.height * 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"{case.name} — {wm.waves.kind} H={wm.waves.height} m T={wm.waves.period} s, dp={case.dp} m")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


__all__ = ["plot_case_layout", "plot_elevations", "plot_run_progress", "plot_spectra", "plot_wave_heights"]
