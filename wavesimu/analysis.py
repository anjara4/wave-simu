"""Analyse de séries temporelles d'élévation de surface libre.

* :func:`zero_crossing` — statistiques par passage à zéro (H1/3, Hmax, Tm…) ;
* :func:`spectrum` — spectre de variance et paramètres spectraux (Hm0, Tp) ;
* :func:`reflection_goda` — coefficient de réflexion (Goda & Suzuki, 1976) ;
* :func:`analyze_gauge` — synthèse pour une sonde ;
* :func:`compare_to_target` — écart à la houle demandée.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .theory import wavenumber


@dataclass
class WaveStats:
    """Statistiques par passage à zéro montant."""

    n_waves: int = 0
    h_mean: float = float("nan")
    h_13: float = float("nan")  #: hauteur significative H1/3
    h_max: float = float("nan")
    h_rms: float = float("nan")
    t_mean: float = float("nan")
    t_13: float = float("nan")
    crest_max: float = float("nan")
    trough_min: float = float("nan")
    heights: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    periods: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.pop("heights")
        d.pop("periods")
        return d


@dataclass
class SpectralStats:
    freq: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    density: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    m0: float = float("nan")
    hm0: float = float("nan")
    tp: float = float("nan")
    tm01: float = float("nan")
    tm02: float = float("nan")

    def as_dict(self) -> Dict[str, float]:
        return {"m0": self.m0, "hm0": self.hm0, "tp": self.tp, "tm01": self.tm01, "tm02": self.tm02}


def _uniform(t: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Ré-échantillonne sur un pas constant et supprime les NaN."""
    t = np.asarray(t, dtype=float)
    eta = np.asarray(eta, dtype=float)
    mask = np.isfinite(t) & np.isfinite(eta)
    t, eta = t[mask], eta[mask]
    if len(t) < 4:
        raise ValueError("série trop courte")
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("pas de temps invalide")
    tu = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    return tu, np.interp(tu, t, eta), dt


def window(t: np.ndarray, eta: np.ndarray, t_start: Optional[float] = None, t_end: Optional[float] = None):
    """Restreint une série à ``[t_start, t_end]``."""
    t = np.asarray(t)
    eta = np.asarray(eta)
    m = np.ones_like(t, dtype=bool)
    if t_start is not None:
        m &= t >= t_start
    if t_end is not None:
        m &= t <= t_end
    return t[m], eta[m]


def zero_crossing(t, eta, detrend: bool = True) -> WaveStats:
    """Découpe la série en vagues par passage à zéro montant."""
    t = np.asarray(t, dtype=float)
    eta = np.asarray(eta, dtype=float)
    m = np.isfinite(eta)
    t, eta = t[m], eta[m]
    if detrend and len(eta):
        eta = eta - np.mean(eta)
    if len(eta) < 3:
        return WaveStats()
    up = np.where((eta[:-1] < 0) & (eta[1:] >= 0))[0]
    if len(up) < 2:
        return WaveStats(crest_max=float(np.max(eta)), trough_min=float(np.min(eta)))
    # interpolation linéaire des instants de passage
    tc = t[up] + (0 - eta[up]) * (t[up + 1] - t[up]) / (eta[up + 1] - eta[up])
    heights, periods = [], []
    for a, b, ta, tb in zip(up[:-1], up[1:], tc[:-1], tc[1:]):
        seg = eta[a + 1 : b + 1]
        if len(seg) == 0:
            continue
        heights.append(float(seg.max() - seg.min()))
        periods.append(float(tb - ta))
    H = np.array(heights)
    T = np.array(periods)
    n = len(H)
    if n == 0:
        return WaveStats()
    order = np.argsort(H)[::-1]
    n13 = max(1, int(round(n / 3)))
    top = order[:n13]
    return WaveStats(
        n_waves=n,
        h_mean=float(H.mean()),
        h_13=float(H[top].mean()),
        h_max=float(H.max()),
        h_rms=float(np.sqrt(np.mean(H**2))),
        t_mean=float(T.mean()),
        t_13=float(T[top].mean()),
        crest_max=float(eta.max()),
        trough_min=float(eta.min()),
        heights=H,
        periods=T,
    )


def spectrum(t, eta, nseg: int = 4, detrend: bool = True, min_seglen: int = 256, pad: int = 4) -> SpectralStats:
    """Spectre de variance par périodogrammes moyennés (fenêtre de Hann).

    Le signal est découpé en au plus ``nseg`` segments (recouvrement 50 %) d'au
    moins ``min_seglen`` échantillons ; les enregistrements courts utilisent
    donc moins de segments pour conserver la résolution fréquentielle. Les
    segments sont complétés par des zéros (facteur ``pad``) et le pic est
    affiné par interpolation parabolique.
    """
    tu, eu, dt = _uniform(t, eta)
    if detrend:
        eu = eu - eu.mean()
    n = len(eu)
    nseg_eff = max(1, min(nseg, n // min_seglen))
    seglen = max(16, n // nseg_eff)
    seglen = int(2 ** math.floor(math.log2(seglen)))
    seglen = min(seglen, n)
    nfft = seglen * max(1, pad)
    step = max(1, seglen // 2)
    win = np.hanning(seglen)
    norm = np.sum(win**2) * (1.0 / dt)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    acc = np.zeros(len(freqs))
    count = 0
    for start in range(0, n - seglen + 1, step):
        seg = eu[start : start + seglen] * win
        spec = np.abs(np.fft.rfft(seg, n=nfft)) ** 2 / norm
        spec[1:-1] *= 2.0  # spectre unilatéral
        acc += spec
        count += 1
    if count == 0:
        return SpectralStats()
    S = acc / count
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    m0 = float(np.sum(S) * df)
    m1 = float(np.sum(S * freqs) * df)
    m2 = float(np.sum(S * freqs**2) * df)
    ip = int(np.argmax(S[1:]) + 1) if len(S) > 1 else 0
    fp = _refine_peak(freqs, S, ip)
    return SpectralStats(
        freq=freqs,
        density=S,
        m0=m0,
        hm0=4.0 * math.sqrt(max(m0, 0.0)),
        tp=1.0 / fp if fp > 0 else float("nan"),
        tm01=(m0 / m1) if m1 > 0 else float("nan"),
        tm02=math.sqrt(m0 / m2) if m2 > 0 else float("nan"),
    )


def _refine_peak(freqs: np.ndarray, S: np.ndarray, ip: int) -> float:
    """Interpolation parabolique de la position du pic autour de l'indice ``ip``."""
    if ip <= 0 or ip >= len(S) - 1:
        return float(freqs[ip]) if len(freqs) else 0.0
    y0, y1, y2 = S[ip - 1], S[ip], S[ip + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    delta = max(-1.0, min(1.0, delta))
    return float(freqs[ip] + delta * (freqs[1] - freqs[0]))


@dataclass
class ReflectionResult:
    kr: float
    h_incident: float
    h_reflected: float
    n_freq: int
    kdl_min: float = 0.05 * math.pi
    kdl_max: float = 0.45 * math.pi

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def reflection_goda(t, eta1, eta2, dl: float, depth: float, g: float = 9.81,
                    fmin: Optional[float] = None, fmax: Optional[float] = None,
                    period: Optional[float] = None) -> ReflectionResult:
    """Coefficient de réflexion par la méthode à deux sondes de Goda & Suzuki.

    ``eta1`` est la sonde amont (côté batteur), ``eta2`` la sonde aval, à une
    distance ``dl`` (m). Seules les fréquences vérifiant
    ``0.05 pi < k dl < 0.45 pi`` (et ``fmin <= f <= fmax`` si fournis) sont
    retenues. Pour une houle régulière, donner ``period`` tronque le signal à
    un nombre entier de périodes, ce qui limite les fuites spectrales.
    """
    if dl <= 0:
        raise ValueError("dl doit être > 0")
    tu, e1, dt = _uniform(t, eta1)
    _, e2, _ = _uniform(t, eta2)
    n = min(len(e1), len(e2))
    if period is not None and period > 0:
        n_per = int(math.floor(n * dt / period))
        if n_per >= 1:
            n = min(n, int(round(n_per * period / dt)))
    e1 = e1[:n] - e1[:n].mean()
    e2 = e2[:n] - e2[:n].mean()
    F1 = np.fft.rfft(e1) * 2.0 / n
    F2 = np.fft.rfft(e2) * 2.0 / n
    freqs = np.fft.rfftfreq(n, d=dt)
    A1, B1 = F1.real, -F1.imag
    A2, B2 = F2.real, -F2.imag
    ai2 = ar2 = 0.0
    count = 0
    for i in range(1, len(freqs)):
        f = freqs[i]
        if fmin is not None and f < fmin:
            continue
        if fmax is not None and f > fmax:
            continue
        k = wavenumber(1.0 / f, depth, g)
        kdl = k * dl
        if not (0.05 * math.pi < kdl < 0.45 * math.pi):
            continue
        s, c = math.sin(kdl), math.cos(kdl)
        denom = 2.0 * abs(s)
        ai = math.sqrt((A2[i] - A1[i] * c - B1[i] * s) ** 2 + (B2[i] + A1[i] * s - B1[i] * c) ** 2) / denom
        ar = math.sqrt((A2[i] - A1[i] * c + B1[i] * s) ** 2 + (B2[i] - A1[i] * s - B1[i] * c) ** 2) / denom
        ai2 += ai * ai
        ar2 += ar * ar
        count += 1
    if count == 0 or ai2 <= 0:
        return ReflectionResult(kr=float("nan"), h_incident=float("nan"), h_reflected=float("nan"), n_freq=0)
    return ReflectionResult(
        kr=math.sqrt(ar2 / ai2),
        h_incident=2.0 * math.sqrt(ai2),  # hauteur équivalente (somme quadratique des composantes)
        h_reflected=2.0 * math.sqrt(ar2),
        n_freq=count,
    )


@dataclass
class GaugeReport:
    name: str
    t_start: float
    t_end: float
    mean_level: float
    stats: WaveStats
    spectral: SpectralStats

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "mean_level": self.mean_level,
            **{f"zc_{k}": v for k, v in self.stats.as_dict().items()},
            **{f"sp_{k}": v for k, v in self.spectral.as_dict().items()},
        }


def analyze_gauge(name: str, t, eta, t_start: Optional[float] = None, t_end: Optional[float] = None) -> GaugeReport:
    """Statistiques complètes d'une sonde sur la fenêtre ``[t_start, t_end]``."""
    tw, ew = window(t, eta, t_start, t_end)
    if len(tw) < 4:
        raise ValueError(f"sonde {name} : trop peu de points dans la fenêtre")
    mean = float(np.nanmean(ew))
    zc = zero_crossing(tw, ew)
    try:
        sp = spectrum(tw, ew)
    except ValueError:
        sp = SpectralStats()
    return GaugeReport(name=name, t_start=float(tw[0]), t_end=float(tw[-1]), mean_level=mean, stats=zc, spectral=sp)


def compare_to_target(report: GaugeReport, height: float, period: float, irregular: bool = False) -> Dict[str, float]:
    """Écart relatif entre houle mesurée et houle demandée."""
    h = report.spectral.hm0 if irregular else report.stats.h_mean
    T = report.spectral.tp if irregular else report.stats.t_mean
    return {
        "height_target": height,
        "height_measured": h,
        "height_ratio": h / height if height else float("nan"),
        "period_target": period,
        "period_measured": T,
        "period_ratio": T / period if period else float("nan"),
    }


def suggested_window(t_end: float, period: float, wavelength: float, gauge_x: float, celerity: float,
                     ramp_periods: float = 1.0) -> Tuple[float, float]:
    """Fenêtre d'analyse : après l'arrivée du front d'onde + rampe, jusqu'à la fin."""
    arrival = gauge_x / max(celerity, 1e-9)
    start = arrival + (ramp_periods + 2.0) * period
    return (min(start, 0.5 * t_end), t_end)


__all__ = [
    "GaugeReport",
    "ReflectionResult",
    "SpectralStats",
    "WaveStats",
    "analyze_gauge",
    "compare_to_target",
    "reflection_goda",
    "spectrum",
    "suggested_window",
    "window",
    "zero_crossing",
]
