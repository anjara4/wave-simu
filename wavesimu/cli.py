"""Interface en ligne de commande ``wavesimu``.

Sous-commandes :

* ``init``     — crée un fichier de cas YAML depuis un modèle ;
* ``check``    — valide un cas et affiche les grandeurs théoriques ;
* ``gencase``  — écrit le XML GenCase et le script ``run.sh`` (sans exécuter) ;
* ``run``      — exécute la chaîne complète (GenCase, DualSPHysics, post) ;
* ``post``     — relance uniquement le post-traitement d'un run existant ;
* ``analyze``  — statistiques de houle aux sondes (+ figures) ;
* ``theory``   — calculatrice de théorie linéaire ;
* ``doctor``   — vérifie l'installation DualSPHysics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .config import Case, ConfigError, load_case, resolve_damping, save_case
from .theory import estimate_cost, estimate_particles, summarize, wavelength


def _err(msg: str) -> None:
    print(f"erreur : {msg}", file=sys.stderr)


def _fmt_table(rows: List[tuple], indent: str = "  ") -> str:
    width = max(len(str(r[0])) for r in rows) if rows else 0
    out = []
    for k, v in rows:
        if isinstance(v, float):
            v = f"{v:.4g}"
        out.append(f"{indent}{str(k):<{width}}  {v}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# commandes
# ---------------------------------------------------------------------------


def cmd_init(args) -> int:
    from .templates import TEMPLATES, get_template

    if args.list:
        for name, fn in TEMPLATES.items():
            print(f"  {name:<10} {fn.__doc__.strip() if fn.__doc__ else ''}")
        return 0
    case = get_template(args.template)
    if args.name:
        case.name = args.name
    out = Path(args.output) if args.output else Path(f"{case.name}.yaml")
    if out.exists() and not args.force:
        _err(f"{out} existe déjà (utilisez --force)")
        return 1
    save_case(case, out)
    print(f"cas '{case.name}' écrit dans {out}")
    return 0


def _load(path: str) -> Optional[Case]:
    try:
        return load_case(path)
    except ConfigError as exc:
        _err(str(exc))
        return None


def check_report(case: Case) -> Dict:
    """Résumé théorique + estimations + avertissements pour un cas."""
    w = case.wavemaker.waves
    s = summarize(w.height, w.period, case.flume.depth, case.physics.gravity)
    parts = estimate_particles(case)
    cost = estimate_cost(case)
    damp = resolve_damping(case, s.wavelength)
    warnings: List[str] = []
    ppw = w.height / case.dp
    if ppw < 10:
        warnings.append(f"seulement {ppw:.1f} particules par hauteur de vague (recommandé >= 10) : réduire dp à {w.height/10:.4f} m")
    if case.flume.length < 3 * s.wavelength:
        warnings.append(f"canal court ({case.flume.length:.1f} m) pour L = {s.wavelength:.2f} m : prévoir >= 3 L")
    if w.height > 0.8 * s.breaking_height:
        warnings.append(f"H = {w.height} m proche de la limite de déferlement ({s.breaking_height:.3f} m)")
    if s.ursell > 26 and w.order == 1:
        warnings.append(f"nombre d'Ursell élevé ({s.ursell:.0f}) : utiliser waves.order = 2")
    stroke = s.piston_stroke if case.wavemaker.kind == "piston" else s.flap_stroke
    if stroke > 0.5 * case.flume.length:
        warnings.append("course du batteur incohérente avec la taille du canal")
    if case.flume.height < case.flume.depth + 1.5 * w.height:
        warnings.append("parois trop basses : prévoir height >= depth + 1.5 H")
    for g in case.gauges:
        if damp and g.x > damp["start"]:
            warnings.append(f"sonde {g.name} située dans la zone d'amortissement (x >= {damp['start']:.2f} m)")
    if parts["total"] > 5_000_000 and not case.is_2d:
        warnings.append(f"~{parts['total']/1e6:.1f} M particules : prévoir un GPU")
    if case.time.end < 10 * w.period:
        warnings.append("durée courte : prévoir au moins 10 périodes pour des statistiques fiables")
    return {
        "case": case.name,
        "theory": s.as_dict(),
        "stroke": stroke,
        "particles": parts,
        "cost": cost,
        "damping": damp,
        "warnings": warnings,
    }


def cmd_check(args) -> int:
    case = _load(args.case)
    if case is None:
        return 1
    rep = check_report(case)
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0
    th = rep["theory"]
    print(f"Cas '{case.name}' valide ({case.dimension}, dp = {case.dp} m)")
    print("Houle théorique (théorie linéaire) :")
    print(_fmt_table([
        ("hauteur H", th["height"]), ("période T", th["period"]), ("profondeur h", th["depth"]),
        ("longueur d'onde L", th["wavelength"]), ("célérité c", th["celerity"]), ("vitesse de groupe cg", th["group_velocity"]),
        ("cambrure H/L", th["steepness"]), ("régime", th["regime"]), ("Ursell", th["ursell"]),
        ("H déferlement (Miche)", th["breaking_height"]),
        (f"course batteur ({case.wavemaker.kind})", rep["stroke"]),
    ]))
    p = rep["particles"]
    c = rep["cost"]
    print("Estimations :")
    print(_fmt_table([
        ("particules fluide", f"{p['fluid']:,}"), ("particules frontière", f"{p['boundary']:,}"),
        ("h (lissage)", c["h"]), ("vitesse du son", c["speedsound"]), ("dt estimé", c["dt_estimate"]),
        ("pas de temps", f"{c['steps_estimate']:,.0f}"),
    ]))
    if rep["damping"]:
        print(f"Amortissement : x in [{rep['damping']['start']:.2f}, {rep['damping']['end']:.2f}] m")
    if rep["warnings"]:
        print("Avertissements :")
        for w in rep["warnings"]:
            print(f"  ! {w}")
    return 0


def _runner(args, case: Case, dry_run: bool = False, **overrides):
    from .runner import RunOptions, Runner
    from .tools import Toolchain

    tc = Toolchain.discover(getattr(args, "dsph_home", None))
    opts = RunOptions(
        gpu=getattr(args, "gpu", False),
        threads=getattr(args, "threads", 0),
        dry_run=dry_run,
        extra_solver_args=getattr(args, "solver_args", None) or [],
        **overrides,
    )
    return Runner(case, args.output, toolchain=tc, options=opts)


def cmd_gencase(args) -> int:
    case = _load(args.case)
    if case is None:
        return 1
    r = _runner(args, case, dry_run=True)
    r.prepare()
    print(f"XML : {r.layout.case_def}\nscript : {r.layout.root / 'run.sh'}")
    if args.layout:
        from .plotting import plot_case_layout

        png = r.layout.root / f"{case.name}_layout.png"
        plot_case_layout(case, png)
        print(f"schéma : {png}")
    return 0


def cmd_run(args) -> int:
    from .runner import RunError

    case = _load(args.case)
    if case is None:
        return 1
    rep = check_report(case)
    for w in rep["warnings"]:
        print(f"! {w}", file=sys.stderr)
    r = _runner(args, case, dry_run=args.dry_run, gencase=not args.skip_gencase,
                solve=not args.skip_solve, post=not args.no_post)
    if not args.dry_run:
        missing = [t for t in ("gencase", "dualsphysics_gpu" if args.gpu else "dualsphysics_cpu") if not r.toolchain.has(t)]
        if (not args.skip_gencase and "gencase" in missing) or (not args.skip_solve and any(m.startswith("dualsphysics") for m in missing)):
            _err("exécutables DualSPHysics introuvables :\n" + r.toolchain.report())
            _err("définissez DUALSPHYSICS_HOME ou utilisez --dry-run pour préparer le cas.")
            return 2
    try:
        results = r.run()
    except RunError as exc:
        _err(str(exc))
        return 3
    total = sum(s.duration for s in results)
    print(f"{len(results)} étape(s) terminée(s) en {total:.1f} s — résultats dans {r.layout.out}")
    if not args.dry_run and not args.no_post and case.gauges:
        print(f"analyse : wavesimu analyze {r.layout.root}")
    return 0


def _case_in_workdir(workdir: Path, explicit: Optional[str]) -> Optional[Case]:
    if explicit:
        return _load(explicit)
    yamls = sorted(workdir.glob("*.yaml"))
    if not yamls:
        _err(f"aucun fichier de cas (.yaml) dans {workdir} ; précisez --case")
        return None
    return _load(str(yamls[0]))


def cmd_post(args) -> int:
    from .runner import RunError

    workdir = Path(args.workdir)
    case = _case_in_workdir(workdir, args.case)
    if case is None:
        return 1
    args.output = str(workdir)
    r = _runner(args, case, dry_run=args.dry_run, gencase=False, solve=False)
    try:
        results = [r._run_step(s) for s in r.plan()]
    except RunError as exc:
        _err(str(exc))
        return 3
    print(f"{len(results)} étape(s) de post-traitement terminée(s)")
    return 0


def cmd_analyze(args) -> int:
    from .analysis import analyze_gauge, compare_to_target, reflection_goda, suggested_window
    from .postprocess import load_elevations, parse_run_out
    from .runner import RunLayout

    workdir = Path(args.workdir)
    case = _case_in_workdir(workdir, args.case)
    if case is None:
        return 1
    lay = RunLayout(workdir, case.name)
    w = case.wavemaker.waves
    th = summarize(w.height, w.period, case.flume.depth, case.physics.gravity)
    out: Dict = {"case": case.name, "gauges": [], "warnings": []}

    if lay.run_out.exists():
        ri = parse_run_out(lay.run_out)
        out["run"] = {"finished": ri.finished, "runtime_s": ri.runtime, "last_time": ri.last_time, "n_parts": ri.n_parts, "particles": ri.particles}
        out["warnings"] += ri.warnings[:10]
    try:
        es = load_elevations(workdir, case.name, [g.name for g in case.gauges], swl=case.flume.depth)
    except FileNotFoundError as exc:
        _err(str(exc))
        return 1
    if len(es.time) < 4:
        _err("séries d'élévation vides : le post-traitement a-t-il été exécuté ?")
        return 1
    t_end = float(es.time[-1])
    gauge_x = {g.name: g.x for g in case.gauges}
    reports = []
    irregular = w.kind == "irregular"
    for name in es.names:
        x = gauge_x.get(name, 0.0)
        t0, t1 = suggested_window(t_end, w.period, th.wavelength, x, th.celerity, w.ramp if not irregular else w.ramp / w.period)
        if args.t_start is not None:
            t0 = args.t_start
        if args.t_end is not None:
            t1 = args.t_end
        try:
            rep = analyze_gauge(name, es.time, es[name], t0, t1)
        except ValueError as exc:
            out["warnings"].append(str(exc))
            continue
        reports.append(rep)
        d = rep.as_dict()
        d.update(compare_to_target(rep, w.height, w.period, irregular))
        d["x"] = x
        out["gauges"].append(d)

    # réflexion (deux premières sondes hors amortissement, à moins de L/2)
    if len(reports) >= 2 and not irregular:
        pairs = [(a, b) for a in reports for b in reports if gauge_x.get(a.name, 0) < gauge_x.get(b.name, 0)]
        best = None
        for a, b in pairs:
            dl = gauge_x[b.name] - gauge_x[a.name]
            kdl = th.wavenumber * dl
            if 0.05 * math.pi < kdl < 0.45 * math.pi:
                score = abs(kdl - 0.25 * math.pi)
                if best is None or score < best[0]:
                    best = (score, a, b, dl)
        if best:
            _, a, b, dl = best
            t0 = max(a.t_start, b.t_start)
            ta, ea = _window(es.time, es[a.name], t0, None)
            _, eb = _window(es.time, es[b.name], t0, None)
            r = reflection_goda(ta, ea, eb, dl, case.flume.depth, case.physics.gravity,
                                fmin=0.5 / w.period, fmax=3.0 / w.period, period=w.period)
            out["reflection"] = {"gauges": [a.name, b.name], "dl": dl, **r.as_dict()}

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=_json_default))
    else:
        _print_analysis(out, w, irregular)

    if args.plot:
        from .plotting import plot_elevations, plot_spectra, plot_wave_heights

        figdir = workdir / "figures"
        plot_elevations(es.time, es.elevation, w.height, figdir / "elevations.png")
        plot_spectra({r.name: r.spectral for r in reports}, w.period, figdir / "spectra.png", fmax=4.0 / w.period)
        if reports:
            plot_wave_heights(reports, w.height, figdir / "heights.png")
        print(f"figures dans {figdir}")
    if args.save:
        p = workdir / "analysis.json"
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        print(f"résultats enregistrés : {p}")
    return 0


def _window(t, eta, t0, t1):
    from .analysis import window

    return window(t, eta, t0, t1)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def _print_analysis(out: Dict, w, irregular: bool) -> None:
    print(f"Analyse du cas '{out['case']}'")
    if "run" in out:
        r = out["run"]
        status = "terminée" if r["finished"] else "incomplète"
        rt = f", {r['runtime_s']:.0f} s de calcul" if r.get("runtime_s") else ""
        print(f"  simulation {status} : t = {r['last_time']:.2f} s, {r['n_parts']} sorties{rt}")
    label = "Hs / Tp" if irregular else "H / T"
    print(f"  cible {label} : {w.height} m / {w.period} s")
    hdr = f"  {'sonde':<8}{'x [m]':>7}{'fenêtre [s]':>16}{'N':>5}{'Hmoy':>8}{'H1/3':>8}{'Hmax':>8}{'Hm0':>8}{'Tmoy':>7}{'Tp':>7}{'H/Hc':>7}"
    print(hdr)
    for g in out["gauges"]:
        print(f"  {g['name']:<8}{g['x']:>7.2f}{g['t_start']:>8.1f}{g['t_end']:>8.1f}{g['zc_n_waves']:>5d}"
              f"{g['zc_h_mean']:>8.4f}{g['zc_h_13']:>8.4f}{g['zc_h_max']:>8.4f}{g['sp_hm0']:>8.4f}"
              f"{g['zc_t_mean']:>7.2f}{g['sp_tp']:>7.2f}{g['height_ratio']:>7.2f}")
    if "reflection" in out:
        r = out["reflection"]
        print(f"  réflexion (Goda-Suzuki, {r['gauges'][0]}-{r['gauges'][1]}, dl = {r['dl']:.2f} m) : Kr = {r['kr']:.3f} "
              f"(Hi = {r['h_incident']:.4f} m, Hr = {r['h_reflected']:.4f} m, {r['n_freq']} fréq.)")
    for wmsg in out["warnings"]:
        print(f"  ! {wmsg}")


def cmd_theory(args) -> int:
    s = summarize(args.height, args.period, args.depth)
    if args.json:
        print(json.dumps(s.as_dict(), indent=2))
        return 0
    print(_fmt_table([
        ("hauteur H [m]", s.height), ("période T [s]", s.period), ("profondeur h [m]", s.depth),
        ("nombre d'onde k [1/m]", s.wavenumber), ("longueur d'onde L [m]", s.wavelength),
        ("célérité c [m/s]", s.celerity), ("vitesse de groupe cg [m/s]", s.group_velocity),
        ("cambrure H/L", s.steepness), ("h/L", s.depth / s.wavelength), ("régime", s.regime),
        ("nombre d'Ursell", s.ursell), ("H déferlement (Miche) [m]", s.breaking_height),
        ("course piston S [m]", s.piston_stroke), ("course volet S [m]", s.flap_stroke),
        ("dp recommandé (H/10) [m]", s.height / 10),
    ]))
    return 0


def cmd_doctor(args) -> int:
    from .tools import Toolchain

    tc = Toolchain.discover(args.dsph_home)
    print(tc.report())
    print()
    try:
        import matplotlib  # noqa: F401

        print("matplotlib : OK (figures disponibles)")
    except ImportError:
        print("matplotlib : absent (pip install 'wavesimu[plot]' pour les figures)")
    if tc.is_complete():
        print("\nInstallation DualSPHysics complète : `wavesimu run` est utilisable.")
        return 0
    print("\nExécutables requis manquants : définissez DUALSPHYSICS_HOME=<racine DualSPHysics> "
          "(https://dual.sphysics.org) ou DSPH_GENCASE / DSPH_DUALSPHYSICS_CPU.")
    return 1


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wavesimu", description="Simulation de vagues en canal à houle avec DualSPHysics.")
    p.add_argument("--version", action="version", version=f"wavesimu {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="crée un fichier de cas YAML depuis un modèle")
    s.add_argument("output", nargs="?", help="fichier YAML de sortie (défaut : <nom>.yaml)")
    s.add_argument("-t", "--template", default="regular", help="modèle (voir --list)")
    s.add_argument("-n", "--name", help="nom du cas")
    s.add_argument("--list", action="store_true", help="liste les modèles disponibles")
    s.add_argument("--force", action="store_true", help="écrase le fichier existant")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("check", help="valide un cas et affiche la houle théorique")
    s.add_argument("case", help="fichier de cas YAML/JSON")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_check)

    def run_args(s, output_required=True):
        s.add_argument("-o", "--output", default="runs", help="répertoire de travail (défaut : runs/)")
        s.add_argument("--dsph-home", help="racine de l'installation DualSPHysics (sinon $DUALSPHYSICS_HOME)")
        s.add_argument("--gpu", action="store_true", help="utilise l'exécutable GPU")
        s.add_argument("--threads", type=int, default=0, help="threads OpenMP (CPU), 0 = tous")
        s.add_argument("--solver-args", nargs="*", help="arguments supplémentaires pour DualSPHysics")

    s = sub.add_parser("gencase", help="écrit le XML GenCase et run.sh sans exécuter")
    s.add_argument("case")
    run_args(s)
    s.add_argument("--layout", action="store_true", help="enregistre un schéma PNG du canal")
    s.set_defaults(func=cmd_gencase)

    s = sub.add_parser("run", help="exécute la chaîne complète")
    s.add_argument("case")
    run_args(s)
    s.add_argument("--dry-run", action="store_true", help="prépare le cas et affiche les commandes sans les lancer")
    s.add_argument("--skip-gencase", action="store_true")
    s.add_argument("--skip-solve", action="store_true")
    s.add_argument("--no-post", action="store_true", help="n'exécute pas le post-traitement")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("post", help="post-traitement d'un run existant")
    s.add_argument("workdir")
    s.add_argument("--case", help="fichier de cas (défaut : premier .yaml du répertoire)")
    s.add_argument("--dsph-home")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_post, gpu=False, threads=0, solver_args=None)

    s = sub.add_parser("analyze", help="analyse les élévations aux sondes")
    s.add_argument("workdir")
    s.add_argument("--case")
    s.add_argument("--t-start", type=float, help="début de la fenêtre d'analyse [s]")
    s.add_argument("--t-end", type=float, help="fin de la fenêtre d'analyse [s]")
    s.add_argument("--plot", action="store_true", help="génère les figures (matplotlib)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--save", action="store_true", help="enregistre analysis.json dans le répertoire")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("theory", help="théorie linéaire : L, c, course du batteur…")
    s.add_argument("-H", "--height", type=float, required=True, help="hauteur de vague [m]")
    s.add_argument("-T", "--period", type=float, required=True, help="période [s]")
    s.add_argument("-d", "--depth", type=float, required=True, help="profondeur [m]")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_theory)

    s = sub.add_parser("doctor", help="vérifie l'installation DualSPHysics")
    s.add_argument("--dsph-home")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
