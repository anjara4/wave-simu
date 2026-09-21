"""Génération du fichier de définition ``<cas>_Def.xml`` pour GenCase.

Le XML produit suit le format de DualSPHysics v5.x : section ``casedef``
(constantes, géométrie construite par commandes de dessin, mouvements) et
section ``execution`` (batteurs à houle, amortissement, sondes, corps
flottants, paramètres numériques).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from . import __version__
from .config import Case, Structure, resolve_damping
from .theory import wavelength

# mk utilisés par convention
MK_WALLS = 0  #: parois et fond du canal (mkbound)
MK_FLUID = 0  #: eau (mkfluid)
MK_STRUCT_START = 20  #: premier mk pour les structures


def _fmt(v: float) -> str:
    """Formate un flottant sans zéros inutiles."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    s = f"{v:.10g}"
    return s


def _sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        if v is None:
            continue
        el.set(k, v if isinstance(v, str) else _fmt(v))
    return el


def _point(parent: ET.Element, tag: str, x: float, y: float, z: float, **extra) -> ET.Element:
    return _sub(parent, tag, x=x, y=y, z=z, **extra)


def structure_mk(case: Case, index: int) -> int:
    s = case.structures[index]
    return s.mk if s.mk is not None else MK_STRUCT_START + index


class CaseGenerator:
    """Construit l'arbre XML d'un :class:`Case`."""

    def __init__(self, case: Case):
        self.case = case.validate()
        self.wavelength = wavelength(case.wavemaker.waves.period, case.flume.depth, case.physics.gravity)
        self.damping = resolve_damping(self.case, self.wavelength)

    # ------------------------------------------------------------------ API
    def build(self) -> ET.ElementTree:
        root = ET.Element("case", app=f"wavesimu {__version__}")
        casedef = ET.SubElement(root, "casedef")
        self._constants(casedef)
        _sub(casedef, "mkconfig", boundcount=240, fluidcount=9)
        self._geometry(casedef)
        self._motion(casedef)
        self._floatings(casedef)
        execution = ET.SubElement(root, "execution")
        self._special(execution)
        self._parameters(execution)
        return ET.ElementTree(root)

    def to_string(self) -> str:
        tree = self.build()
        ET.indent(tree, space="    ")
        body = ET.tostring(tree.getroot(), encoding="unicode")
        return '<?xml version="1.0" encoding="UTF-8" ?>\n' + body + "\n"

    def write(self, directory: Union[str, Path]) -> Path:
        """Écrit ``<name>_Def.xml`` dans ``directory`` et retourne son chemin."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.case.name}_Def.xml"
        path.write_text(self.to_string(), encoding="utf-8")
        return path

    # ------------------------------------------------------------ sections
    def _constants(self, parent: ET.Element) -> None:
        c = self.case
        ph = c.physics
        cd = ET.SubElement(parent, "constantsdef")
        _sub(cd, "lattice", bound=1, fluid=1)
        _point(cd, "gravity", 0, 0, -ph.gravity, comment="Gravitational acceleration", units_comment="m/s^2")
        _sub(cd, "rhop0", value=ph.rhop0, comment="Reference density of the fluid", units_comment="kg/m^3")
        _sub(cd, "hswl", value=0, auto="true", comment="Maximum still water level to calculate speedofsound using coefsound", units_comment="metres (m)")
        _sub(cd, "gamma", value=ph.gamma, comment="Polytropic constant for water used in the state equation")
        _sub(cd, "speedsystem", value=0, auto="true", comment="Maximum system speed (by default the dam-break propagation is used)")
        _sub(cd, "coefsound", value=ph.coefsound, comment="Coefficient to multiply speedsystem")
        _sub(cd, "speedsound", value=0, auto="true", comment="Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem)")
        _sub(cd, "coefh", value=ph.coefh, comment="Coefficient to calculate the smoothing length (h=coefh*sqrt(3*dp^2) in 3D)")
        _sub(cd, "cflnumber", value=ph.cfl, comment="Coefficient to multiply dt")

    def back_x(self) -> float:
        """Abscisse arrière du canal : le fond est prolongé derrière le batteur
        pour que celui-ci ait toujours un plancher lors de son recul."""
        wm = self.case.wavemaker
        back = max(0.5, 2.0 * self.case.wavemaker.waves.height / max(1e-9, self._transfer()))
        return wm.x - wm.thickness - back

    def _transfer(self) -> float:
        from .theory import flap_transfer, piston_transfer, wavenumber

        c = self.case
        k = wavenumber(c.wavemaker.waves.period, c.flume.depth, c.physics.gravity)
        return piston_transfer(k, c.flume.depth) if c.wavemaker.kind == "piston" else flap_transfer(k, c.flume.depth)

    def _domain(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        c = self.case
        fl = c.flume
        margin = 0.2
        xmin = self.back_x() - margin
        xmax = fl.length + margin
        if c.is_2d:
            ymin = ymax = 0.0
        else:
            ymin, ymax = -margin, fl.width + margin
        zmin = -margin
        zmax = fl.height + margin
        return (xmin, ymin, zmin), (xmax, ymax, zmax)

    def _geometry(self, parent: ET.Element) -> None:
        c = self.case
        fl = c.flume
        geo = ET.SubElement(parent, "geometry")
        d = _sub(geo, "definition", dp=c.dp, comment="Initial inter-particle distance", units_comment="metres (m)")
        pmin, pmax = self._domain()
        _point(d, "pointmin", *pmin)
        _point(d, "pointmax", *pmax)
        commands = ET.SubElement(geo, "commands")
        ml = ET.SubElement(commands, "mainlist")
        ET.SubElement(ml, "setshapemode").text = "actual | dp | bound"
        _sub(ml, "setdrawmode", mode="full")

        # --- canal (parois) ---------------------------------------------
        _sub(ml, "setmkbound", mk=MK_WALLS)
        box = ET.SubElement(ml, "drawbox")
        faces = "bottom | right" if c.is_2d else "bottom | right | front | back"
        ET.SubElement(box, "boxfill").text = faces
        x0 = self.back_x()
        _point(box, "point", x0, 0, 0)
        _point(box, "size", fl.length - x0, fl.width, fl.height)

        # --- plage inclinée ----------------------------------------------
        if fl.beach is not None:
            b = fl.beach
            h_end = min(b.height_at(fl.length), fl.height)
            self._comment(ml, f"Plage inclinee : pente {b.slope} a partir de x={b.start}")
            _sub(ml, "setmkbound", mk=MK_WALLS)
            prism = ET.SubElement(ml, "drawprism", mask="0")
            ys = (-c.dp, c.dp) if c.is_2d else (0.0, fl.width)
            for y in ys:
                _point(prism, "point", b.start, y, 0)
                _point(prism, "point", fl.length, y, 0)
                _point(prism, "point", fl.length, y, h_end)

        # --- batteur -------------------------------------------------------
        wm = c.wavemaker
        self._comment(ml, f"Batteur {wm.kind} (mk={wm.mk})")
        _sub(ml, "setmkbound", mk=wm.mk)
        pb = ET.SubElement(ml, "drawbox")
        ET.SubElement(pb, "boxfill").text = "solid"
        _point(pb, "point", wm.x - wm.thickness, 0, 0)
        _point(pb, "size", wm.thickness, fl.width, fl.height)

        # --- structures ----------------------------------------------------
        for i, s in enumerate(c.structures):
            self._structure(ml, s, structure_mk(c, i))

        # --- eau -------------------------------------------------------------
        self._comment(ml, f"Eau au repos, profondeur {fl.depth} m")
        _sub(ml, "setmkfluid", mk=MK_FLUID)
        fb = _sub(ml, "fillbox", x=wm.x + wm.thickness + 2 * c.dp, y=0 if c.is_2d else 0.5 * fl.width, z=0.5 * fl.depth)
        ET.SubElement(fb, "modefill").text = "void"
        _point(fb, "point", wm.x, 0, 0)
        _point(fb, "size", fl.length - wm.x, fl.width, fl.depth)
        _sub(ml, "shapeout", file="")

    def _structure(self, ml: ET.Element, s: Structure, mk: int) -> None:
        c = self.case
        self._comment(ml, f"Structure '{s.name}' ({s.kind}, mk={mk})")
        _sub(ml, "setmkbound", mk=mk)
        ox, oy, oz = s.origin
        if s.kind in ("box", "floating_box"):
            sx, sy, sz = s.size
            if c.is_2d:
                oy, sy = 0.0, 0.0
            box = ET.SubElement(ml, "drawbox")
            ET.SubElement(box, "boxfill").text = "solid"
            _point(box, "point", ox, oy, oz)
            _point(box, "size", sx, sy, sz)
        elif s.kind == "cylinder":
            height = s.size[2]
            cyl = _sub(ml, "drawcylinder", radius=s.radius)
            _point(cyl, "point", ox, oy, oz)
            _point(cyl, "point", ox, oy, oz + height)

    def _comment(self, parent: ET.Element, text: str) -> None:
        parent.append(ET.Comment(f" {text} "))

    def _motion(self, parent: ET.Element) -> None:
        wm = self.case.wavemaker
        motion = ET.SubElement(parent, "motion")
        obj = _sub(motion, "objreal", ref=wm.mk)
        _sub(obj, "begin", mov=1, start=0)
        _sub(obj, "mvnull", id=1)

    # ---------------------------------------------------------- execution
    def _special(self, parent: ET.Element) -> None:
        special = ET.SubElement(parent, "special")
        self._wavepaddle(special)
        self._damping(special)
        self._gauges(special)

    def _wavepaddle(self, parent: ET.Element) -> None:
        c = self.case
        wm = c.wavemaker
        w = wm.waves
        wp = ET.SubElement(parent, "wavepaddles")
        irregular = w.kind == "irregular"
        tag = wm.kind + ("_spectrum" if irregular else "")
        p = ET.SubElement(wp, tag)
        _sub(p, "mkbound", value=wm.mk, comment="Mk-Bound of selected particles")
        _sub(p, "waveorder", value=w.order, comment="Order wave generation 1:1st order, 2:2nd order (def=1)")
        _sub(p, "start", value=w.start, comment="Start time (def=0)")
        _sub(p, "duration", value=w.duration, comment="Movement duration, Zero is the end of simulation (def=0)")
        _sub(p, "depth", value=c.flume.depth, comment="Fluid depth in front of the paddle (def=0)")
        if irregular:
            _sub(p, "fixeddepth", value=0, comment="Fluid depth without paddle (def=0)")
        if wm.kind == "piston":
            _point(p, "pistondir", 1, 0, 0, comment="Movement direction (def=(1,0,0))")
        else:
            _sub(p, "variabledraft", value=wm.hinge, comment="Position of the wavemaker hinge (above the bottom <0; below the bottom >0) (default=0)")
            _point(p, "flapaxis0", wm.x, -1, 0, comment="Point 0 of axis rotation")
            _point(p, "flapaxis1", wm.x, 1, 0, comment="Point 1 of axis rotation")
        if irregular:
            _sub(p, "spectrum", value=w.spectrum, comment="Spectrum type: jonswap,pierson-moskowitz")
            _sub(p, "discretization", value=w.discretization, comment="Spectrum discretization: regular,random,stretched,cosstretched (def=stretched)")
            _sub(p, "waveheight", value=w.height, comment="Significant Wave Height")
            _sub(p, "waveperiod", value=w.period, comment="Peak Wave Period")
            _sub(p, "peakcoef", value=w.peak_coef, comment="Peak enhancement coefficient (def=3.3)")
            _sub(p, "waves", value=w.n_waves, comment="Number of waves to create irregular waves (def=50)")
            _sub(p, "randomseed", value=w.seed, comment="Random seed to initialize a pseudorandom number generator")
            _sub(p, "serieini", value=0, autofit="true", comment="Initial time in irregular wave serie (default=0 and autofit=false)")
            _sub(p, "ramptime", value=w.ramp, comment="Time of ramp (def=0)")
            if wm.save_motion:
                _sub(p, "savemotion", time=c.time.end, timedt=c.time.output_dt, xpos=2 * self.wavelength, zpos=-c.flume.depth,
                     comment="Saves motion data. xpos and zpos are optional. zpos=-depth of the measuring point")
                _sub(p, "saveserie", timemin=0, timemax=c.time.end, timedt=c.time.output_dt, xpos=0, comment="Saves serie data (optional)")
        else:
            _sub(p, "waveheight", value=w.height, comment="Wave height")
            _sub(p, "waveperiod", value=w.period, comment="Wave period")
            _sub(p, "phase", value=w.phase, comment="Initial wave phase in function of PI (def=0)")
            _sub(p, "ramp", value=w.ramp, comment="Periods of ramp (def=0)")
            if wm.save_motion:
                _sub(p, "savemotion", periods=2, periodsteps=20, xpos=2 * self.wavelength, zpos=-c.flume.depth,
                     comment="Saves motion data. xpos and zpos are optional. zpos=-depth of the measuring point")
        if wm.kind == "piston" and wm.awas.enabled:
            self._awas(p)

    def _awas(self, parent: ET.Element) -> None:
        c = self.case
        a = c.wavemaker.awas
        w = c.wavemaker.waves
        el = ET.SubElement(parent, "awas_zsurf")
        start = a.start if a.start is not None else (w.ramp * w.period if w.kind == "regular" else w.ramp)
        _sub(el, "startawas", value=start, comment="Time to start AWAS correction (def=ramp*waveperiod)")
        _sub(el, "swl", value=c.flume.depth, comment="Still water level (free-surface water)")
        _sub(el, "elevation", value=w.order, comment="Order wave to calculate elevation 1:1st order, 2:2nd order (def=2)")
        _sub(el, "gaugex", valueh=a.gauge_x, comment="Position in X from piston to measure free-surface water (def=5*Dp)")
        _sub(el, "gaugey", value=0 if c.is_2d else 0.5 * c.flume.width, comment="Position in Y to measure free-surface water")
        _sub(el, "gaugezmin", value=0, comment="Minimum position in Z to measure free-surface water, it must be in water (def=domain limits)")
        _sub(el, "gaugezmax", value=c.flume.height, comment="Maximum position in Z to measure free-surface water (def=domain limits)")
        _sub(el, "gaugedp", value=0.1, comment="Resolution to measure free-surface water, it uses Dp*gaugedp (def=0.1)")
        _sub(el, "coefmasslimit", value=0.4 if c.is_2d else 0.5, comment="Coefficient to calculate mass of free-surface (def=0.5 on 3D and 0.4 on 2D)")
        _sub(el, "savedata", value=1, comment="Saves CSV with information 1:by part, 2:more info 3:by step (def=0)")
        _sub(el, "limitace", value=a.limit_acc, comment="Factor to limit maximum value of acceleration, with 0 disabled (def=2)")
        if a.correction:
            _sub(el, "correction", coefstroke=a.coef_stroke, coefperiod=a.coef_period, powerfunc=a.power_func,
                 comment="Drift correction configuration (def=no applied)")

    def _damping(self, parent: ET.Element) -> None:
        if self.damping is None:
            return
        c = self.case
        d = c.damping
        dz = ET.SubElement(ET.SubElement(parent, "damping"), "dampingzone")
        y = 0 if c.is_2d else 0.5 * c.flume.width
        _point(dz, "limitmin", self.damping["start"], y, 0, comment="Location where minimum reduction is applied")
        _point(dz, "limitmax", self.damping["end"], y, 0, comment="Location where maximum reduction is applied")
        _sub(dz, "overlimit", value=d.overlimit, comment="The scale of maximum reduction over limitmax (def=1)")
        _sub(dz, "redumax", value=d.redumax, comment="Maximum reduction in velocity (def=10)")
        fx, fy, fz = d.factor
        _point(dz, "factorxyz", fx, fy, fz, comment="Application factor in components (def=(1,1,1))")

    def _gauges(self, parent: ET.Element) -> None:
        c = self.case
        if not c.gauges:
            return
        gauge_dt = c.time.gauge_dt if c.time.gauge_dt is not None else c.time.output_dt
        g = ET.SubElement(parent, "gauges")
        dflt = ET.SubElement(g, "default")
        _sub(dflt, "savevtkpart", value="false", comment="Creates VTK files for each PART (default=false)")
        _sub(dflt, "computedt", value=gauge_dt, comment="Time between measurements. 0:all steps (default=TimeOut)", units_comment="s")
        _sub(dflt, "output", value="true", comment="Creates CSV files of measurements (default=false)")
        _sub(dflt, "outputdt", value=gauge_dt, comment="Time between output data (default=TimeOut)", units_comment="s")
        for gauge in c.gauges:
            y = gauge.y if not c.is_2d else 0.0
            swl = _sub(g, "swl", name=gauge.name, comment="Calculates surface water level (SWL)")
            _point(swl, "point0", gauge.x, y, 0, comment="Initial point", units_comment="m")
            _point(swl, "point2", gauge.x, y, c.flume.height, comment="Final point", units_comment="m")
            _sub(swl, "pointdp", valuedp=0.5, comment="Distance between check points (valuedp in Dp units)")
            _sub(swl, "masslimit", coefdp=0.4 if c.is_2d else 0.5, comment="Mass value to detect fluid particles (coefdp in Dp units)")

    def _floatings(self, parent: ET.Element) -> None:
        c = self.case
        floats = [(i, s) for i, s in enumerate(c.structures) if s.kind == "floating_box"]
        if not floats:
            return
        fl = ET.SubElement(parent, "floatings")
        for i, s in floats:
            f = _sub(fl, "floating", mkbound=structure_mk(c, i))
            _sub(f, "rhopbody", value=s.density, comment="Density of the body")

    def _parameters(self, parent: ET.Element) -> None:
        c = self.case
        ph = c.physics
        t = c.time
        params = ET.SubElement(parent, "parameters")

        def p(key, value, comment, units=None):
            _sub(params, "parameter", key=key, value=value, comment=comment, units_comment=units)

        p("SavePosDouble", 0, "Saves particle position using double precision (default=0)")
        p("Boundary", 1 if ph.boundary == "dbc" else 2, "Boundary method 1:DBC, 2:mDBC (default=1)")
        p("StepAlgorithm", 1 if ph.step_algorithm == "verlet" else 2, "Step Algorithm 1:Verlet, 2:Symplectic (default=1)")
        p("VerletSteps", 40, "Verlet only: Number of steps to apply Euler timestepping (default=40)")
        p("Kernel", 1 if ph.kernel == "cubic" else 2, "Interaction Kernel 1:Cubic Spline, 2:Wendland (default=2)")
        p("ViscoTreatment", 1 if ph.visco_treatment == "artificial" else 2, "Viscosity formulation 1:Artificial, 2:Laminar+SPS (default=1)")
        p("Visco", ph.viscosity, "Viscosity value")
        p("ViscoBoundFactor", ph.visco_bound_factor, "Multiply viscosity value with boundary (default=1)")
        p("DensityDT", ph.density_diffusion, "Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)")
        p("DensityDTvalue", ph.density_diffusion_value, "DDT value (default=0.1)")
        p("Shifting", ph.shifting, "Shifting mode 0:None, 1:Ignore bound, 2:Ignore fixed, 3:Full (default=0)")
        p("ShiftCoef", ph.shift_coef, "Coefficient for shifting computation (default=-2)")
        p("ShiftTFS", ph.shift_tfs, "Threshold to detect free surface. Typically 1.5 for 2D and 2.75 for 3D (default=0)")
        p("RigidAlgorithm", ph.rigid_algorithm, "Rigid Algorithm 1:SPH, 2:DEM, 3:CHRONO (default=1)")
        p("FtPause", 0.0, "Time to freeze the floatings at simulation start (warmup) (default=0)", "seconds")
        p("CoefDtMin", 0.05, "Coefficient to calculate minimum time step dtmin=coefdtmin*h/speedsound (default=0.05)")
        p("DtAllParticles", 0, "Velocity of particles used to calculate DT. 1:All, 0:Only fluid/floating (default=0)")
        p("TimeMax", t.end, "Time of simulation", "seconds")
        p("TimeOut", t.output_dt, "Time out data", "seconds")
        p("PartsOutMax", 1, "%/100 of fluid particles allowed to be excluded from domain (default=1)", "decimal")
        p("RhopOutMin", 700, "Minimum rhop valid (default=700)", "kg/m^3")
        p("RhopOutMax", 1300, "Maximum rhop valid (default=1300)", "kg/m^3")
        dom = _sub(params, "simulationdomain", comment="Defines domain of simulation (default=Uses minimum and maximum position of the generated particles)")
        _sub(dom, "posmin", x="default", y="default", z="default", comment="e.g.: x=0.5, y=default-1, z=default-10%")
        _sub(dom, "posmax", x="default", y="default", z="default+50%")


def generate_case_xml(case: Case) -> str:
    """Retourne le contenu XML ``<cas>_Def.xml`` pour un :class:`Case`."""
    return CaseGenerator(case).to_string()


def write_case(case: Case, directory: Union[str, Path]) -> Path:
    """Écrit le XML du cas dans ``directory`` et retourne son chemin."""
    return CaseGenerator(case).write(directory)


def gauge_points_file(case: Case, path: Union[str, Path], dz: Optional[float] = None) -> Path:
    """Écrit un fichier de points pour ``MeasureTool -elevation``.

    Pour chaque sonde, une colonne verticale de points entre ``z=0`` et le
    haut du canal est écrite (format ``POINTS`` : une ligne ``x y z`` par
    point).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dz = dz if dz is not None else 0.5 * case.dp
    n = int(math.ceil(case.flume.height / dz)) + 1
    lines = ["POINTS"]
    for g in case.gauges:
        y = 0.0 if case.is_2d else g.y
        for i in range(n):
            lines.append(f"{g.x:.6f} {y:.6f} {i * dz:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["CaseGenerator", "generate_case_xml", "gauge_points_file", "structure_mk", "write_case"]
