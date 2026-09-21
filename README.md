# wavesimu — simulation de vagues avec DualSPHysics

`wavesimu` est un outil Python pour **simuler des vagues dans un canal à houle**
avec le code SPH [DualSPHysics](https://dual.sphysics.org). Il couvre toute la
chaîne : description du cas en YAML, dimensionnement par la théorie linéaire,
génération du XML GenCase, exécution du solveur (CPU ou GPU), post-traitement
(surface libre, sondes, forces) et analyse de la houle (hauteurs, périodes,
spectre, réflexion).

```
YAML (cas)  ──►  wavesimu check   : validation + théorie linéaire (L, c, course du batteur…)
            ──►  wavesimu run     : GenCase ▸ DualSPHysics ▸ PartVTK / IsoSurface / MeasureTool / ComputeForces
            ──►  wavesimu analyze : H1/3, Hmax, Hm0, Tp, coefficient de réflexion, figures
```

## Fonctionnalités

- **Canal 2D ou bassin 3D** : longueur, profondeur d'eau, hauteur des parois, largeur.
- **Batteur piston ou volet (flap)**, houle **régulière** (1er/2e ordre) ou **irrégulière**
  (JONSWAP, Pierson-Moskowitz), rampe de démarrage, **absorption active (AWAS)**.
- **Zone d'amortissement** numérique en fin de canal, **plage inclinée** (run-up, déferlement).
- **Sondes de niveau** (`<gauges>` de DualSPHysics et MeasureTool).
- **Structures** : obstacle fixe, cylindre vertical (3D), **caisson flottant** ; forces (ComputeForces)
  et mouvement (FloatingInfo).
- **Paramètres SPH** exposés : noyau, schéma temporel, viscosité, diffusion de densité,
  shifting, DBC/mDBC, CFL…
- **Mode `--dry-run`** : prépare tous les fichiers et un script `run.sh` pour lancer le cas
  sur une autre machine (cluster, poste GPU).
- **Analyse** : passage à zéro (H moy., H1/3, Hmax, T moy.), spectre (Hm0, Tp, Tm01, Tm02),
  réflexion par la méthode à deux sondes de Goda & Suzuki, comparaison à la houle cible,
  figures matplotlib.

## Installation

```bash
git clone https://github.com/anjara4/wave-simu.git
cd wave-simu
pip install -e ".[dev]"          # numpy, pyyaml, matplotlib, pytest
```

DualSPHysics n'est pas distribué avec l'outil. Téléchargez la version 5.x sur
<https://dual.sphysics.org>, décompressez-la et indiquez sa racine :

```bash
export DUALSPHYSICS_HOME=~/DualSPHysics_v5.4     # contient bin/linux ou bin/windows
wavesimu doctor                                   # vérifie que les exécutables sont trouvés
```

Un exécutable peut aussi être désigné individuellement, par exemple
`DSPH_GENCASE=/chemin/GenCase_linux64` ou `DSPH_DUALSPHYSICS_CPU=…`.

## Prise en main

```bash
# 1. Créer un cas à partir d'un modèle (regular, irregular, flap, beach, structure, floating, tank3d)
wavesimu init mon_canal.yaml --template regular

# 2. Vérifier le cas : longueur d'onde, course du batteur, nombre de particules, avertissements
wavesimu check mon_canal.yaml

# 3. Lancer la simulation (CPU par défaut, --gpu pour la version GPU)
wavesimu run mon_canal.yaml -o runs/essai1 --threads 8

# 4. Analyser les sondes et produire les figures
wavesimu analyze runs/essai1 --plot --save
```

Sans DualSPHysics installé, `wavesimu run … --dry-run` (ou `wavesimu gencase`) écrit
le XML, le fichier de sondes et `run.sh` ; il suffit ensuite de lancer `run.sh` là où
les exécutables sont disponibles, puis `wavesimu analyze` sur le répertoire.

### Exemple de fichier de cas

```yaml
name: flume_regular
dimension: 2d          # 2d | 3d
dp: 0.01               # distance inter-particules (m) — viser >= 10 particules par hauteur de vague
flume:
  length: 12.0         # longueur du canal (m)
  depth: 0.5           # niveau d'eau au repos (m)
  height: 1.0          # hauteur des parois (m)
  # beach: {start: 6.0, slope: 0.1}   # plage inclinée optionnelle
wavemaker:
  kind: piston         # piston | flap
  waves:
    kind: regular      # regular | irregular
    height: 0.10       # H (ou Hs)
    period: 1.5        # T (ou Tp)
    order: 2
    ramp: 1            # périodes de rampe
  awas: {enabled: true}
damping: {enabled: true, start: 8.5}
gauges:
  - {name: WG1, x: 2.0}
  - {name: WG2, x: 4.0}
  - {name: WG3, x: 4.5}
  - {name: WG4, x: 6.0}
structures:
  - {kind: box, name: reef, origin: [6.0, 0, 0], size: [0.6, 0, 0.3]}
physics: {viscosity: 0.01, density_diffusion: 2, kernel: wendland}
time: {end: 15.0, output_dt: 0.05}
```

Les fichiers du dossier [`examples/`](examples) montrent toutes les options.
Les clés inconnues et les valeurs incohérentes (batteur, profondeur, sondes hors
canal, houle déferlante…) sont rejetées avec un message explicite.

### Théorie linéaire

```bash
wavesimu theory -H 0.1 -T 1.5 -d 0.5
```

affiche la longueur d'onde, la célérité, la vitesse de groupe, la cambrure, le nombre
d'Ursell, la hauteur limite de déferlement (Miche), la course du piston (Biésel) et du
volet, et le `dp` recommandé.

## Arborescence d'un run

```
runs/essai1/
├── flume_regular.yaml            # copie du cas
├── flume_regular_Def.xml         # entrée GenCase
├── flume_regular_gauges.txt      # points MeasureTool (une colonne verticale par sonde)
├── run.sh                        # chaîne complète, réutilisable sur une autre machine
├── logs/                         # journal de chaque étape
├── wavesimu_run.json             # manifeste (commandes, codes retour, durées)
├── analysis.json / figures/      # produits par `wavesimu analyze --save --plot`
└── flume_regular_out/
    ├── data/                     # fichiers .bi4 de DualSPHysics
    ├── Run.out                   # journal du solveur
    ├── particles/                # PartFluid_XXXX.vtk (ParaView)
    ├── surface/                  # Surface_XXXX.vtk (surface libre)
    ├── measuretool/              # Elevation_Elevation.csv
    └── forces/                   # Forces_<structure>.csv / Floating_<structure>.csv
```

## Utilisation en Python

```python
from wavesimu import load_case
from wavesimu.runner import Runner, RunOptions
from wavesimu.postprocess import load_elevations
from wavesimu.analysis import analyze_gauge, reflection_goda

case = load_case("examples/flume_regular.yaml")
Runner(case, "runs/api", options=RunOptions(gpu=False)).run()

es = load_elevations("runs/api", case.name, [g.name for g in case.gauges])
rep = analyze_gauge("WG2", es.time, es["WG2"], t_start=6.0)
print(rep.stats.h_13, rep.spectral.tp)
kr = reflection_goda(es.time, es["WG2"], es["WG3"], dl=0.5, depth=case.flume.depth).kr
```

## Conseils de mise en données

- **Résolution** : `dp ≤ H/10` ; `wavesimu check` avertit si ce n'est pas le cas.
- **Longueur du canal** : au moins 3 longueurs d'onde entre batteur et zone d'amortissement.
- **Amortissement** : 1 à 2 longueurs d'onde ; ne placez pas de sonde dedans.
- **Réflexion** : deux sondes distantes de 0,1 à 0,4 L pour la méthode de Goda-Suzuki
  (`wavesimu analyze` choisit automatiquement la meilleure paire).
- **Durée** : au moins 10 périodes après l'arrivée du front d'onde à la dernière sonde.
- **3D** : le nombre de particules croît vite ; commencez avec un `dp` grossier et passez au GPU.

## Tests

```bash
pytest
```

Les tests n'ont pas besoin de DualSPHysics : la chaîne complète est vérifiée avec de
faux exécutables qui imitent les sorties (Run.out, CSV MeasureTool).

## Limites connues

- Le format XML suit DualSPHysics **v5.x** ; certaines balises (`<gauges>`, mDBC) exigent
  v5.0 ou plus récent.
- Les noms d'exécutables sont ceux de la distribution officielle
  (`GenCase_linux64`, `DualSPHysics5.4CPU_linux64`, …) ; utilisez `DSPH_<OUTIL>` si les vôtres diffèrent.
- La lecture des fichiers VTK/bi4 n'est pas incluse : utilisez ParaView pour la visualisation.
