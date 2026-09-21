"""wavesimu — outil de simulation de vagues basé sur DualSPHysics.

Le package couvre toute la chaîne :

* :mod:`wavesimu.config`      — description d'un cas (YAML) et validation ;
* :mod:`wavesimu.theory`      — théorie linéaire de la houle (dispersion, course du batteur…) ;
* :mod:`wavesimu.casegen`     — génération du fichier ``<cas>_Def.xml`` pour GenCase ;
* :mod:`wavesimu.tools`       — localisation des exécutables DualSPHysics ;
* :mod:`wavesimu.runner`      — exécution de GenCase / DualSPHysics / post-traitement ;
* :mod:`wavesimu.postprocess` — lecture des sorties (Run.out, CSV MeasureTool, sondes) ;
* :mod:`wavesimu.analysis`    — statistiques de houle (zero-crossing, spectre, réflexion) ;
* :mod:`wavesimu.plotting`    — figures matplotlib ;
* :mod:`wavesimu.cli`         — interface en ligne de commande ``wavesimu``.
"""

from .config import Case, load_case, save_case  # noqa: F401

__version__ = "0.1.0"
