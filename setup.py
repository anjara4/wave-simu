"""Shim de compatibilité pour les anciennes versions de pip (< 21.3) qui ne
gèrent pas l'installation éditable depuis pyproject.toml seul."""

from setuptools import setup

setup()
