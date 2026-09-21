# syntax=docker/dockerfile:1
# Image wavesimu + DualSPHysics v5.4 (CPU) prête à l'emploi.
#
#   docker build -t wavesimu -f docker/Dockerfile .
#   docker run --rm -v "$PWD:/work" wavesimu run mon_cas.yaml -o runs/essai1
#
# Étape 1 : récupération des outils précompilés (GenCase, PartVTK, MeasureTool,
# IsoSurface) et compilation du solveur CPU depuis les sources officielles.
ARG DSPH_VERSION=v5.4.0

FROM ubuntu:24.04 AS builder
ARG DSPH_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone --depth 1 --branch "${DSPH_VERSION}" --filter=blob:none --sparse \
        https://github.com/DualSPHysics/DualSPHysics DualSPHysics \
    && cd DualSPHysics \
    && git sparse-checkout set bin/linux src/source src/lib
# gcc >= 13 exige <cstdint> ; le module d'amarrage MoorDynPlus est désactivé
# (non nécessaire pour la génération de houle).
RUN cd DualSPHysics/src/source && make -f Makefile_cpu -j"$(nproc)" \
        COMPILE_MOORDYNPLUS=NO \
        CCFLAGS="-c -O3 -fopenmp -std=c++0x -D_WITHMR -ffast-math -include cstdint -DDISABLE_MOORDYNPLUS" \
    && ls -la ../../bin/linux \
    && rm -rf ../../src

# Étape 2 : image d'exécution légère.
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/DualSPHysics/bin /opt/DualSPHysics/bin
ENV DUALSPHYSICS_HOME=/opt/DualSPHysics \
    LD_LIBRARY_PATH=/opt/DualSPHysics/bin/linux \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1
WORKDIR /opt/wavesimu
COPY pyproject.toml setup.py README.md ./
COPY wavesimu ./wavesimu
RUN pip install --no-cache-dir ".[plot]"
# Le répertoire de travail de l'utilisateur est monté sur /work.
WORKDIR /work
ENTRYPOINT ["wavesimu"]
CMD ["--help"]
