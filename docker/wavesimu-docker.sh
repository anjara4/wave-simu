#!/usr/bin/env bash
# Lance wavesimu dans le conteneur Docker en montant le répertoire courant.
# Usage : docker/wavesimu-docker.sh run mon_cas.yaml -o runs/essai1
set -euo pipefail
IMAGE="${WAVESIMU_IMAGE:-wavesimu:latest}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE absente : construction (quelques minutes)..." >&2
  docker build -t "$IMAGE" -f "$(dirname "$0")/Dockerfile" "$(dirname "$0")/.."
fi
exec docker run --rm -it -v "$PWD:/work" -w /work "$IMAGE" "$@"
