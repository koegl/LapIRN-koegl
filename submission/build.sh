#!/usr/bin/env bash
# Build the submission image. Run from anywhere; the build context is the repo
# root, which is what lets the Dockerfile COPY Code/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-psmareg_lapirn}"
cd "$REPO_ROOT"
docker build -f submission/Dockerfile -t "$IMAGE" .
echo "built $IMAGE from $(git rev-parse --short HEAD)"
