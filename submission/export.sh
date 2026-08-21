#!/usr/bin/env bash
# Export the image as the .tar.gz that goes into the submission zip.
set -euo pipefail
IMAGE="${IMAGE:-psmareg_lapirn}"
OUT="${OUT:-$(dirname "${BASH_SOURCE[0]}")/${IMAGE}.tar.gz}"
docker save "$IMAGE" | gzip -c > "$OUT"
ls -lh "$OUT"
