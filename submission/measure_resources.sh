#!/usr/bin/env bash
# One pair, instrumented, to fill the TODOs in README.txt: peak GPU VRAM, peak
# container RAM, and peak CPU usage. Run it on an otherwise idle machine.
set -euo pipefail
# awk's %f must print "1.5", not "1,5" -- the README is read by the organizers.
export LC_ALL=C

IMAGE="${IMAGE:-psmareg_koegl}"
IMAGES_DIR="${IMAGES_DIR:-/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs}"
# snap Docker is confined to its own private /tmp, so a bind source there does not
# exist as far as the daemon is concerned -- keep the output dir under $HOME.
OUT="${OUT:-$HOME/psmareg_measure}"
ID="${ID:-0001}"
NAME=psmareg_measure
mkdir -p "$OUT"

# GPU baseline: other processes may already hold memory, so the container's peak
# is measured as the rise above whatever is resident before it starts.
GPU_BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "GPU already in use by others: ${GPU_BASE} MiB"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --name "$NAME" \
    --ipc=host \
    --memory 60g \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --user "$(id -u):$(id -g)" \
    --network=none \
    --mount "type=bind,source=${IMAGES_DIR},target=/app/input,readonly" \
    --mount "type=bind,source=${OUT},target=/app/output" \
    "$IMAGE" \
        "/app/input/PSMARegPSMA_${ID}_0000_00.nii.gz" \
        "/app/input/PSMARegPSMA_${ID}_0001_00.nii.gz" \
        "/app/input/PSMARegPSMA_${ID}_0000_01.nii.gz" \
        "/app/input/PSMARegPSMA_${ID}_0001_01.nii.gz" \
        "/app/output/disp_${ID}_00_${ID}_01.nii.gz" > "$OUT/run.log" 2>&1 &
RUN_PID=$!

MAX_GPU=0; MAX_MEM=0; MAX_CPU=0
while kill -0 "$RUN_PID" 2>/dev/null; do
    STATS=$(docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}}' "$NAME" 2>/dev/null || true)
    if [[ -n "$STATS" ]]; then
        CPU=$(awk '{gsub(/%/,"",$1); print $1}' <<<"$STATS")
        # MemUsage is "1.23GiB / 60GiB"; take the first field and normalise to MiB
        MEM=$(awk '{print $2}' <<<"$STATS" | awk '
            /GiB/ {gsub(/GiB/,""); print $1*1024; next}
            /MiB/ {gsub(/MiB/,""); print $1;      next}
            /KiB/ {gsub(/KiB/,""); print $1/1024; next}
            {print 0}')
        awk -v a="$CPU" -v b="$MAX_CPU" 'BEGIN{exit !(a>b)}' && MAX_CPU=$CPU
        awk -v a="$MEM" -v b="$MAX_MEM" 'BEGIN{exit !(a>b)}' && MAX_MEM=$MEM
    fi
    GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    (( GPU > MAX_GPU )) && MAX_GPU=$GPU
    sleep 0.5
done
wait "$RUN_PID" || echo "container exited non-zero -- see $OUT/run.log"

echo
echo "=== fill these into README.txt ==="
awk -v g="$MAX_GPU" -v b="$GPU_BASE" 'BEGIN{printf "GPU VRAM peak : %.1f GB  (%d MiB total minus %d MiB baseline)\n",(g-b)/1024,g,b}'
awk -v m="$MAX_MEM" 'BEGIN{printf "RAM peak      : %.1f GB\n", m/1024}'
awk -v c="$MAX_CPU" 'BEGIN{printf "CPU peak      : %.0f%% -> %.1f cores fully busy\n", c, c/100}'
echo "host has $(nproc) cores"
echo
tail -4 "$OUT/run.log"
