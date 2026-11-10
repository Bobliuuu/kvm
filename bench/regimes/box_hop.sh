#!/usr/bin/env bash
# ON THE GPU BOX: run the prebuilt hop_bench variants back to back under ONE short lock hold. Usage: box_hop.sh OUT_FILE [cases]
set -uo pipefail
OUT="$1"; CASES="${2:-1:1000,1:100,4:250,16:250,64:100,108:100,256:100}"
{ nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; echo "commit $(cat COMMIT)"; echo "cases $CASES"; } > "$OUT"
flock -w "${LOCK_WAIT:-3600}" ~/.gpu.lock timeout "${RUN_TIMEOUT:-600}" bash -c "for b in ${BINS:-hop_base hop_chain hop_base hop_chain}; do echo \"== \$b\"; ./\$b --cases $CASES; done" >> "$OUT" 2>&1
echo "exit $?" >> "$OUT"
