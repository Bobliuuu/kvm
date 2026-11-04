#!/usr/bin/env bash
# Generate every case's bundle + CPU golden (CPU only, no GPU needed), JOBS cases in parallel (default 8):
#   examples/models/prebuild.sh OUT_DIR [cases.txt]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?usage: prebuild.sh OUT_DIR [cases.txt]}"; CASES="${2:-$ROOT/examples/models/cases.txt}"; JOBS="${JOBS:-8}"
trim() { sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
while IFS='|' read -r name model steps mode tol flags; do
  name="$(echo "$name" | trim)"; case "$name" in ''|'#'*) continue;; esac
  model="$(echo "$model" | trim)"; steps="$(echo "$steps" | trim)"; mode="$(echo "$mode" | trim)"
  [ -f "$OUT/$name/bundle/golden.json" ] && { echo "$name: exists"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
  python3 "$ROOT/examples/models/gen_bundle.py" "$OUT/$name" --model "$model" --steps "$steps" $([ "$mode" = precise ] && echo --precise) $flags 2>&1 | grep -v '^INFO' | tail -1 &
done < "$CASES"
wait
