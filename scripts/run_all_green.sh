#!/usr/bin/env bash
# One command: build everything (cached), run every suite on Modal A100s in parallel, print ONE PASS/FAIL table.
#   scripts/run_all_green.sh [TAG] [--only exec,litmus,...] [--rebuild]
# Detached: the run survives this shell. Results: volume kvm-green:/TAG/{table.txt,table.json,<job>.log}.
# Fetch the table later:  modal volume get kvm-green /TAG/table.txt -
# Needs: modal CLI logged in. Cost: about 14 A100-hours worst case, binaries are cached by source hash so a re-run of an unchanged tree
# skips every nvcc build (about 40 min each, on CPU containers).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-green-$(date -u +%Y%m%d-%H%M)}"; [ $# -gt 0 ] && shift
ARGS=()
while [ $# -gt 0 ]; do case "$1" in --only) ARGS+=(--only "$2"); shift 2;; --rebuild) ARGS+=(--rebuild); shift;; *) echo "unknown option $1" >&2; exit 2;; esac; done
echo "tag: $TAG"
modal run --detach scripts/green_all.py --tag "$TAG" ${ARGS[@]+"${ARGS[@]}"}
echo "table: modal volume get kvm-green /$TAG/table.txt -"
