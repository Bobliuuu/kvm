#!/usr/bin/env bash
# gpt_162m, 1 optimizer step, executor ms/step for each (tiling bundle, executor build) pair, interleaved within ONE GPU-lock hold so the
# ratios come from the same minutes of the same box (best of REPS launches per pair), plus a task trace of the last pair.
#   flock -w 7200 ~/.gpu.lock bash bench/vs_pytorch/tiles162.sh OUT_DIR "BUNDLE_DIR..." "EXEC..." [REPS]
# BUNDLE_DIR: made by `gen.py DIR --model gpt_162m --steps 1 --tiling PRESET` (a100_mm64 | a100 | a100_trunc); EXEC: kvm_exec builds
# (e.g. -DKVM_MIN_BLOCKS=1 / 2). Each pair writes OUT_DIR/<bundle>__<exec>.{json,log}.
set -uo pipefail
OUT="${1:?OUT_DIR}"; BUNDLES="${2:?bundle dirs}"; EXECS="${3:?executables}"; REPS="${4:-5}"
mkdir -p "$OUT"
for round in 1 2; do
  for b in $BUNDLES; do
    for e in $EXECS; do
      tag="$(basename "$b")__$(basename "$e")"
      echo "== round $round $tag $(date -u +%T)"
      python3 bench/vs_pytorch/exec_bench.py "$b" "$e" --reps "$REPS" --out "$OUT/$tag.r$round.json" 2>&1 | tee "$OUT/$tag.r$round.log"
    done
  done
done
echo "== done $(date -u +%T)"
