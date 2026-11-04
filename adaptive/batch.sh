#!/usr/bin/env bash
# Runs ON the GPU box, with the GPU lock already held by the caller (adaptive/run_adaptive.sh takes it ONCE per batch). From the scratch dir.
#   batch.sh GROUP      GROUP: smoke | main | multi | bw | suites | all
# Env: REPS (5) DUR seconds (24) OUT (runs) WAD  BATCH_TAG. Every run is followed by the Doom oracle comparison (every frame, raw bytes + fingerprint + state).
set -uo pipefail
cd "$(dirname "$0")/.."
GROUP="${1:?group}"; REPS="${REPS:-5}"; DUR="${DUR:-24}"; OUT="${OUT:-runs}"; WAD="${WAD:-doom/third_party/doom1.wad}"
FPS="${FPS:-35}"
declare -A CFG
CFG[global]="--sched global"
CFG[static]="--sched static --shares 1,3,6"
CFG[fixed_fair]="--sched dynamic --weights 1,1,1"
CFG[fixed_prio]="--sched dynamic --weights 8,4,1 --min 1,0,0 --urgent-us 28571,0,0"
CFG[fixed_resv_peak]="--sched dynamic --weights 4,4,1 --min 2,40,0 --urgent-us 28571,0,0"
CFG[v1_demand]="--sched dynamic --weights 4,4,1 --urgent-us 28571,0,0 --ctl v1 --doom-min-hi 8"
CFG[adaptive]="--sched dynamic --weights 2,1,1 --ctl adaptive --target-wait-us 1000 --floor-pct 10 --doom-min-hi 8"
GATE="${GATE:---gate-serve 20000 --gate-train 25000}"   # tasks in flight per class; the live window is --live-cap (65536)
CFG[adaptive_gated]="${CFG[adaptive]} $GATE"
CFG[fixed_prio_gated]="${CFG[fixed_prio]} $GATE"
CFG[global_gated]="${CFG[global]} $GATE"
CFG[adaptive_tuned]="${CFG[adaptive]} $GATE --gate-tune"
CFG[alone]="--sched dynamic --weights 2,1,1 --no-train --no-serve"
# the phased load: serve burst, training pause, Doom scene change (extra work in every frame)
PHASES="--serve-rate 25 --burst 8 12 400 --pause-train 14 17 --scene 18 22 12000"
ORACLE_FRAMES=$(( DUR * FPS + 40 )); [ -d oracle ] || python3 doom/cpu_run.py --frames "$ORACLE_FRAMES" --out oracle --log warn

one() {   # name rep extra-args...   -> $OUT/$name/rep$rep
  local name="$1" rep="$2"; shift 2; local d="$OUT/$name/rep$rep"; rm -rf "$d"; mkdir -p "$d"
  echo "=== $(date -u +%T) $name rep$rep: adaptive_run scn --out $d $*"
  ${BIN:-./adaptive_run} "${SCN:-scn}" --out "$d" --live-cap ${LIVE_CAP:-65536} --ring-words ${RING_WORDS:-4194304} --wad "$WAD" --tag "$name-rep$rep" --duration-s "$DUR" --seed "$rep" "$@" > "$d/run.log" 2>&1; local rc=$?
  grep -E "doom:|FAIL|ERROR|error" "$d/run.log" | tail -3
  local ok=1
  for k in "$d"/doom[0-9]*; do [ -d "$k" ] || continue
    python3 doom/compare.py oracle "$k" > "$k/compare.txt" 2>&1 || ok=0; head -1 "$k/compare.txt" | sed "s|^|  $(basename $k): |"; done
  echo "  exit $rc, oracle $([ $ok = 1 ] && echo PASS || echo FAIL)"; echo "$rc $ok" > "$d/status"
  rm -f "$d"/doom*/frames.bin "$d/trace.csv.gz" 2>/dev/null; gzip -f "$d/trace.csv" 2>/dev/null || true
}
echo "# batch $GROUP @ $(date -u +%FT%TZ)  commit ${BATCH_COMMIT:-?}  reps $REPS dur $DUR  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
case "$GROUP" in
  smoke) mkdir -p "$OUT"; "$0" suites; REPS=1 "$0" main; REPS=1 "$0" bw;;
  main)  for rep in $(seq 1 "$REPS"); do for c in global global_gated static fixed_fair fixed_prio fixed_prio_gated fixed_resv_peak v1_demand adaptive adaptive_gated; do one $c $rep ${CFG[$c]} $PHASES; done; done;;
  main_big) export SCN=scn_big LIVE_CAP=131072 RING_WORDS=8388608 GATE="--gate-serve 50000 --gate-train 60000"
    for g in adaptive fixed_prio global; do CFG[${g}_gated]="${CFG[$g]} $GATE"; done
    for rep in $(seq 1 "${REPS_BIG:-3}"); do for c in global global_gated static fixed_prio fixed_prio_gated v1_demand adaptive adaptive_gated; do one big_$c $rep ${CFG[$c]} $PHASES; done; done;;
  multi) export BIN=./adaptive_run_n8   # N Doom instances beside training + serve (steady load), paced at 35 Hz, then unpaced (max fps) to find the breaking point
    for rep in $(seq 1 "${REPS_MULTI:-3}"); do for n in 1 2 4 8; do for c in global fixed_prio adaptive; do
      one "multi_n${n}_$c" $rep ${CFG[$c]} --doom-instances $n --serve-rate 40 --doom-min-hi $((n + 4))
    done; done; done
    for rep in $(seq 1 "${REPS_MULTI:-3}"); do for n in 1 2 4 8; do
      one "unpaced_n${n}_adaptive" $rep ${CFG[adaptive]} --doom-instances $n --fps 1000 --duration-s 3 --serve-rate 40 --doom-min-hi $((n + 4))
    done; done;;
  bw)    # AdamW sweep (bandwidth-bound co-runner, class 3, weight 16 so it really takes the memory system) beside Doom + serve + train
    W4="--sched dynamic --weights 4,4,1,16 --urgent-us 28571,0,0,0 --serve-rate 40 --sweep-gb 1.6 --sweep-inflight 6"
    for rep in $(seq 1 "$REPS"); do
      one bw_none $rep --sched dynamic --weights 4,4,1,16 --urgent-us 28571,0,0,0 --serve-rate 40
      one bw_sweep $rep $W4
      one bw_sweep_arbiter $rep $W4 --arbiter
      one bw_sweep_adaptive $rep ${CFG[adaptive]} --serve-rate 40 --sweep-gb 1.6 --sweep-inflight 6 --arbiter-assist
      one bw_sweep_cap $rep $W4 --bw-cta 1000000,1000000,1000000,24
    done;;
  all_a) mkdir -p "$OUT"; "$0" suites; "$0" main; "$0" main_big;;
  tuned) for rep in $(seq 1 "$REPS"); do for c in alone adaptive_gated adaptive_tuned fixed_prio_gated; do one $c $rep ${CFG[$c]} $PHASES; done; done;;
  all_b) mkdir -p "$OUT"; "$0" tuned; DUR=16 "$0" multi; DUR=16 REPS=4 "$0" bw;;
  suites) mkdir -p "$OUT"
    ./test_adaptive 2>&1 | tee "$OUT/suites_functional.txt"; ./test_adaptive --expect-fail floor_neg 2>&1 | tee "$OUT/suites_negative.txt"
    ./test_adaptive granularity 2>&1 | tee "$OUT/suites_granularity.txt"; ./test_adaptive hol 2>&1 | tee "$OUT/suites_hol.txt";;
esac
echo "# batch $GROUP done $(date -u +%FT%TZ)"
