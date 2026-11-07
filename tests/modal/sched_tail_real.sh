#!/usr/bin/env bash
# Real Doom + GPT training + GPT serving on one persistent kernel, express admission / strict tier / controller on and off, for
# tests/modal/run_suite.py --suite custom --cmd 'bash tests/modal/sched_tail_real.sh'. Builds the Doom engine objects first (clang NVPTX).
# Results (summary.json per run, oracle comparison) go to /logs/$TAG/runs-*.
set -u
export KVM_ARCH=sm_80 PATH=$PATH:/usr/local/cuda/bin DEBIAN_FRONTEND=noninteractive
echo "== deps"; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq clang llvm curl gcc g++ git >/dev/null 2>&1; clang --version | head -1
echo "== doom"; bash doom/setup.sh && bash doom/build.sh && bash doom/cpu_build.sh || { echo "DOOM BUILD FAILED"; exit 1; }
echo "== scenarios"
python3 adaptive/make_scenario.py --out scn --train-runs 3 --serve-pool 48 --train-total 600 2>&1 | tail -1
python3 adaptive/make_scenario.py --out scn_big --model '{"name":"big","vocab":4096,"d_model":512,"layers":6,"heads":8,"seq":256,"batch":8}' --train-runs 2 --serve-pool 32 --train-total 300 2>&1 | tail -1
echo "== build adaptive_run"; time bash adaptive/build.sh 2>&1 | grep -E "error|fatal" -A3; ls -la adaptive_run || exit 1
mkdir -p /logs/$TAG
export OUT=/logs/$TAG/runs-big REPS=${REPS:-3} DUR=24 WAD=doom/third_party/doom1.wad
bash adaptive/batch.sh tail_big 2>&1 | grep -E "^===|doom:|exit|FAIL|ERROR|error"
export OUT=/logs/$TAG/runs-small
bash adaptive/batch.sh tail 2>&1 | grep -E "^===|doom:|exit|FAIL|ERROR|error"
echo "== done"
