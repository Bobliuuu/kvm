#!/bin/bash
# The fused forms' reach over the hold-out's MLP space (holdout draw: hidden 8 x 6..80 = 48..640 but the panel's widths, depth 1..4), at an
# A100's budget and 108 workers: each shape's bundle (gen_mlp, 1 step) through the pass of #31's head, and the form it takes.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); T=${T:?a checkout of the tree}; PY=${PY:?a python with numpy}
cd "$T" && g++ -std=c++17 -O2 -Iinclude -Itests/static tests/static/test_fuse_pass.cpp -o "$D/test_fuse_pass" || exit 1
for k in $(seq 6 80); do h=$((8 * k)); [ $h = 72 ] || [ $h = 256 ] && continue
  for d in 1 2 3 4; do o=$D/b/h$h-d$d; [ -f $o/bundle/metadata.json ] || "$PY" -I "$T/bench/exec_vs_torch/gen_mlp.py" $o --hidden $h --depth $d --steps 1 --no-golden > /dev/null 2>&1 || echo "gen h$h-d$d FAILED"; done
done
cd "$D/b" && "$D/test_fuse_pass" $(ls -d h*-d* | sort -V | sed 's|$|/bundle|') > "$D/reach.log" 2>&1; echo "test rc=$?"
