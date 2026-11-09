#!/usr/bin/env bash
# ON THE GPU BOX, repository root. A/B of the persistent kernel's resident blocks per SM on the MLP bundles box_job.sh made:
# ./kvm_exec (98304 B dynamic smem -> 1 block/SM, 107 workers) vs ./kvm_exec_smem72 (built with -DKVM_GPT_MMA_SMEM=0 -> 73728 B -> 2 blocks/SM).
# The -D only shrinks the smem reserved for the bf16 matmul op, which these f32 bundles never run; outputs are compared bitwise first.
#   flock -w 28800 ~/.gpu.lock bash bench/exec_vs_torch/ab_occupancy.sh WORK_DIR(out/final2/work_h72) [REPS]
set -uo pipefail
W="${1:?work dir}"; REPS="${2:-5}"
F="--max-tasks 262144 --live-cap 131072 --ring-words 8388608"
for b in kvm_exec kvm_exec_smem72; do ./$b $F "$W/time_n50_r1" 2>&1 | grep -E "device" | head -1; done
for b in kvm_exec kvm_exec_smem72; do ./$b --quiet --dump-buffers "$W/gate/bundle" | grep '^buffer' | md5sum | cut -c1-12 | sed "s/^/dump md5 $b /"; done
for r in 1 8 32; do for k in $(seq 1 "$REPS"); do
  for b in kvm_exec kvm_exec_smem72; do echo "R=$r $b $(./$b --quiet $F "$W/time_n50_r$r" | grep -E '^OK')"; done
done; done
