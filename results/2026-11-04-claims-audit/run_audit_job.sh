#!/usr/bin/env bash
# Detached audit job: rsync the tree to a PRIVATE scratch dir, then run every group on the GPU box under ONE hold of the GPU lock.
#   KVM_HOST=user@host [PYBIN=python-with-torch-on-the-box] results/2026-11-04-claims-audit/run_audit_job.sh WORKDIR PREBUILT_MODELS_DIR NEWMODEL_BUNDLES_DIR
# Prints the remote log dir; fetch with: scp -r $KVM_HOST:~/kvm-audit-x-all/audit_logs <dest>   (JOB_DONE marks the end of audit_logs/job.log)
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST}"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PRE="$2"; NM="$3"; PYBIN="${PYBIN:-python3}"
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30"; R='~/kvm-audit-x-all'
rsync -az -e "$SSH" --delete --exclude .git --exclude logs --exclude results --exclude docs "$ROOT/" "$HOST:$R/"
$SSH "$HOST" "mkdir -p $R/prebuilt $R/nm $R/audit_logs"; rsync -az -e "$SSH" "$PRE/" "$HOST:$R/prebuilt/"; rsync -az -e "$SSH" "$NM/" "$HOST:$R/nm/"
rsync -az -e "$SSH" "$ROOT/results/2026-11-04-claims-audit/newmodel/" "$HOST:$R/tb/"
cat > "$1/job.sh" <<JOB
cd $R; export PATH=\$PATH:/usr/local/cuda/bin; L=audit_logs; export L KVM_ARCH=sm_80 KVM_NO_LOCK=1
git_c=\$(cat COMMIT 2>/dev/null); echo "audit job start \$(date -u +%FT%TZ)" > \$L/job.log
bash tests/gpu/build.sh > \$L/build_exec.txt 2>&1; bash tests/gpu/build_sched.sh > \$L/build_sched.txt 2>&1
NODOOM=1 bash tests/gpu/build_adaptive.sh > \$L/build_adaptive.txt 2>&1
nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 kvm_exec.cu -o kvm_exec > \$L/build_kvm_exec.txt 2>&1
( cd tests/gpu_ops && NV='nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I.' && \$NV test_ops.cu -o test_ops && \$NV bench_ops.cu -o bench_ops && \$NV -DBENCH_MINB=2 bench_ops.cu -o bench_minb2 ) > \$L/build_ops.txt 2>&1
flock -w 28800 ~/.gpu.lock bash -c '
 echo LOCKED \$(date -u +%FT%TZ) >> \$L/job.log; nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader > \$L/gpu.txt
 echo "[A] test_exec functional"; timeout 1800 ./test_exec > \$L/exec_functional.txt 2>&1; echo exit \$? >> \$L/exec_functional.txt
 echo "[B] test_sched functional + negative"; timeout 900 ./test_sched --quiet cost share reserve edf inherit rand > \$L/sched_functional.txt 2>&1; echo exit \$? >> \$L/sched_functional.txt
 for n in starve_neg inherit_neg; do timeout 300 ./test_sched --quiet --expect-fail \$n >> \$L/sched_negative.txt 2>&1; echo exit \$? >> \$L/sched_negative.txt; done
 echo "[C] litmus + gpt"; bash tests/litmus/run_litmus.sh > \$L/litmus.txt 2>&1; bash tests/gpu/run_gpt.sh > \$L/gpt_modes.txt 2>&1
 echo "[D] test_adaptive"; timeout 900 ./test_adaptive --quiet telemetry loop floor admit_gran > \$L/adaptive.txt 2>&1; echo exit \$? >> \$L/adaptive.txt
 timeout 300 ./test_adaptive --quiet --expect-fail floor_neg > \$L/adaptive_neg.txt 2>&1; echo exit \$? >> \$L/adaptive_neg.txt
 echo "[E] new models, correctness dumps"; for c in xfmr_odd mlp_ce resmlp_rms bad_dims; do timeout 600 ./kvm_exec --blocks 4 --quiet --dump-buffers nm/\$c/bundle > nm/\$c.dump 2> nm/\$c.err; echo \$c exit \$? >> \$L/newmodel_run.txt; done
 echo "[E2] new models timing"; for r in 1 2 3 4 5; do echo "mlp_ce 200 steps rep \$r"; ./kvm_exec nm/t_mlp/bundle 2>&1 | grep "kernel done"; done > \$L/time_kvm.txt
 for r in 1 2 3 4 5; do echo "xfmr_odd 50 steps rep \$r"; ./kvm_exec nm/t_xfmr/bundle 2>&1 | grep "kernel done"; done >> \$L/time_kvm.txt
 $PYBIN tb/torch_bench.py mlp_ce 200 5 > \$L/time_torch.txt 2>&1; $PYBIN tb/torch_bench.py xfmr_odd 50 5 >> \$L/time_torch.txt 2>&1; $PYBIN tb/libref.py >> \$L/time_torch.txt 2>&1
 echo "[F] model zoo subset"; bash examples/models/run_box.sh prebuilt \$L/models --cases prebuilt/cases_sub.txt > \$L/models_run.txt 2>&1
 echo "[G] ops perf"; ( cd tests/gpu_ops; ./test_ops > ../../\$L/ops_test.txt 2>&1; echo exit \$? >> ../../\$L/ops_test.txt; BENCH_REPS=5 BENCH_ONLY=mm ./bench_ops > ../../\$L/ops_bench_default.txt 2>&1; BENCH_REPS=5 BENCH_ONLY=mm ./bench_minb2 > ../../\$L/ops_bench_minb2.txt 2>&1; BENCH_REPS=3 BENCH_ONLY=mem ./bench_ops > ../../\$L/ops_bench_mem.txt 2>&1 )
 echo UNLOCKED \$(date -u +%FT%TZ) >> \$L/job.log'
echo JOB_DONE >> \$L/job.log
JOB
scp -q -o BatchMode=yes "$1/job.sh" "$HOST:$R/job.sh"
$SSH "$HOST" "cd $R && nohup bash job.sh > job.out 2>&1 < /dev/null &"
echo "detached; poll: ssh \$KVM_HOST 'tail -3 $R/audit_logs/job.log'"
