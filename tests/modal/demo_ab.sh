#!/usr/bin/env bash
# demo large (gpt_162m bf16s, 6 steps, serve every 2, 700 Doom frames) on this tree and on the base executor (origin/perf/combined's
# kvm_exec.cu + include/, kept in tests/modal/base), back to back in one container. For tests/modal/run_suite.py --suite custom.
set -u
export KVM_ARCH=sm_80 NVCC_ARCH=sm_80
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"precision":"bf16s"}'
run() {  # tag
  DEMO_TAG=$1 MODEL="$M" TILING=a100 STEPS=6 WINDOW=1 SERVE_EVERY=2 FRAMES=700 PRECISE=0 LR=3e-4 WARMUP=2 MAXB=8 SANITIZE=0 GOLDEN=0 RUN_ARGS="" LOCK_WAIT=60 \
    bash demo/remote_demo.sh > /logs/$TAG/$1.log 2>&1
  echo "demo $1 exit $?"
  mkdir -p /logs/$TAG/$1; cp runs/$1/check.log runs/$1/report.json runs/$1/kvm_run.log /logs/$TAG/$1/ 2>/dev/null
}
mkdir -p /logs/$TAG; touch ~/.gpu.lock
run new
cp kvm_exec.cu /tmp/new_kvm_exec.cu; mkdir -p /tmp/new_inc; cp include/kvm_exec.cuh include/kvm_sched.cuh /tmp/new_inc/
cp tests/modal/base/kvm_exec.cu kvm_exec.cu; cp tests/modal/base/include/kvm_exec.cuh tests/modal/base/include/kvm_sched.cuh include/
run base
cp /tmp/new_kvm_exec.cu kvm_exec.cu; cp /tmp/new_inc/* include/
