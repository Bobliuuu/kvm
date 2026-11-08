#!/usr/bin/env bash
# Doom + serve demo (large preset, gpt_162m bf16s) on this tree.
set -u
export KVM_ARCH=sm_80 NVCC_ARCH=sm_80
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"precision":"bf16s"}'
mkdir -p /logs/$TAG; touch ~/.gpu.lock
DEMO_TAG=new MODEL="$M" TILING=a100 STEPS=6 WINDOW=1 SERVE_EVERY=2 FRAMES=700 PRECISE=0 LR=3e-4 WARMUP=2 MAXB=8 SANITIZE=0 GOLDEN=0 RUN_ARGS="" LOCK_WAIT=60 bash demo/remote_demo.sh > /logs/$TAG/new.log 2>&1
echo "demo exit $?"; mkdir -p /logs/$TAG/new; cp runs/new/check.log runs/new/report.json runs/new/kvm_run.log /logs/$TAG/new/ 2>/dev/null; tail -20 /logs/$TAG/new.log
