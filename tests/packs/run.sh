#!/usr/bin/env bash
# Pluggable-pack flow end to end on a GPU box (run from the repo root there; the CPU half is tests/test_packs.py):
# build kvm_exec with packs/example_saxpy through tools/kvm_build.sh, compile the workload with the SAME packs on the python side
# (golden from packs/example_saxpy/ref.py), run the executor with --golden. Usage: tests/packs/run.sh OUTDIR
set -euo pipefail
OUT="${1:?outdir}"; mkdir -p "$OUT"
eval "$(tools/kvm_build.sh --print-env --pack packs/example_saxpy)"
python3 runtime/compile_workload.py --emit-golden tests/packs/workload.json "$OUT/bundle"
tools/kvm_build.sh --pack packs/example_saxpy -o "$OUT/kvm_exec_packs" kvm_exec.cu
echo BUILD OK
${KVM_RUN_PREFIX:-} "$OUT/kvm_exec_packs" --blocks 4 --golden "$OUT/bundle"
