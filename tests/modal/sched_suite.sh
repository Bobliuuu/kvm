#!/usr/bin/env bash
# Scheduler + adaptive suites and the bench suite, for tests/modal/run_suite.py --suite custom (mirrors run_sched_tests.sh).
set -u
export KVM_ARCH=sm_80
bash tests/gpu/build_sched.sh 2>&1 | grep -E "error" -A3; timeout 3000 ./test_sched; echo "sched exit: $?"
bash tests/gpu/build.sh 2>&1 | grep -E "error" -A3; timeout 1200 ./test_exec bench; echo "bench exit: $?"
