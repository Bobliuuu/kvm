# Runs launched on the A100 box (nohup + flock, output on the box)

| UTC start | What | Tree | Box dir | Status |
|---|---|---|---|---|
| 2026-11-06 06:35 | test_sched functional (cost share reserve edf inherit rand) + bench (calib dispatch arbiter rebalance baselines overload affinity mix) | merge perf/arbiter, 62322bf | kvm-integ-arb/out.txt | queued on GPU lock |
| 2026-11-06 06:36 | kvm.cu selftest/smoke/r8/watchdog + test_exec | merge feat/queue, 1fee371 | kvm-integ-q/out.txt | queued on GPU lock |
| 2026-11-06 07:05 | Modal A100-40GB private: build test_sched/test_exec/kvm; sched functional+bench+mix, exec suite, kvm selftest/r8/watchdog | tree 3 commits after 1fee371 (arbiter+queue merges + op-coverage check) | Modal app ap-kbL6YtszgPPDy7UFSQQegp, volume kvm-validate, `modal run --detach tests/gpu/modal_validate.py` | running |
