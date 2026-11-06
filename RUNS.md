# Runs launched on the A100 box (nohup + flock, output on the box)

| UTC start | What | Tree | Box dir | Status |
|---|---|---|---|---|
| 2026-11-06 06:35 | test_sched functional (cost share reserve edf inherit rand) + bench (calib dispatch arbiter rebalance baselines overload affinity mix) | merge perf/arbiter, 62322bf | kvm-integ-arb/out.txt | queued on GPU lock |
| 2026-11-06 06:36 | kvm.cu selftest/smoke/r8/watchdog + test_exec | merge feat/queue, 1fee371 | kvm-integ-q/out.txt | queued on GPU lock |
