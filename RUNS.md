# Run log (perf/game-parallel)

Shared Lambda A100-SXM4-40GB (host not recorded here), every GPU run inside `flock ~/.gpu.lock`, scratch `~/kvm-scratch-game`, jobs launched detached (`setsid nohup`).
Modal (own A100) was authorised but the launch was denied by the permission system in this session, so nothing ran there and no Modal app exists.

| when (UTC, 2026-11-06) | job | spec | where the numbers are |
|---|---|---|---|
| 06:5x | classic phase profile | `bin/doomprof` (`DOOM_PROF=1` build), 700 frames | results/2026-11-06-a100-game-parallel/classic-phase-profile.jsonl |
| 08:0x | stage 1 verify | `harness.py verify` (draw tiles only) | .../stage1-draw-tiles/ |
| 09:0x | stage 2a verify | planes drawn by one thread each | .../stage2a-planes-one-thread-each/ |
| 09:1x | stage 2 verify | planes compute spans, tiles draw pixels | .../stage2-planes-spans-tiles/ |
| 09:3x | speed | `harness.py speed` (3 reps, 700 frames) | .../speed/ |
| 09:2x-10:0x | sanitize; study x3 + heavy; instances 1-16; adaptive_run gamepar x3; 64-instance attempt (faulted) | `harness.py sanitize/study/instances`, `adaptive/batch.sh gamepar` | .../sanitize, .../study, .../instances, .../adaptive-gamepar |

Command pattern: `flock -w 14400 ~/.gpu.lock timeout <s> python3 doom/modal/harness.py <job> out/<job> <ninst>` from `~/kvm-scratch-game`
(`harness.py` is the in-container runner of `doom/modal/app.py`; it runs unchanged on any box with the built engine objects).
