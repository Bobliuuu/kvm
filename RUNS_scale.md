# Runs for feat/scale-evidence (no cloud spend; the Modal launch was denied by the permission classifier)
| when (UTC) | what | where | cost |
|---|---|---|---|
| 2026-11-06 | `scale_gpu.py golden` (8 stacks, bitwise vs CPU golden) | A100 box, lock-wrapped, scratch `kvm-scratch-scale` | $0 (own box) |
| 2026-11-06 | `scale_gpu.py pair` d256..d4096, ranks 4/2 (6 results) | same | $0 |
| 2026-11-06 | Modal a100x2/a100x4/h100x8 (`dist/modal_scale.py`) | NOT RUN: `modal run --detach` was refused by the permission classifier | $0 |
