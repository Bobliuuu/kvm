# kvm.cu timeline from the VM's own per-instruction timestamps (A100)

Goal 4 of AGENTS.md. `kvm.cu` built with `-DKVM_TIMELINE` stores, per instruction, `%globaltimer` at claim, at dependencies satisfied
and at done, plus `%smid` and block index (32 bytes/instruction, written by thread 0 after the instruction's last signal). The default
build compiles all of it out. Renderer: `tools/kvm_timeline.py` (rows = workers grouped by SM; 7 colors = 5 opcodes + wait + idle).

Machine: NVIDIA A100-SXM4-40GB, 108 SMs, occupancy 2 blocks/SM = 216 workers, driver 580.105.08, CUDA 12.8, sm_80,
`nvcc -O3 -arch=sm_80 -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4 [-DKVM_TIMELINE]`, commit of `kvm.cu` = this PR's head.
Model: 2-layer MLP B=128 D=128 H=256 O=64, TF32, SGD. Opcodes: H1 fwd hidden, Y fwd out + MSE, DH1 bwd hidden, DW2 / DW1 bwd weight + SGD.

Commands: `./kvm_tl --runs R --steps S --timeline FILE`, then `python3 -I tools/kvm_timeline.py FILE out.png --panels op,step,run`.
Raw logs: `run.log` (selftest, 1 run, overhead), `run-multi.log` (16 and 48 runs). Data: `timeline-*.csv.gz`.

## Correctness (instrumented build)
Selftest ok. 1 run x 6 steps, 16 and 48 runs x 40 steps, and 1/8/32 runs x 300 steps (overhead loop): all end in PASS;
loss vs CPU fp32 within 3.6e-5; VM weights bitwise equal (max |dW| 0.00e+00) to the three baselines in every run.

## Overhead (VM total ms, 300 steps, 5 repeats, same box and process family; runs 1 / 8 / 32)
| runs | plain | -DKVM_TIMELINE | delta |
|---|---|---|---|
| 1 | 6.71-6.76 | 6.65-6.70 | noise (instrumented not slower) |
| 8 | 8.80-8.82 | 8.87-8.88 | +0.6% |
| 32 | 28.71-28.78 | 29.12-29.19 | +1.5% |

Cost = 2 globaltimer reads before the work, 1 after, one 32 B store per instruction (about 4 stores/us/SM at 32 runs).

## Figures
- `timeline-1run.png`: 1 run, 6 steps, 720 instructions in 128 us. Workers execute 9% of the time and wait on counters 77%: the
  step is a dependency chain (5 phases, one global-memory hop each), not compute. No step overlaps the next, correctly: step s+1's H1 reads
  the W1/W2 that step s's DW1/DW2 write, so one run cannot overlap its own steps.
- `timeline-16run.png` (+ `-zoom.png`, first 80 us): 16 runs (LR sweep) x 40 steps, 76,800 instructions in 2.02 ms. Execute 65%, wait 17%.
  The runs advance as a wave through H1 -> Y -> DH1 -> DW2 -> DW1 across all 216 workers; wait shrinks from 77% to 17% because other
  runs' instructions fill the dependency stalls. Step s+1 starts (first instructions of the next step) while the tail of step s of other
  runs is still executing: 39 of 39 step boundaries overlap, 4,729 instructions began before the previous step had fully finished.
- `timeline-48run.png`: 48 runs x 40 steps, 230,400 instructions in 6.18 ms; execute 65%, wait 14%.

Caveats: "overlap" is across runs (the previous step of a different run still draining); each single run keeps its true dependencies.
Per-step cost at 16 runs is 3.17 us/run-step and 3.23 at 48: the GPU is saturated, so more runs add no speed. At 48 runs the VM is
0.98x of the batched-graph baseline (batching alone gets most of the win, as in README). The timeline shows why: idle is gone, so the
remaining VM advantage (launch/tail removal) is small. The timeline does not look like the original post's picture (73 rows,
7 colors, 1.75 ms); we match its form, not its content.
