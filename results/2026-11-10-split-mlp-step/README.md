# The split MLP step: a fused training step for MLPs one block cannot hold

**Verdict.**
- **What it is:** an MLP training step too wide for one block now runs as one gang of Q x S blocks (`fmlp_split`), each block holding 1/S of
  every layer.
- **Speed:** at depth 1-4 its bed checks against float64 at every shape that fits, and on the A100 the h576-d1 step takes 52.1 us at S = 5
  (light-load clock 1,093 MHz; 40.4 us at 1,410) and 34.2 us at S = 12.
- **Reach:** the fusion pass takes the panel's mlp256 (h256-d4) and the hold-out's four MLP draws (h576-d1, h200-d1 x 16 runs, h128-d4,
  h224-d2 x 8 runs) in this form, and leaves the one-block runs as they were. Every rewritten bundle's records replay through its gangs'
  barriers without deadlock.
- **Not yet measured:** the executor compiles with all of it to PTX, but its GPU checks against the unfused path and its timing wait for the
  next build. At an A100's default 108 workers the split shapes take 18 shares (108 members), a gang no GPU has run yet: the A100 bed ran 5 to
  12 shares.

## The step

Member q * S + s takes row slice q (16 rows) and share s, the hidden units [h_s, h_s + w_s) of every layer: those columns of each layer's weights
and bias, and those rows of the head. The members meet at the gang barrier (`include/kvm_gang.cuh`) 2 x depth + 1 times a step:

| crossing | what crosses it (through L2) |
|---|---|
| all-gather, after each hidden layer but the top | each member's columns of the layer; the slice's whole layer read back as the next input |
| logits | each member's partial logits (16 x V); the slice's S partials summed in share order, then the cross-entropy |
| reduce-scatter, below each layer but the first | each member's full-width partial of the activation gradient; each sums its own columns over the slice's shares |
| partials | the partial gradients, for the owners |
| step end | the owners' updates: the Q members of share s split its parameters and sum the Q slices' partials in slice order; every member owns some embedding rows |

Every sum has a fixed order, so a run is deterministic. GEMMs are the fused MLP step's (m16n8k8 TF32), in chunks of 128 past one GEMM's width.

## What changed

* `include/kvm_fused_split.h` holds the descriptor and the shared-memory layout. `include/kvm_fused_split.cuh` holds `split_run`, one member's
  share of one step.
* `include/kvm_ops_fmlp.def`: `fmlp_split` is op 192, in the fused-step ops' range 176..199.
* `include/kvm_static.cuh`: the static worker prefetches the run's split descriptor during the record's waits and dispatches on its S and Q.
  `split_run` passes the run's error word to every barrier, so a failure ends the gang.
* `include/kvm_fuse.h` decides from the DAG alone:
  - **The split form:** a run the one-block form cannot take (a width past 128, or no slice inside the budget) takes the split form when it has
    one hidden width, at most 4 layers, 96 rows a step, and a share count whose member layout fits the budget.
  - **Share count:** the pass picks the largest share count every split run takes whose gangs fit the workers at once, and otherwise each run
    its fewest.
  - **Edges:**
    - a step's in-edges reach every member, and its out-edges leave from member 0 (a gang finishes together);
    - consecutive steps join member to member.
  - **Parser:** it now reads what wide layers lower to: GEMMs on 128x128 and 256x128 tiles, a bias gradient in column ranges, an update in
    element ranges, and ordering noops between a GEMM's tiles or between updates.
* `kvm_exec.cu`: `make_fused_state` builds each split run's descriptor with its partials, exchanges and barrier words. The planner prices the op at
  40 us.
* `bench/fused_mlp/split_bed.cu`: the bed runs `split_run` itself on one gang, step after step, against a float64 reference.

## Results

**A100-SXM4-40GB, h576-d1** (`split1_a100/`; the GPU at its light-load clock, 1,093 MHz at 33-64 W, so cycles are the stable measure):

| S (blocks) | us a step | cycles a step | us at 1,410 MHz |
|---|---|---|---|
| 5 (30) | 52.06 | 56.9k | 40.4 |
| 6 (36) | 44.33 | 48.5k | 34.4 |
| 8 (48) | 38.12 | 41.6k | 29.5 |
| 12 (72) | 34.18 | 37.3k | 26.4 |

**RTX 3080 Ti, depth 1-4, S = 8-10:**

| shape | float64 check: loss / worst update, relative | op body, cycles a step | standalone kernel, cycles a step |
|---|---|---|---|
| h576-d1 | 1.3e-6 / 5.8e-4 | 49.3k | 46.9k |
| h256-d2 | 2.7e-7 / 1.5e-3 | 73.6k | 67.6k |
| h224-d2 | 1.3e-7 / 1.5e-3 | 70.5k | 64.2k |
| h128-d3 | 2.7e-7 / 4.9e-3 | 69.8k | (not run) |
| h128-d4 | 2.0e-7 / 1.7e-2 (the embedding) | 86.2k | 77.1k |

- **The two kernel forms** (`split2_*.log`): the op body reads its descriptor from shared memory each step, while the standalone kernel held its
  shape in kernel parameters, and that costs the body 5-12%.
- **The check's bound** is on the update (after minus before), 2e-2. The embedding's error grows with depth (four TF32 GEMMs chain into it); the
  panel's gate judges the weights themselves.
- **`split0_3080ti.log`:** the first depth-1 bed. Its per-phase split predates the per-warp phase clock, but its step time stands. Its three GPU-lock lines show the queue as `split-bed`.

**Fusion pass** (`fuse_pass.log`, CPU, gen_mlp bundles, the pass run afresh at 1, 4, 8, 48, 107 and 108 workers, and each bundle's expected
form checked at 108, an A100's default):

| bundle | at 108 workers |
|---|---|
| h256-d4 (the panel's mlp256), h576-d1, h128-d4, h256-d2, h224-d2, h200-d3 | split form: one gang a step of 6 x 18 = 108 members (12.9k-23.5k floats a member); at 107 workers, 6 x 16 = 96 |
| h200-d1 x 16 runs | split form: 48 gangs of 6 x 2 = 12, in waves over the workers |
| h224-d2 x 8 runs | split form: 24 gangs of 6 x 4 = 24 |
| h48-d2, h72-d2, mlp72 x 8 runs | the one-block form, as before |
| h1024-d2 | unchanged: no share count fits (at best 99,292 floats a member, at 16 shares, against 38,400) |

- **Every rewritten bundle** is acyclic, keeps its other tasks as they were, and plans with every gang's members on distinct workers.
- **Its emitted records replay as the device runs them** (`tests/static/replay.h`): every gang's barrier is crossed, nothing deadlocks, and
  every edge of the DAG (successors and versions) is honoured. A mutant planner that puts a gang's members on one worker deadlocks the replay of
  every split bundle.
- **16 shares** (`fix(fused)`): at 8 to 15 shares h256-d4's widest share is 24 units, padded to 32 columns, so a member needs 39,900 floats
  against 38,400. Without 16, the panel's mlp256 split only at 108 workers.

**Executor** (`ptx_check.log`): `kvm_exec.cu` with every change above, LEAN, compiles to PTX (`-gencode arch=compute_80,code=compute_80`) with
rc 0. That checks every changed line of C++. ptxas has not run on the executor.

**The op body's registers** (`bed_regs.log`, the bed's full sm_80 build, 22 s): all 11 `split_run` instantiations (2 to 18 shares) use 227-234
registers, with no spill.

## Not yet done

- **The executor build** and `tests/static/run_check.sh` on the split form: `KVM_CHECK_MLP_HIDDEN=256 KVM_CHECK_MLP_DEPTH=4
  tests/static/run_check.sh 16 48 107 108` (the panel's mlp256), then width 128 at depth 4. That compares split runs with the CPU golden and with
  float64 and eager PyTorch, and checks that they repeat bitwise.
- **The A100 read** of mlp256-r1 and the hold-out's MLP draws against today's executor.
