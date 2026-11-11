# The grid MLP step: one copy of the model, each weight matrix cut K x N over a gang

**Verdict.**
- **What it is:** an MLP training step for widths the split step cannot hold (#30's `fmlp_split` keeps every layer's whole input width on each
  member, so mlp1024, h1024-d2, needs 99,292 floats a member at best against a 38,400 budget). The grid step keeps one copy of the model: each
  weight matrix is cut into its own Ka x Nb grid of blocks over the gang's members, every member works on all 96 rows, and SGD updates each block
  in place. mlp1024 at 107-108 members needs 37,104 floats a member.
- **Correct on a GPU:** on an RTX 5070 Laptop (sm_120, 36 SMs) the step checks against float64 at four shapes, and at h96-d2 its errors equal the
  split step's on the same model and data.
- **In the executor:** `fmlp_grid` (op 194). The fusion pass takes the panel's mlp1024 in this form at 107-108 workers, and the plan's records replay
  through every gang's barrier.
- **Not yet measured:** mlp1024 itself (it needs 107-108 members: an A100) and every A100 time.

## The step

| phase | what crosses the barrier after it (through L2) |
|---|---|
| layer 0 (K = E, uncut): one column block a member | the member's columns of H |
| hidden layer l: the partial X[:, K_a] W[K_a, N_b] | Ka partials, then each member's piece of H (rows R_a, columns N_b) after its sum, bias and GELU |
| head: partial logits over the member's rows of the head | the partials, then each row's logit gradient after the cross-entropy |
| backward, head and hidden layers: the input-gradient partial dZ[:, N_b] W[K_a, N_b]^T, from the block before its update; dW = X^T dZ into the block | the partials, then each member's piece of dZ |
| layer 0: its partial of dX0 | the partials; the embedding rows' owners sum their tokens' rows |
| write-back | the step's end |

4 L + 2 barriers a step. Sums over many partials run as 8 chains added in a fixed tree, so a run is deterministic and 8 loads are in flight
instead of one (on the 5070, h512-d1 went from 91.7 to 43.7 us a step with that change and the same errors). The cross-entropy's and the embedding
owners' sums then spread over the whole block (35.8 us), and the narrow GEMMs (layer 0's three, the head's output gradient) run on warps over row
blocks instead of one warp a 16-column block (31.9 us). Both keep every sum and its order, so the check lines repeat digit for digit: h512-d1 runs
2.9x faster than the first version with the same numerics.

## Results: RTX 5070 Laptop, 36 members, 20 steps, the first 3 checked against float64

The 5070's times are sm_120 at a laptop's clocks: they show where the step's time goes, never an A100's verdict.

| shape | grids (layer 0, hidden, head) | loss, max relative | worst update, relative L2 | us a step |
|---|---|---|---|---|
| h96-d2 | 1 x 6, 6 x 1, 6 x 1 | 3.6e-7 | 1.44e-3 (the embedding) | 48.5 |
| h192-d2 | 1 x 12, 12 x 2, 12 x 1 | 4.0e-7 | 1.39e-3 (the embedding) | 49.0 |
| h256-d1 | 1 x 16, -, 16 x 1 | 7.7e-7 | 5.8e-4 (the embedding) | 31.7 |
| h512-d1 | 1 x 32, -, 32 x 1 | 8.7e-7 | 6.0e-4 (the embedding) | 31.9 |
| h256-d2 | refused: 101,312 bytes dynamic + 128 static > 101,376 a block | | | |

- **Against the split step on the same model and data** (`split_h96-d2_s6_5070.log`, h96-d2, 6 shares): the same update errors (W 5.49e-4,
  b 4.75e-4, Wh 4.37e-4, emb 1.43e-3) at 20.5 us a step, against the grid step's 48.5. At widths the split step holds, it is the faster form: it crosses 2 L + 1 barriers to the
  grid's 4 L + 2.
- **The check's bound** is the split bed's: the loss within 1e-3 and every update within 2e-2 relative L2 of float64.

## The executor op

- **The fusion pass** offers the grid form to a run that the one-block and split forms both refuse: one gang of every worker a step, when that member fits
  the budget. `fuse_pass.log` (CPU, gen_mlp bundles): the panel's mlp1024 (h1024-d2) becomes one gang of 108 members a step (layer 0 1 x 64, hidden
  13 x 8, head 64 x 1, 37,104 floats a member). Every other bundle keeps its form (split, one-block), and every gang's records replay through its
  barriers.
- **The static worker** prefetches a grid run's descriptor during the record's waits and runs `grid_run` with the run's error word.
  `make_fused_state` builds each run's descriptor and scratch.
- **The executor's PTX compile** with all of it (`ptx_check.log`): rc 0 at `2722a41` (sm_80, LEAN, nvcc 12.8), with the same warnings as the base,
  none of them in this change. ptxas has not run on the executor.
- **The grid bed's sm_80 build** (`bed_regs.log`, the binary the A100 read runs): 223 registers, no spill.

## Not yet done

- **The A100 read of mlp1024** (h1024-d2) at 108 members: `grid_bed --hidden 1024 --depth 2 --members 108`, beside today's 304.7 us a step for
  the row (block 1's final panel) and PyTorch's 104.6. Then the executor build, `KVM_CHECK_MLP_HIDDEN=1024 tests/static/run_check.sh 107 108`, and the row's
  paired read.
- **Depth 4:** every hidden block stays resident through the step, so h1024-d4 needs 60,528 floats a member; streaming the blocks per layer lifts it.
