# The grid MLP step: one copy of the model, each weight matrix cut K x N over a gang

**Verdict.**
- **What it is:** an MLP training step for widths the split step cannot hold (#30's `fmlp_split` keeps every layer's whole input width on each
  member, so mlp1024, h1024-d2, needs 99,292 floats a member at best against a 38,400 budget). The grid step keeps one copy of the model: each
  weight matrix is cut into its own Ka x Nb grid of blocks over the gang's members, every member works on all 96 rows, and SGD updates each block
  in place. mlp1024 at 107-108 members needs 37,104 floats a member.
- **Correct on a GPU:** on an RTX 5070 Laptop (sm_120, 36 SMs) the step checks against float64 at four shapes, and at h96-d2 its errors equal the
  split step's on the same model and data.
- **Not yet measured:** mlp1024 itself (it needs 107-108 members: an A100) and every A100 time. The executor op is not wired yet.

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
instead of one (on the 5070, h512-d1 went from 91.7 to 43.7 us a step with that change and the same errors).

## Results: RTX 5070 Laptop, 36 members, 20 steps, the first 3 checked against float64

The 5070's times are sm_120 at a laptop's clocks: they show where the step's time goes, never an A100's verdict.

| shape | grids (layer 0, hidden, head) | loss, max relative | worst update, relative L2 | us a step |
|---|---|---|---|---|
| h96-d2 | 1 x 6, 6 x 1, 6 x 1 | 3.6e-7 | 1.44e-3 (the embedding) | 56.7 |
| h192-d2 | 1 x 12, 12 x 2, 12 x 1 | 4.0e-7 | 1.39e-3 (the embedding) | 59.4 |
| h256-d1 | 1 x 16, -, 16 x 1 | 7.7e-7 | 5.8e-4 (the embedding) | 39.2 |
| h512-d1 | 1 x 32, -, 32 x 1 | 8.7e-7 | 6.0e-4 (the embedding) | 42.2 |
| h256-d2 | refused: 101,312 bytes dynamic + 128 static > 101,376 a block | | | |

- **Against the split step on the same model and data** (`split_h96-d2_s6_5070.log`, h96-d2, 6 shares): the same update errors (W 5.49e-4,
  b 4.75e-4, Wh 4.37e-4, emb 1.43e-3) at 20.5 us a step. At widths the split step holds, it is the faster form: it crosses 2 L + 1 barriers to the
  grid's 4 L + 2.
- **The check's bound** is the split bed's: the loss within 1e-3 and every update within 2e-2 relative L2 of float64.

## Not yet done

- **The A100 read of mlp1024** (h1024-d2) at 108 members: `grid_bed --hidden 1024 --depth 2 --members 108`, beside today's 304.7 us a step for
  the row (block 1's final panel) and PyTorch's 104.6.
- **The executor op** (`fmlp_grid`), the fusion pass's grid form for runs the split form refuses, and its descriptors.
- **Depth 4:** every hidden block stays resident through the step, so h1024-d4 needs 60,528 floats a member; streaming the blocks per layer lifts it.
