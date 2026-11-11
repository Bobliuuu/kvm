# The grid step at widths that are odd multiples of 8

The grid step cut every weight in granules of 16 columns, so a hidden width such as 600 or 488 was refused, and the split form could not hold
those shapes either. Such a width H is now cut as Hp = H + 8:
- the padding's weights and biases load as zeros and are never written back;
- the step's activations and partials hold Hp columns, so the padded units carry exact zeros through every sum (gelu(0) = 0, and the next layer's padded rows are zero).

Widths that are multiples of 16 run unchanged. Every figure here is (unverified) until the team's verifier names it.

## Reach over the hold-out's MLP draws

Every width-depth cell the hold-out draws (widths 48-640 in steps of 8 but the panel's 72 and 256, depth 1-4), one generated bundle each,
through the fusion pass at 108 workers and an A100's budget (`raw/reach.sh`, `raw/table.py`; the W = 108 lines of each pass log are
`raw/fuse_pass_108_*.log`):

| | one-block | split | grid | unfused |
|---|---|---|---|---|
| before (`raw/forms_before_pad.txt`) | 24 | 178 | 45 | **45 (15%)** |
| after (`raw/forms_after_pad.txt`) | 24 | 178 | 90 | **0** |

The 45 unfused cells were all odd multiples of 8 at depth 2-4: no share count fit the split form, and the grid form refused the width.

## RTX 5070 Laptop (sm_120, 36 members, 20 steps, the first 3 against float64)

- **Multiples of 16, against the previous bed** (`raw/same_*.log`): check lines identical, digit for digit, at h96-d2, h256-d1 and h512-d1.
- **Odd multiples of 8, against float64** (`raw/new_*.log`): all pass.

  | shape | loss max rel | worst update, rel L2 |
  |---|---|---|
  | h88-d2 | 4.83e-7 | 1.41e-3 |
  | h104-d2 | 2.77e-7 | 1.45e-3 |
  | h200-d1 | 8.05e-7 | 5.92e-4 |
  | h488-d1 | 1.11e-6 | 5.91e-4 |
  | h504-d1 | 7.16e-7 | 5.91e-4 |
  | h152-d3 | 3.60e-7 | 9.31e-4 |

  h120-d2 does not fit the 5070's shared memory at 36 members (`raw/new_h120-d2.log`); an A100's 108 members take it.
- **Builds** (`raw/build.log`, `raw/grid_bed_pad*.build.log`): the executor compiles to PTX (the panel's flags, LEAN) with only the base's 4 warnings. The grid bed holds 228 registers on sm_80 with no spill, and 255 on sm_120.

Depth 3 and 4 at the hold-out's widths fit only an A100's 108 members. The next window's bed read runs h496-d3 and h512-d4 there.
