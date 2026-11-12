# The GPT step's weight gradients in attention's waits (A100)

The fused GPT step's attention backward runs B x H x QB items on a gang of G members, and where the items do not fill the gang the other
members wait. This branch gives them the step's weight gradients to run in those waits: Zero Bubble's split of the backward into B (input
gradients) and W (weight gradients), inside one gang.
- **The MLP's doers:** members without an item run dW2, db1 and dW1 for up to 4 fc1 owners each.
- **The attention's doers:** when 2 W <= G and QB > 1, member W + w runs item w's dWo, dWq and dbq.
- **Its own kernel instance:** ZBW's code is compiled into an instance of the step kernel that the host picks only for a run whose plan
  engages it.

Every element sums as it did, so the step is bitwise the base's.

Every figure here was re-derived from these raw files by the team's verifier (Nov 12). In `raw/`, the host's working folders read
`/root/work/` and the GPU lock's own lines are removed. Each job's log is renamed for its folder (`panel.log`, `bed.log`, `idle.log`), and its end
marker likewise. Everything else is as the runs wrote it.

Arms in the bed logs:
- **base:** the PR's base, #54 + #56;
- **lever2:** ZBW's two commits, with its code in every run;
- **zfix:** this branch, ZBW's code in its own instance;
- **attnonly:** lever2 with the MLP's doers switched off (`KVM_FGPT_ZBW_OWNERS=0`);
- **control2:** lever2 with the attention doer's dWo at the other TF32 precision;
- **tail** and **shift:** the idle-code read's arms (below).

## The panel's 9 rows against the base

`raw/panel/` holds this branch against its base, each built with the panel's flags, on one A100-SXM4-40GB: 3 paired rounds a row, every weight
buffer gated bitwise (`--gate-tol 0`).

| row | base, ms | this branch, ms | paired |
|---|---|---|---|
| gpt-tiny | 6.509 | 6.201 | x1.0497 |
| gpt-small | 13.786 | 13.733 | x1.0040 |
| gpt-162m | 353.220 | 352.673 | x1.0020 |
| mlp72-r1 | 5.583 | 5.582 | x0.9993 |
| mlp72-r8 | 5.515 | 5.509 | x1.0016 |
| mlp72-r32 | 8.262 | 8.236 | x1.0036 |
| mlp-hetero-r8 | 6.288 | 6.272 | x1.0021 |
| mlp256-r1 | 4.570 | 4.554 | x1.0099 |
| mlp1024-r1 | 6.498 | 6.480 | x1.0028 |

All 9 are bitwise, and the geometric mean is x1.0082.
- **Only gpt-tiny's step engages ZBW,** with 12 doers at its gang of 16.
- **gpt-small's gang of 32 is filled by its 32 items,** so it runs the instance without ZBW's code. The other rows run no fused GPT step.
- **mlp256-r1's rounds straddle 1.0** (x0.9998-1.0346).
- **Kernels** (`raw/panel/key-res-usage.txt`, `cuobjdump -res-usage`): the base's 2 step instances and this branch's 4 all use 168 registers, a
  56-byte stack and no local memory. The other 18 kernels are unchanged.

## The step's bed

`raw/bed/`: `bench/fused_gpt/fgpt_bed.cu`, arms base, lever2 and zfix. The bed picks zfix's instance by the executor's rule, and its logs say
which. It ran bitwise first (20 steps; every arm's five outputs hashed), then 5 rotating rounds of 20 steps on the phase clock.
The bed is built with plain `nvcc -O3 -std=c++17 -arch=sm_80`, not with the executor's flags (`--fmad=false -Xcompiler -ffp-contract=off
-DKVM_LEAN`), so it runs other code generation of the same step: it shows the mechanism, and the panel's 9 rows above, on the executor's
own builds, carry the gain.
Two builds also place their code differently in GPU memory, which alone has moved a panel row by up to 5.5% (same device code, another load order), so each row's gain here carries that uncertainty.

| shape | ZBW | lever2 | zfix (this branch) |
|---|---|---|---|
| gpt-small at 32 | off | x0.9801 | x1.0015 |
| d128 at 32 | off | x0.9862 | x1.0036 |
| gpt-tiny at 16 | on, both halves | x1.0446 | x1.0462 |
| a d128 at 32 (k4) | on, 8 doers | x1.0484 | x1.0478 |
| the d128 hold-out draw at 64 | on, 16 doers | x1.0244 | x1.0246 |

(base / arm, medians.) `raw/bed/bed-res-usage.txt`: the base's 4 instances use 160 / 159 / 157 / 158 registers, lever2's 168, and zfix's
instances without ZBW's code 160 / 159 / 157 / 158.

## Why the instance: ZBW's code, idle

`raw/idle-code/` holds arms base and lever2 as above, plus:
- **tail:** lever2 with its three scratch buffers placed after every other buffer;
- **shift:** the base's code with lever2's scratch layout.

All four are bitwise on every shape.

| shape | lever2 | tail | shift |
|---|---|---|---|
| gpt-small at 32 (ZBW off) | x0.9801 | x0.9753 | x1.0007 |
| d128 at 32 (ZBW off) | x0.9861 | x0.9822 | x1.0038 |
| gpt-tiny at 16 (ZBW on) | x1.0462 | x1.0496 | x0.9955 |

The layout costs nothing (shift reads like the base), and the code costs 1.4-2.0% where ZBW stays off (tail reads like lever2). The phase clock
puts that cost in phases ZBW never runs (the attention backward's item phases, LayerNorm's backward), so it comes through code generation.
Hence ZBW's own instance.

## Bitwise

`raw/bitwise/`: 30 steps on 8 shapes, every arm's g, losses, m, p and v hashed (`bitwise.verdict`).
- **lever2 and attnonly equal the base everywhere.** The shapes are gpt-tiny at 8, 16 and 32; a second tiny draw at 16; a d128 at 32 (k4);
  gpt-small at 32; a d128 at 32; and the hold-out's d128 at 64.
- **control2 differs on the four shapes where the attention's doers run, and equals the base on the four where they do not.** So the
  comparison sees the doers' work.
