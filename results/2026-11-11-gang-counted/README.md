# The gang crossing past one flag line: a counted barrier (A100)

This replaces the rule of `results/2026-11-11-gang-cross/` (flag lines up to 64 members, the counter form past them). That README stays as
dated history.

`kvm_gang::cross` (include/kvm_gang.cuh) keeps the flag lines (`barrier_lines`) while a gang fits one line, `KVM_GANG_LINES_MAX` = 32
members, and takes a counted barrier (`barrier_counted`) past it, in place of the counter form.

- **The counted barrier** is one 64-bit count of arrivals a gang, zeroed once and never reset. A member adds itself with
  `red.release.gpu.add`, so its arrival costs no round trip, and its thread 0 polls the count until it reaches the crossing's target.
- **No reset between steps or runs.** A member takes its first target from the count it reads before its first arrival, rounded down to a
  multiple of the gang's size. Late starters therefore agree with early ones.
- **Who follows the rule:** the split and grid MLP steps cross through `cross`. The fused GPT step picks its form once, at its kernel's
  entry, by the same rule (`kvm_gang::counted_form`), so each of its crossing sites carries one form.

Both forms are exact barriers with the same ordering and abort contract. The choice changes cost only, never a result: every buffer below
is bitwise across the arms.

Every figure here was re-derived from these raw files by the team's verifier (Nov 11). In `raw/`, the host's working folders read
`/root/work/` and the GPU lock's own lines are removed; everything else is as the runs wrote it.

## The barrier alone

`bench/gang/barrier_bench.cu` on an A100-SXM4-80GB (108 SMs): ns a crossing, the median of 5 launches of 4,000 crossings, with data checked
between crossings. The counted form, the flag lines and the counter form, from `raw/counted-alone/barrier_bench.log`:

| members | 2 | 4 | 8 | 16 | 32 | 33 | 48 | 64 | 96 | 108 |
|---|---|---|---|---|---|---|---|---|---|---|
| counted | 948 | 1,069 | 1,206 | 1,290 | 1,346 | 1,365 | 1,402 | 1,406 | 1,498 | 1,551 |
| flag lines | 994 | 1,055 | 1,161 | 1,226 | 1,312 | 1,578 | 1,882 | 2,159 | 2,968 | 4,425 |
| counter | 1,676 | 1,982 | 2,107 | 2,177 | 2,195 | 2,212 | 2,220 | 2,215 | 2,299 | 2,355 |

- **Up to one line (4-32 members)** the flag lines are faster than the counted barrier, by 14-64 ns a crossing. At 2 members the counted
  barrier is 46 ns faster.
- **From the second line (33 members) up** they are slower, by 213 ns at 33 and 2,874 ns at 108: every member's warps poll every line.
- **The counted barrier** stays within 0.95-1.55 µs at every size, under the counter form everywhere, which pays an atomic's round trip and
  then waits on the last arrival's publish.

The same bench on this branch, with the shipped crossing (`cross`) beside the forms it chooses between and the counter form it replaced
(`raw/bitwise-device-bench/b.log`; the same GPU and launch shape, a later run):

| members | 2 | 8 | 16 | 32 | 33 | 48 | 64 | 108 |
|---|---|---|---|---|---|---|---|---|
| `cross` (this branch) | 1,039 | 1,218 | 1,287 | 1,358 | 1,391 | 1,427 | 1,431 | 1,580 |
| counted | 948 | 1,206 | 1,290 | 1,344 | 1,366 | 1,401 | 1,405 | 1,551 |
| flag lines | 1,020 | 1,203 | 1,273 | 1,350 | 1,613 | 1,895 | 2,174 | 3,467 |
| counter | 1,677 | 2,107 | 2,177 | 2,195 | 2,213 | 2,221 | 2,216 | 2,354 |

- **Up to one line** the two forms are within 72 ns of each other in this run. From 33 members the flag lines need a second line and fall
  behind by 247 ns, and by 1,916 ns at 108.
- **`cross` costs 8-29 ns a crossing more than the form it takes, called directly:** a run-time branch on the gang's size, with both forms'
  code at the call. That cost is the subject of "The 16- and 32-member GPT rows" below.

## Inside the steps

### The MLP steps' beds: the counted barrier against the earlier crossing

`bench/fused_mlp/split_bed.cu` and `grid_bed.cu` on the A100: each bed runs the op's own step on one gang, checks 3 steps against
float64 and times 50. The arms are the counted barrier at every size (candidate) and the earlier crossing: flag lines up to 64 members,
the counter form past them (control). Each row was read in 5 rounds, the arms' order alternating; no run was slow (above 1.10x its own
arm's fastest), and each row's check lines are identical across the arms (`raw/counted-beds/read.log`).

| bed | members | the control's form | control, µs a step | counted, µs a step | paired |
|---|---|---|---|---|---|
| split h256-d4, 8 shares | 48 | flag lines (two lines) | 114.11 | 113.85 | x1.002 |
| split h256-d4, 10 shares | 60 | flag lines (two lines) | 120.88 | 116.52 | x1.037 |
| split h256-d4, 12 shares | 72 | counter | 123.74 | 117.25 | x1.054 |
| split h256-d4, 16 shares | 96 | counter | 102.87 | 97.27 | x1.060 |
| split h256-d4, 18 shares | 108 | counter | 109.41 | 102.98 | x1.063 |
| split h576-d1, 18 shares | 108 | counter | 40.22 | 38.41 | x1.047 |
| grid h1024-d2 | 96 | counter | 139.19 | 127.44 | x1.092 |
| grid h1024-d2 | 108 | counter | 136.86 | 125.26 | x1.093 |
| grid h496-d3 | 108 | counter | 161.07 | 145.14 | x1.110 |
| grid h512-d4 | 108 | counter | 207.70 | 188.99 | x1.100 |
| grid h600-d3 (padded to 608) | 108 | counter | 168.56 | 152.81 | x1.102 |

On two lines the counted barrier ties the flag lines at 48 members and beats them by 3.7% at 60. Past 64 it beats the counter form by
4.7-11.0%.

### The counted barrier at every size, in the executor: an earlier read

The executor with the counted barrier at every size (candidate) against the same executor on the earlier crossing (control), both on one
A100, 5 interleaved rounds a row with no slow run, every row's weights bitwise (`raw/counted-everywhere-executor/`: `ab_*.json`, the gang
sizes from `report_*_cand.log`). That code predates the fused GPT step's width rule, so gpt-tiny ran a 32-member gang (it runs 16 now):

| row | members a gang | paired |
|---|---|---|
| ho-mlp-h128-d4-r1 | 108 | x1.0710 |
| mlp1024-r1 (grid) | 108 | x1.0553 |
| mlp256-r1 | 108 | x1.0308 |
| ho-mlp-h576-d1-r1 | 108 | x1.0191 |
| ho-mlp-h224-d2-r8 | 8 gangs of 24 | x0.9997 |
| ho-mlp-h200-d1-r16 | 16 gangs of 12 | x0.9950 |
| gpt-small | 32 | x0.9950 |
| gpt-tiny | 32 | x0.9937 |
| gpt-162m, mlp72-r1, mlp72-r8, mlp72-r32, mlp-hetero-r8 | no gang | x0.9996-1.0005 |

Every gain sits on a 108-member gang, and every loss on a gang of 12-32: the counted barrier at every size gave up up to 0.6% where one
flag line serves. Hence the bound at one line.

### The fused GPT step's bed, by bound

`bench/fused_gpt/fgpt_bed.cu` on the A100, built from this branch three times: `KVM_GANG_LINES_MAX` 0 (the counted barrier at every size),
32 (this branch) and 64 (the flag lines up to two lines, as the fused GPT step crossed before). All three builds cross through `cross`, so
they differ in the form a gang takes, not in code. Each shape ran at its gang width, 20 steps with the phase clock, in 5 rounds with the
arms' order rotating. Every run's outputs hash equal across the three builds, and the 0 and 64 builds pass the workload's float64 checks
first (`raw/gpt-bed-bounds/`).

| shape | members | counted (0), µs a step | this branch (32) | flag lines (64) | counted against this branch | flag lines against this branch |
|---|---|---|---|---|---|---|
| gpt-tiny | 16 | 138.0 | 135.4 | 135.5 | x0.981 | x0.999 |
| gpt-small | 32 | 300.6 | 293.0 | 293.3 | x0.975 | x0.999 |
| d128 | 64 | 850.5 | 850.7 | 858.1 | x1.000 | x0.992 |
| d128 hold-out draw | 64 | 1,062.0 | 1,061.6 | 1,062.5 | x1.000 | x1.000 |

Inside the step, the flag lines beat the counted barrier by 1.9-2.5% on one-line gangs, and two flag lines lose to it by up to 0.8% at 64
members. The arms that choose the same form read x0.999-1.000.

### The executor: this branch against its base

This branch against its base (`2a8692b`, the head of #48), each built with the panel's flags, both on one A100-SXM4-80GB: the panel's
9 rows and four hold-out MLP draws, 5 interleaved rounds a row, `--warm-ms 500`, no slow run in either arm (`raw/executor/`: `ab_*.json`,
the gang sizes from `report_*_cand.log`). Every row's weights are bitwise equal (gate `w.` at tolerance 0).

Two builds of the branch were read. Its first commit (`5b48276`) was read on every row. Its head, with the fused GPT step's crossing form
chosen at its entry (below), was read on the GPT rows, 9 rounds each (`raw/entry/`). Every kernel except the fused GPT one has the same
SASS in the two builds, all 18 of them (`raw/entry/sass-identity-against-the-first-build.txt`), so the first commit's figures stand for
the head on every other row.

| row | members a gang | base, ms | this branch, ms | paired | rounds |
|---|---|---|---|---|---|
| ho-mlp-h128-d4-r1 (200 steps) | 108 | 15.491 | 14.484 | **x1.070** | x1.069-1.071 |
| ho-mlp-h224-d2-r8 | 8 gangs of 108 | 21.059 | 20.261 | **x1.039** | x1.035-1.042 |
| mlp1024-r1 (grid) | 108 | 6.784 | 6.542 | **x1.037** | x1.034-1.041 |
| mlp256-r1 (read again, 9 rounds) | 108 | 4.822 | 4.679 | **x1.030** | x1.025-1.040 |
| ho-mlp-h200-d1-r16 | 16 gangs of 108 | 26.569 | 26.077 | **x1.019** | x1.017-1.021 |
| ho-mlp-h576-d1-r1 (read again, 9 rounds) | 108 | 1.882 | 1.827 | **x1.030** | x1.014-1.043 |
| gpt-162m, mlp72-r1, mlp72-r8, mlp72-r32, mlp-hetero-r8 | no gang | | | x1.000 | x0.993-1.004 |
| gpt-small (the head, 9 rounds) | 32 | 14.324 | 14.327 | x1.000 | x0.998-1.003 |
| gpt-tiny (the head, 9 rounds) | 16 | 6.623 | 6.621 | x1.001 | x0.998-1.001 |

mlp256-r1 and h576-d1 first read x1.026 and x1.014 (`raw/executor/`), but their rounds spread x1.024-1.048 and x0.999-1.031 while the
host's CPU quota throttled the read. They were read again alone, 9 rounds each, with no period throttled (`raw/reread/`); the table
quotes that read. Geometric means of the paired medians, with the head's GPT rows: the panel's 9 rows **x1.0075**, the four hold-out MLP
draws **x1.0394**.

### The fused GPT step

The head against the base on gpt-tiny, gpt-small and a d128 hold-out draw (d 128, 4 layers, T 96, B 2, 30 steps), 9 rounds a row, no slow
run, every weight bitwise (`raw/entry/`). Their per-step state is bitwise equal too: 110, 158 and 234 buffers (202,768, 1,398,160 and
10,306,800 bytes) on the fused path, compared byte for byte (`raw/entry/run.log`).

| row | members | base, µs a step | the head | paired | rounds |
|---|---|---|---|---|---|
| d128 hold-out draw | 64 | 1,048.5 | 1,020.6 | **x1.027** | x1.026-1.029 |
| gpt-small | 32 | 286.5 | 286.5 | x1.000 | x0.998-1.003 |
| gpt-tiny | 16 | 132.5 | 132.4 | x1.001 | x0.998-1.001 |

### The fused GPT step's crossing form, chosen at its entry

In the branch's first build the fused GPT step crossed through `cross`, which inlines both forms at each crossing site behind a run-time
branch on the gang's size. Its 16- and 32-member gangs kept the flag lines, yet lost 0.6-0.9% with every round below 1: gpt-tiny x0.994
and gpt-small x0.992, while the 64-member d128 draw gained x1.006 (9 rounds, `raw/gpt-executor/`). The cause was code, not the crossing's
form. The SASS census of `kvm_fgpt_kernel` (`raw/census/`):

| the fused GPT program, `kvm_fgpt::run` | base | first build |
|---|---|---|
| instructions | 40,499 | **41,519 (+1,020)** |
| basic blocks (the whole kernel) | 2,985 | **3,154 (+169)** |
| registers touched | 171 | 170 |
| local loads / stores | 10 / 5 | 10 / 5 |
| HMMA | 178 | 178 |

Registers, spills and tensor-core work are unchanged. The bench above prices the branch at 8-29 ns a crossing.

The head chooses the form once. The program and its kernel take it as a template parameter, and the launch picks the instance by the
gang's size (`kvm_gang::counted_form`), so each crossing site carries one form. The flag-line instance is the base's kernel byte for byte:
41,120 instructions with the same SASS (`raw/entry/sass-fgpt.txt`). The counted instance is 40,688 instructions. The 16- and 32-member
rows read x1.000-1.001, and the 64-member draw x1.027 (above).

## Checks

- **Device test** (`tests/static/test_gang_device.cu`), every form (counted, flag lines, `cross` at 8 and at 32) in two layouts, 2,000
  rounds of two crossings: every member reads every slot of its round, the control words end as clean runs leave them, a second launch from
  the first's words with staggered starts passes the same checks, and an aborted gang ends. PASS on the A100
  (`raw/bitwise-device-bench/b.log`) and on an RTX 5070 Laptop GPU (`raw/laptop-5070/`).
- **`run_check.sh`** on this branch: the default rows, h128 d4 and h1024 d2 pass (`raw/executor/rc_*.log`). h256 d4 at 107 workers misses
  its float64 bound (5.12e-4 against 2.10e-4). Every read of the PRs below this one today shows the same 5.12e-4, and the crossing changes
  no arithmetic.

## Raw

- `counted-alone/`: the barrier bench and the device test of the counted barrier as first written, on the A100.
- `counted-beds/`: the split and grid beds, the counted barrier against the earlier crossing.
- `counted-everywhere-executor/`: the executor with the counted barrier at every size, against the earlier crossing.
- `executor/`, `reread/`: this branch against its base, the panel and the hold-out MLP draws.
- `entry/`: the head's GPT rows against the base, with the SASS checks between the builds and the scripts that made them.
- `gpt-executor/`: the first build's fused GPT rows and the d128 hold-out draw.
- `bitwise-device-bench/`: the GPT rows' state compared byte for byte, the device test and the bench, on this branch.
- `census/`: the SASS census of the fused GPT kernel, base and first build.
- `laptop-5070/`: the device test and the bench on an RTX 5070 Laptop GPU.
