# Fused steps on kernel entries of their own

Every fused MLP form now runs on a static-kernel entry of its own, and the generic `kvm_static_kernel` holds no fused function:

| entry | runs |
|---|---|
| `kvm_static_gang_split<S>` | the split step, one entry per share count |
| `kvm_static_gang_grid` | the grid step |
| `kvm_static_block` | the one-block form (`fmlp_slice`, `fmlp_sgd`) |

The gangs also cross their barriers on flag lines (`kvm_gang::barrier_lines`).

## Why: a function's registers depend on its kernel's other functions

ptxas allocates an out-of-line function's registers knowing what every other function of the same kernel holds across its calls. The fused
MLP functions hold many values across their division and reciprocal helpers. So each of them costs the others registers, and the split
function, with its eleven share counts inlined into one body, spills even alone.

The probe (`raw/probe/regprobe.cu`) is the static kernel's fused part without the executor: the wrappers exactly as `kvm_static.cuh`
had them, out of line, called from a worker loop that holds its state across the calls. Each variant was compiled for sm_80 and read with
the team's SASS census (`sass_check.sh`, then `raw/probe/loops_by_function.py` to attribute each MMA loop to its function).

| the kernel holds | function | registers | LDL / STL (static) | its MMA loops' reloads a pass, worst |
|---|---|---|---|---|
| grid alone | `kvm_s_fmlp_grid` | 219 | 1 / 0 | 0 |
| one-block form (slice, sgd), split, grid | `kvm_s_fmlp_grid` | 184 | 85 / 35 | 35 |
| split alone, its 11 share counts in one function | `kvm_s_fmlp_split` | 252 | 2,202 / 1,266 | 33 (loops of up to 64 HMMA) |
| **entry: grid alone** (the panel's flags) | `kvm_s_fmlp_grid` | 219 | 1 / 0 | **0** |
| **entry: split at 18 shares** (the panel's flags) | `kvm_s_fmlp_split<18>` | 229 | 4 / 30 | **3** (0 in loops of up to 64 HMMA) |

The first three rows use nvcc's default flags. The entries use the panel's: `--fmad=false -Xcompiler -ffp-contract=off -DKVM_LEAN`. All figures (unverified) until the team's verifier names them.
The split beds, with one share count a kernel, held 227-234 registers and no spill.

## What changed

- **`include/kvm_static.cuh`:**
  - The worker loop is a template on its entry.
  - `generic` runs every op but the fused ones. Its dispatch refuses them (`KVM_ERR_BAD_OP`).
  - `block`, `split` (one share count) and `grid` each run their fused functions and, through `kvm_dispatch_fused`, the steps' data windows (`select_window`) and ordering noops. Anything else fails the run.
- **`include/kvm_fuse.h`:**
  - The pass records the entry (`kvm_fuse_entry`). A fused bundle holds one form (the one-block form, or one gang form at one share count) and, besides its fused tasks, only windows and noops.
  - **One form a bundle:** runs that took different forms move to the widest form among them when each fits it, and split runs share one share count. A bundle that still holds fused runs beside other work comes back unfused, with the reason in `KvmFuseResult::dropped`.
- **`kvm_exec.cu`:**
  - `run_static` launches the chosen entry and logs it (`static kernel: entry split, 18 shares, ...`); a bundle the pass left unfused is logged as a warning, with its reason.
  - It refuses three launches: a fused task bound for the generic entry; an entry holding more static shared memory than the generic one, whose share the fused budget subtracted; and a launch whose resident blocks fall short of the plan's workers, since a gang member that never starts would hang its barrier.
- **`include/kvm_gang.cuh`** (#29): both barriers return through the block's OR (`__syncthreads_or`), so no entry holds static shared memory for one.
- **Names:** no fused entry's name contains `kvm_static_kernel`, which every seat's census reads the generic kernel by.
- **Side effect, and a prediction:** the generic entry, which the GPT rows' static runs use, compiles with no function that holds values across helper calls. The next build's census should show its dispatch and tiles keeping the registers the dynamic kernel's keep (production's static kernel lost 49-64 there).

## Checks

An independent review of the change found no wrong-entry, fallback or barrier fault. It found five defects, all fixed here:
- a mixed-form bundle ran fully unfused;
- the shared-memory check passed only through rounding;
- no residency check;
- an uncapped grid gang;
- a test check that compared a value with itself.


- **`raw/fuse_pass.log`** (CPU, `tests/static/test_fuse_pass`): the panel's and the hold-out's MLP bundles through the pass at 1, 4, 8, 48, 107 and 108 workers. At 108, each bundle's expected form and entry hold:

  | bundles | entry |
  |---|---|
  | h1024-d2 | grid |
  | h256-d4, h128-d4, h576-d1, h256-d2 | split, 18 shares |
  | h200-d1 x 16 runs | split, 2 shares |
  | h224-d2 x 8 runs | split, 4 shares |
  | h72-d2, h48-d2 | one-block |

  No bundle is `mixed`, and every gang's records replay through its barriers. Two more cases:
  - **The panel's mixed-width row** (8 widths, h48-h104 d2, merged): one-block at an A100's budget, and split at one share count (3) at a 24,032-float budget (`raw/fuse_pass_small.log`).
  - **An MLP run merged with a GPT run:** nothing fuses, and the reason is named.

  Each split member's gang is Q x the entry's share count.
- **`raw/static_gang.log`:** the planner's gang tests pass unchanged.
- **On the A100:** the build's own `sass_check` reads the entries by name before any timing, in the next window's read of the split and grid steps.

## The grid step on flag lines: an A/B on an RTX 5070 Laptop

The grid bed on the counter barrier (the code of `5c56c90`, "old") against `kvm_gang::barrier_lines` returning through the block's OR
("new", this branch):
- 36 members, the panel's numerics flags, sm_120.
- 20 steps, the first 3 against float64.
- Two rounds, the arms in alternating order (`raw/flag-lines/`).

Every figure in this section is (unverified) until the team's verifier names it.

| shape | check lines, old against new | µs a step, old (rounds 1, 2) | new (rounds 1, 2) |
|---|---|---|---|
| h96-d2 | 5 of 5 identical | 48.46, 48.29 | 45.83, 45.79 |
| h256-d1 | 5 of 5 identical | 31.37, 30.32 | 28.48, 28.42 |
| h512-d1 | 5 of 5 identical | 32.25, 32.18 | 30.16, 30.21 |

The flag lines are faster in all 6 pairs.

| grid bed kernel | old (counter) | new (flag lines) |
|---|---|---|
| sm_120 (5070) | 255 registers, spills 212 B | 255 registers, spills 312 B stored / 452 B loaded |
| sm_80 (A100) | 223 registers, no spill | 223 registers, no spill |

**The entries with flag lines** (the probe rebuilt on this branch, sm_80, the panel's flags; `raw/probe/panel-flags-*-flag-lines/`):

| entry | registers | worst MMA-loop reloads a pass |
|---|---|---|
| grid | 228 | 0 |
| split<18> | 225 | 3 (0 in loops of up to 64 HMMA) |

Within 9 registers of the counter barrier's 219 and 229.

**The executor's PTX compile** (`raw/ptx_check.log`: `kvm_exec.cu`, the panel's flags, `-DKVM_LEAN`, code=compute_80): rc 0, with
only the 4 warnings the base already has (`kvm_ops_gpt.cuh`'s `#pragma unroll`).
