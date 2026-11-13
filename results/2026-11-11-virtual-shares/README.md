# Virtual shares for the split MLP step (Nov 11)

A split run's arithmetic runs over its shape's own count of virtual shares, Sv, and a gang may carry them on fewer members, S <= Sv
(`include/kvm_fused_split.h` has the design). These files check it.

- **`fuse-pass/`** (CPU, `tests/static/test_fuse_pass` on gen_mlp bundles at 1, 4, 8, 48, 107 and 108 workers): every expected form and
  entry holds, and every gang's records replay through its barriers. At 108 workers a run alone carries its 18 virtual shares on 18
  members; 16 runs of h200-d1 carry them on 2 (12-member gangs), and 8 runs of h224-d2 on 4 (24-member gangs). The test checks each
  split member's gang against its own run's S, up to the entry's Sv.
- **`bed/`** (A100-SXM4-80GB, `bench/fused_mlp/split_bed.cu`):
  - `memcheck.log`, `racecheck.log`: h96-d2 with 6 virtual shares on 2 members, 0 errors and 0 hazards.
  - `eq-*.log`, `equality.txt`: the check's 3 steps at Sv = 18 dumped on S = 18, 9, 6, 4, 3 and 2, compared byte for byte with S = 18.
    h200-d1 at every S, h224-d2 at 9, 6 and 4, and h256-d4 at 9 are identical. The other counts do not fit a member ("does not fit").
    Every run passes its float64 check.
  - `t-new-*.log` against `t-old-*.log`, `summary.txt`: the step at Sv = 18 on S members against the step whose arithmetic is S shares,
    one gang, warmed, at about 1,405 MHz. S = Sv leaves a single run as it was (h200 28.92 against 28.82 µs, h256 87.89 against 88.11,
    h224 47.29 against 45.99, one run each). Carrying 18 virtual shares on fewer members costs h200 on 2 members 40.33 against 34.22 µs
    (+18%), and h224 on 4 members 66.63 against 59.79 (+11%): every member still writes and every sum still reads 18 partials.
- **`read/`** (A100-SXM4-80GB): this head's executor against `main`'s (`4bbdaab`) on the panel's 9 rows and the hold-out's four MLP
  draws, 5 interleaved rounds a row (`ab_<row>.json`: the paired medians are computed from unrounded times, the stored arrays are
  rounded to 0.01 ms). `report.py` ran each row once on both binaries first and set its gate (`gates.txt`): bitwise where both fusion
  reports agree, the panel's 2e-3 on the losses where they differ. `summary.txt` is the reader's table. Two of its FAIL lines are not
  this change's:
  - mlp72-r32's bitwise gate: both binaries run its one-block form (3 slices of 32 rows), so `report.py` set the bitwise gate, but the
    granule fix this branch is stacked on changes that form's sums by design. Read again under the loss gate: `ab_mlp72-r32.json`.
  - `run_check` h256-d4 at 107 workers (`rc_h256d4.log`): 5.12e-4 against a float64 bound of 2.10e-4. That run's arithmetic is 16
    shares, unchanged by this branch, and the stack's reads before it print the same line; `main`'s own per-op path reads 2.98e-4 against
    the same bound (`results/2026-11-10-split-mlp-step/float64-distance/`). The 108-worker run (18 shares) passes at 1.59e-4.
- **`base/`** (A100-SXM4-80GB, one hold): this head's executor against the head it is stacked on (the one-block granule fix, key
  `5d5938db`), 5 interleaved rounds a row, every row's weights bitwise to that head's (`ab_<row>.json`). Four runs alone (the S = Sv path):
  mlp256-r1, mlp1024-r1, h576-d1 and h128-d4; and the two multi-run draws, 16 runs of h200-d1 and 8 of h224-d2. `gpu.sh` is the whole
  hold, whose first two parts belong to the granule fix's read and to `read/ab_mlp72-r32.json`.
- **`copies/`** (A100-SXM4-80GB): on this head's executor, `run_mlp.py`'s gates 1 to 3 on h256-d4, h224-d2, h200-d1, h576-d1 and h128-d4
  (gate 2: 4 replicated copies bitwise to the run alone): all pass. `rep_check.py` (`results/2026-11-10-split-mlp-step/replicated-copies/`)
  at 16 copies of h200-d1, 4 and 8 of h224-d2, 32 of h256-d4 and 32 of h72-d2: every buffer of every copy bitwise to the run alone.
  `single_runs.txt`: one run each of ho576, ho128 and mlp256 dumped on this head and on the stack without virtual shares: every buffer
  line identical (only the run's own timing line differs).
- **`bed_regs.log`**: the bed's 11 entries (one per virtual share count) at 229-233 registers, no spill.
- **`ptx_check.log`**: `kvm_exec.cu` compiles to PTX (sm_80, LEAN) with no new warning.
- **`python_tests.log`**: the three unittest suites on this branch's code (run at `c646862`, before its rebase onto the granule fix's
  results commit, which adds files under `results/` only), on the A100 host's Python (numpy and torch present, no GPU visible to the
  tests): 157, 74 and 137 tests, all OK.

Paths on the host are shown as `/root/reads/`.
