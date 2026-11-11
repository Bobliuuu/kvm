# The one-block step's gradients by 16-row granules, on the gang stack (Nov 11)

The read of this PR's head against the gang stack it is stacked on (#33), on one A100-SXM4-80GB, and `main`'s own checks on the same host.
Paths on the host are shown as `/root/reads/`.

- **`main-gates/`** (`main`'s own binary, `4bbdaab`): `run_mlp.py`'s gates 1 to 3 on h72-d2 and h48-d2 (`h72-d2.log`, `h48-d2.log`; gate 2
  is 4 replicated copies, every buffer bitwise to the run alone): both pass. `rep_check.py` at 8 and 32 copies of each (`rep-*.log`):
  8 copies pass; at 32, every copy differs from the run alone, in 6 of h72's 73 buffers and 8 of h48's (the embedding, weights and biases).
  `report_<row>_<binary>.log`: the fusion report of `main`, the stack (#33's code) and this head on mlp72-r1, -r8, -r32 and the mixed-width
  row: all three plan r1, r8 and the mixed row as 6 slices of 16 rows, and r32 as 3 slices of 32. `fuse_pass_gate.log`: `main`'s own
  fusion-pass test on the gate bundles above (CPU): the run alone fuses into the one-block form, and its 4 copies into 4 such runs, for
  h72 and h48, so gate 2 was read on the one-block form.
- **`sasshash_cmp.txt`** (CPU, before any timing): each kernel entry's SASS hashed with addresses stripped, this head's binary against the
  stack's. Only `kvm_static_block`, the one-block form's own entry, differs; the generic kernel and every split and grid entry are identical.
- **`ab_bitwise.json`, `ab_bitwise.log`**: mlp72-r1, mlp72-r8 and the mixed-width row, 5 paired rounds, every weight bitwise to the stack's
  (they take 16-row slices, whose arithmetic the granules keep).
- **`ab_r32.json`, `ab_r32.log`**: mlp72-r32, its losses within the panel's 2e-3 (its 32-row slices now sum 16-row granules).
- **`report_r32_cand.log`, `report_r32_ctrl.log`**: both binaries' fusion reports on mlp72-r32: 3 slices of 32 rows each.
- **`rep-*.log`**: 8 and 32 replicated copies of h72, and 4, 8 and 32 of h48, against the run alone, every buffer bitwise
  (`rep_check.py`, in `results/2026-11-10-split-mlp-step/replicated-copies/`).
- **`run_check.log`**: `tests/static/run_check.sh` at 1, 4, 16, 107 and 108 workers on this head: PASS.
- **`fuse-pass/`** (CPU, `tests/static/test_fuse_pass` on gen_mlp bundles at 1, 4, 8, 48, 107 and 108 workers): every expected form and entry
  holds, 0 failures; the one-block bundles plan as before (h72: 6 slices of 16 rows; 8 runs of h72: 6 of 16 each).
- **`python_tests.log`**: the three unittest suites on this head, on the A100 host's Python (numpy and torch present, no GPU visible to
  the tests): 157, 74 and 137 tests, all OK.
- **`gpu.sh`, `binaries.md5`**: what ran, and the two binaries.
