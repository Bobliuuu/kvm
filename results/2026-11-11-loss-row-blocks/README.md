# The MLP step's matcher reads a loss lowered in row blocks (Nov 11)

A tiling may lower the loss in row blocks (the frontend's `Tiling.ce_rows`; #39 sets 16 rows in the a100 preset). These files check that
the fusion pass now fuses such bundles to exactly what it makes from the one-task loss, and warns when an MLP run it was written for
stays unfused. Paths on the host are shown as `/root/reads/`.

- **`fuse_pass.log`** (CPU, `tests/static/test_fuse_pass.cpp`, g++ -O2): `gen_mlp.py` bundles (`--no-golden`, the default a100 tiling) of
  h72-d2, h48-d2 and h128-d4 at 3 steps and h72-d2 at 50, from `main`'s frontend and from `main` + #39's two frontend commits, and the
  panel's own 200-step h72 bundle, through `main`'s test and pass and this branch's. #39's bundles (159 tasks for h72 against 144) refuse on `main`'s pass ("expected the reduce after the fused softmax
  CE") and fuse on this one to the same digest as `main`'s bundles (h72 `f78062de68ead70f`, h48 `d4ec8817b17e59ef`); h128-d4 reaches the
  same shared-memory refusal from either frontend. The digest covers the pass's whole result: the rewritten tasks, edges and arguments,
  every fused run's descriptor and the interior buffers. For each bundle that fuses, the test also lowers its loss again in 16-row
  blocks in both forms (fused, 159 tasks for h72; per op, 177) and requires the same digest, and breaks the covering four ways (a gap, an
  overlap, a short cover, another vocabulary in one block): each stays unfused, with a warning.
  The long bundles also show the test's own fix: `main`'s test packed an op's argument types into 32 bits, so `matmul_tile`'s `flags`
  (argument 11 of 13) read as a buffer, and a flags word of 514 (F_DGELU | 2) "touched" buffer 514 wherever a bundle had one. It refused
  the panel's h72 at step 15 and the 50-step bundles at step 22. With 64 bits they fuse: the panel's h72 into 9,199 tasks replaced by
  4,801, the executor's own report on that bundle (`executor_report_panel_h72_main.log`, `main`'s binary), and the 50-step bundles from
  both frontends to one digest.
- **`fuse_pass_panel.log`** (CPU): the panel's own bundles from both frontends (its generator, the referee's bench): mlp72-r1, -r8 and the
  mixed-width row fuse on this pass from either frontend to one digest each, as `main`'s pass fuses them from its own; on `main`'s
  frontend, `main`'s test misses one of the eight runs of -r8 and of the mixed row (the 32-bit types). mlp256 and mlp1024 are shapes
  `main`'s template leaves out: unfused, without a warning.
- **`merge_with_39.log`**: `git merge-tree --write-tree` of this branch with #39, in both orders: exit 0, no conflicted file, one tree
  (`3868958a`). That tree's own `gen_mlp.py` makes the 159-task h72 bundle, and its own `test_fuse_pass` fuses it to `main`'s digest.
- **`python_tests.log`**: the three unittest suites on this branch (run at `71af38a`; the commit after it changes a C++ test only), on the
  A100 host's Python (numpy and torch present, no GPU visible to the tests): 157, 74 and 137 tests, all OK.
- **`ce-fix/`** (A100-SXM4-80GB, one hold): `main`'s binary (`4bbdaab`) against this branch's (key `e8b69fa3`, built from `24f3ccf`,
  binary md5 `249fcee7`; the commit after it changes only which unchanged runs log a warning).
  - `run_check.log`: `tests/static/run_check.sh` at 1, 4, 16, 107 and 108 workers on this branch's binary: PASS.
  - `dump_<bundle>_<binary>.log`, `dumps.txt`: the 3-step h72-d2 from both frontends, run once with `--dump-buffers` on both binaries.
    This branch fuses both (`main`'s 137 tasks and #39's 152 into the same 73), and both dump the same 73 buffer lines as `main`'s binary
    on `main`'s bundle. `main`'s binary leaves #39's bundle unfused (159 tasks per op). The dumps themselves stay on the host.
  - `ab_<row>.json`, `ab_<row>.log`: the panel's four one-block rows from #39's frontend (the panel's own generator), 5 paired rounds,
    warm: `main` per op against this branch fused, losses within 2e-3, x3.733-4.382. And mlp72-r1 from `main`'s frontend, every weight
    bitwise, x1.001.
  - `gpu.sh`, `binaries.md5`: what ran, and the two binaries.
