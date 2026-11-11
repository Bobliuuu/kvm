# The gang stack's pass reads a loss lowered in row blocks (Nov 11)

The stack's one-block, split and grid steps all read the loss through the same matcher. A tiling that lowers the loss in row blocks (the
frontend's `Tiling.ce_rows`; #39 sets 16 rows in the a100 preset) left every one of them unfused. This branch carries the matcher change
of the `main` PR to the stack's pass.

- **`fuse_pass.log`** (CPU, this branch's `tests/static/test_fuse_pass.cpp` at `b540932`, g++ -O2; its first line names the build): MLP bundles from `main`'s frontend (`4bbdaab`) and
  from `main` + #39's two frontend commits, each with the form the pass must make at 108 workers.
  - `gen_mlp.py` bundles at 3 steps: h72-d2 and h48-d2 (one-block); h256-d4, h224-d2, h200-d1, h576-d1 and h128-d4 (split, 18 shares);
    h1024-d2 (grid).
  - The panel's own bundles (its generator, the referee's bench): mlp72-r1, -r8 and the mixed-width row (one-block), mlp256-r1 (split),
    mlp1024-r1 (grid).
  - Every bundle takes its expected form on its expected entry, from both frontends, and each pair fuses to one digest: the rewritten
    tasks, edges and arguments, every fused run's descriptor (shapes, buffers, slices, form and gang), the interior buffers and the entry.
    So the stack runs the same tasks on #39's bundles as on `main`'s.
  - For each bundle that fuses, the test lowers its loss again in 16-row blocks in both forms and requires the same digest, and breaks
    the covering four ways (a gap, an overlap, a short cover, another vocabulary in one block): each stays unfused, with a warning.
  - Every rewritten bundle's records replay through its gangs' barriers, as the test always checks.

The change is host code only (the fusion pass, and one log line in `kvm_exec.cu`): the device code is the stack's, so the stack's reads of
its fused steps hold for these bundles, whose rewritten form is byte for byte the one from `main`'s frontend.
