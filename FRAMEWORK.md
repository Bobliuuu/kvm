# KVM framework: host compiler, ABI v2, weight versioning

Workload JSON -> `runtime/compile_workload.py` -> bundle -> GPU executor (and the CPU reference `runtime/workload_vm.py`).
The device never parses JSON; it reads typed 32-bit words.

```
workload.json --compile--> tasks.bin  successors.bin  args.bin  metadata.json --> executor
      |                                                                              |
      +--------------- workload_vm.py (same tasks, edges, gates) --> golden.json <--diff
```

## Single sources of truth (parsed, never copied)

| What | File | Parsed by |
|---|---|---|
| Descriptor layout (`KvmTaskDesc`, 64 B), `KVM_ABI_VERSION`, `KVM_NO_VAR` | `include/kvm_task_abi.h` | `runtime/abi.py` |
| Operators: name, id, `arg:type` list (`buf`/`buf?`/`i32`/`f32`) | `include/kvm_ops.def`, then `kvm_ops_gpt.def`, `kvm_ops_extra.def` (ids 96-122, generic model-zoo ops, `matmul_mma_bf16` = 122), `kvm_ops_game.def`, `kvm_ops_comm.def` when present (executor order) | `runtime/abi.py` |
| What an op does + buffer extents / dtypes it needs | `runtime/ref_ops.py` (core), `ref_gpt.py` (GPT) | compiler (bounds), CPU VM |

Ids are sparse and unique across registry files. `buf?` marks an optional buffer argument: the workload passes JSON `null`
and the word is `-1` (`abi.ABSENT_BUF`); a required `buf` given `null` is a compile error, so `-1` can only reach an
argument the registry declared optional.

Add an op: append an `X(...)` line to a registry (never renumber), add its CUDA body, add its entry to `ref_ops.OPS`
(a test fails until you do). An op with no CPU model goes in `OPAQUE_OPS` (below).

## Workload schema

```json
{ "variables": ["weight"],
  "buffers":   [{"name": "w", "rows": 1, "cols": 1, "init": {"kind": "fill", "value": 0.5}}],
  "graphs":    [{"name": "train", "tasks": [
      {"id": "upd", "op": "sgd", "args": {"w": "w", "g": "grad", "lr": 0.0625, "n": 1},
       "after": ["bwd"], "priority": 0, "deadline_us": -1, "publishes": "weight",
       "reads": [{"var": "weight", "policy": "exact", "version": 1}]}]}] }
```

Buffer args are buffer names or `null` (absent optional); they resolve to `KvmBuf` table indices = declaration order.
`i32`/`f32` are JSON numbers. A buffer has `dtype` `f32` (default, `KvmBuf.dtype` 0) or `i32` (1, index and counter
buffers) or `bf16` (2, raw 16-bit storage, 2 bytes per element; init `fill` / `values` / `file`, values rounded to bf16; a bundle without a bf16 buffer is
unchanged, so the ABI stays backward compatible). Each op declares which of its buffer args are `i32` (checked at compile time, e.g. `ids`, `tgt`, `step`)
and which may be `bf16` (`anydt` in `runtime/ref_*.py`: the matmul pair `matmul_tile_bf16` / `matmul_mma_bf16`, `round_bf16`, and the storage-aware norms, rope,
gelu_bwd, residual adds, embedding, colsum, cross entropy); any other op given a bf16 buffer is a compile error and, on the device, `E_DTYPE`.
Unknown keys, ops or args, missing/extra/mistyped args, int32/float32 overflow, NaN/inf (unless `--allow-nonfinite`), values
below an op's minimum, and tasks that touch more elements than their buffer holds are `CompileError`s naming `graph/task`
and the argument. `rows`/`cols` default to 1 (a scalar buffer).

Initial contents (`init`): `{"kind":"fill","value":v}`, `{"kind":"values","values":[...]}` (small buffers), and for
large ones, without huge JSON:

| kind | JSON | metadata for the executor / host |
|---|---|---|
| `seeded` (f32) | `{"seed":S,"scale":c,"dist":"normal"\|"uniform"}` | `seed, scale, dist, draws, bias, coef` |
| `file` | `{"path":"data/w.f32"}` little-endian f32 (or i32), `rows*cols` elements | `path`, copied into the bundle at that relative path |

`seeded` is counter-based (SplitMix64 at counter `i*draws+d`, `u = (z>>40)*2^-24`, `value = ((acc - bias) * coef)`, `acc` = `u`
or `(u0+u1)+(u2+u3)`, one rounded float32 operation per step, no fma); exact formulas in `runtime/initgen.py`. `normal` is an
Irwin-Hall sum of 4 (std `scale`, bounded tails), `uniform` is U(-scale, scale). Any consumer can fill element `i` alone and
get the CPU reference's bits. `file` paths must be relative without `..`; the compiler reads them from `--data-dir` (default:
the workload's directory) and `write_bundle` copies them into the bundle.

## Bundle (all little-endian)

| File | Content |
|---|---|
| `tasks.bin` | `task_count` x 64-byte `KvmTaskDesc` in header field order |
| `successors.bin` | int32 batch-local task indices, CSR via `succ_off`/`succ_cnt` |
| `args.bin` | int32 words; `f32` args are bit-cast; a task's words start at `args_off`, count = operator arity |
| `metadata.json` | `abi_version, task_count, edge_count, args_words, task_names, initial_ready, operator_ids, buffers (name, index, rows, cols, dtype 0/1/2, init), variables, publishers`, plus `reads` (`task, var, policy, version, observed, gate`) and `final_versions` |

`workload_id` = index of the graph. Output is byte-identical for identical input.

## Weight versioning

A variable is a monotone version counter, starting at `base_version` (default 0; `--base-versions var=N` /
`compile_spec(spec, base_versions=...)` when a later workload is admitted after live versions moved).

```
 version:      0 ----- upd1 ----> 1 ----- upd2 ----> 2
 exact 1 reader:         [ reads v1 ]        (upd2 waits for the reader: edge reader -> upd2)
 at_least 2 reader:                                   [ runs once version >= 2 ]
```

- **Publishers** of one variable are ordered by program order (graph order, then task order) and get `base+1, base+2, ...`.
  Consecutive publishers must be connected by a dependency path, else `unordered publishers`.
- **`at_least N`**: `wait_var/wait_ver` = the variable and N; runnable once `version >= N`. No edge is added.
- **`exact N`** (snapshot isolation): the same wait, plus an edge reader -> publisher of N+1, so the next update cannot
  overwrite data while the reader runs. Rejected when `N < base_version` (stale), when N+1 is already ordered before the
  reader, or when N was never going to be published.
- **One wait slot, one publish slot** per descriptor. Extra waiting reads become `noop` tasks `__gate/<graph>/<task>/<k>`
  (same priority/deadline) that carry the wait and precede the reader; the reader's own slot carries the first. A task that
  reads and publishes needs no gate.
- A wait for a version above `base` that nothing publishes, or that is published only after the waiter, is a compile error
  (deadlock). Version 0 is never waited on. For a nonzero `base`, the first publisher waits for `version >= base`, so batches
  publish in order.

`metadata.reads[].observed` is the version an `exact` reader must see (null for `at_least`: any `>= version`).

## CPU reference and golden output

`python3 runtime/workload_vm.py wl.json [--trace] [--deterministic]` runs the compiled program with threads, priority with
aging, deadlines as tie-break, and dynamic `Runtime.submit()` (each submission is compiled against the versions projected
from everything already admitted). `--trace` logs `(task, observed versions)`. It checks `exact` reads at start and finish and
fails the task on a violation.

`python3 runtime/compile_workload.py wl.json out --emit-golden` also runs a deterministic CPU schedule and writes
`golden.json`: final buffers, final versions, and each reader's observed version.

Operators with no CPU model (`doom_*`, the comm ops) are *opaque*: the VM refuses them unless run with
`allow_opaque` (`--allow-opaque`), then executes them as noops after checking their declared extents. Nothing silently skips an op.

GPT operators (`ref_gpt.py`) compute reductions and transcendentals in double and round every stored value to float32
(AdamW follows the device's float32 step order). That matches the device to its documented tolerance (norms, activations, loss,
optimizer 1e-5; attention 2e-5; TF32 matmul 4e-3), not bitwise. `sample` and `argmax` are exact: same Philox-4x32-10 stream.

Core-op float32: every operation rounds to float32, reductions run in index order, multiply-add is not fused. Integer-valued data
matches a GPU bit for bit; other data can differ by 1 ulp per fused operation.

## Compile cost

Roughly 15 us per task plus 0.3 us per edge: a 315k-task, 6.6M-edge graph (the 162M GPT tiling shape) compiles in about 5 s
plus 3 s to parse the JSON, peaking near 1.2 GB (mostly the JSON itself). The hot path is a per-operator decode plan,
array-based CSR graph, one topological pass for cycle detection and a topological-position bound for reachability queries, and
the cyclic GC is paused during compilation.

## Honest limits

- The CPU GPT reference is pure Python: fine for tiny shapes and per-op diffs, hopeless for the 162M model (use the
  frontend's own float64 VM or the GPU as the oracle there). `seeded` generation costs ~10 us/element in the reference.

- No automatic buffer hazard detection: ordering of buffer reads and writes is the author's job (`after` or versions).
  Only version-variable ordering is checked.
- `exact` isolates a reader only from publishers in its own batch. A later batch that publishes the same variable can run
  while an earlier batch's exact reader is still running; the CPU VM reports that as `snapshot violated`, the GPU does not
  check it.
- Waits are conservative across batches: a read of an already-published version is still emitted as a (satisfied) wait.
- `golden.json` buffers are schedule-independent only if the workload orders conflicting accesses and its `at_least`
  readers can observe a single version (the live example does).
- `runtime/lower.py` (counter waits/signals -> DAG, used by the queue integration) is separate and unchanged.
- `.gitignore` hides `kvm_*` build outputs; `!include/kvm_*` keeps the headers and `.def` files tracked. `tests/gpu/kvm_ops_test.def|.cuh`
  were once lost to that pattern and are force-added.

## Status of the layers this document describes

| Layer | State | Evidence |
|---|---|---|
| ABI v2, compiler, CPU VM, golden, `seeded`/`file` init | implemented + tested: 145 Python tests (about 3 s); an ABI v2 bundle runs on the A100 and matches its golden 10/10 buffers bitwise | `tests/`, `results/2026-11-03-a100-exec` |
| Executor (`kvm_exec.cu`) consuming bundles: admission, slot recycling, version waits, error codes | implemented + measured: 8582 checks, 0 failed (two recorded runs); recycling streams a 2.1M-task DAG through 65,536 slots | `results/2026-11-04-a100-exec-recycling`, `results/2026-11-04-a100-sched-integrated` |
| Known executor failure | `recycle` long-running admission (3M tasks) failed 2 of 3 runs once on the adaptive tree, same flake on its base commit; not root-caused | `results/2026-11-04-a100-adaptive` |
| Cross-bundle `exact` isolation | not built (documented limit above); the GPU does not check it | this file |
| Several admissions / `base_versions` from MANIFEST.json in the CLI | not consumed by `kvm_exec`; the CPU runner `kvmfe.run_manifest` is the reference | `results/2026-11-04-a100-exec-recycling` |
| Registry packs | core 0-31, GPT 32-63 (`ref_gpt.py`), comm 64-95 and Doom/game ops (opaque on the CPU), extra 96-122 (`ref_extra.py`) | `include/kvm_ops_list.inc` |

## Run the tests

```bash
python3 -I -m unittest discover -s tests -v        # 145 tests; the frontend has its own: ... -s tests/frontend (59 tests)
```

GPU entry points, runtimes and lock behaviour: [RUNNING_TESTS.md](RUNNING_TESTS.md).
