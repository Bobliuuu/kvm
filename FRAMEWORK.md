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
| Operators: name, id, `arg:type` list (`buf`/`i32`/`f32`) | `include/kvm_ops.def` | `runtime/abi.py` |
| What an op does + buffer extents it needs | `runtime/ref_ops.py` | compiler (bounds), CPU VM |

Add an op: append an `X(...)` line to `kvm_ops.def` (never renumber), add its CUDA body, add its entry to `ref_ops.OPS`
(a test fails until you do).

## Workload schema

```json
{ "variables": ["weight"],
  "buffers":   [{"name": "w", "rows": 1, "cols": 1, "init": {"kind": "fill", "value": 0.5}}],
  "graphs":    [{"name": "train", "tasks": [
      {"id": "upd", "op": "sgd", "args": {"w": "w", "g": "grad", "lr": 0.0625, "n": 1},
       "after": ["bwd"], "priority": 0, "deadline_us": -1, "publishes": "weight",
       "reads": [{"var": "weight", "policy": "exact", "version": 1}]}]}] }
```

Buffer args are buffer names (resolved to `KvmBuf` table indices = declaration order); `i32`/`f32` are JSON numbers.
Unknown keys, ops or args, missing/extra/mistyped args, int32/float32 overflow, NaN/inf (unless `--allow-nonfinite`), values
below an op's minimum, and tasks that touch more elements than their buffer holds are `CompileError`s naming `graph/task`
and the argument. `init.kind` is `fill` or `values`; `rows`/`cols` default to 1 (a scalar buffer).

## Bundle (all little-endian)

| File | Content |
|---|---|
| `tasks.bin` | `task_count` x 64-byte `KvmTaskDesc` in header field order |
| `successors.bin` | int32 batch-local task indices, CSR via `succ_off`/`succ_cnt` |
| `args.bin` | int32 words; `f32` args are bit-cast; a task's words start at `args_off`, count = operator arity |
| `metadata.json` | `abi_version, task_count, edge_count, args_words, task_names, initial_ready, operator_ids, buffers, variables, publishers`, plus `reads` (`task, var, policy, version, observed, gate`) and `final_versions` |

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

Float32: every operation rounds to float32, reductions run in index order, multiply-add is not fused. Integer-valued data
matches a GPU bit for bit; other data can differ by 1 ulp per fused operation.

## Honest limits

- No automatic buffer hazard detection: ordering of buffer reads and writes is the author's job (`after` or versions).
  Only version-variable ordering is checked.
- `exact` isolates a reader only from publishers in its own batch. A later batch that publishes the same variable can run
  while an earlier batch's exact reader is still running; the CPU VM reports that as `snapshot violated`, the GPU does not
  check it.
- Waits are conservative across batches: a read of an already-published version is still emitted as a (satisfied) wait.
- `golden.json` buffers are schedule-independent only if the workload orders conflicting accesses and its `at_least`
  readers can observe a single version (the live example does).
- `runtime/lower.py` (counter waits/signals -> DAG, used by the queue integration) is separate and unchanged.
- `include/` must be tracked in git: the repo `.gitignore` pattern `kvm_*` currently matches `include/kvm_ops.def` and
  `include/kvm_task_abi.h` (needs a `!include/kvm_*` exception).

## Run the tests

```bash
python3 -I -m unittest discover -s tests -v
```
