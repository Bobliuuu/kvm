# FRONTEND.md: from a model description to a task DAG

`frontend/kvmfe` is the Python layer above the scheduler. It turns a declarative model config into the workload JSON that
`runtime/compile_workload.py` packs into an ABI v2 bundle for the persistent kernel. The scheduler below it sees tasks, buffers,
edges and weight versions. It has no notion of train, serve or game, and nothing in it is specific to one model.

```
 ModelConfig / GameConfig / ServeRequest              declarative, JSON-able (presets: gpt_tiny, gpt_small, gpt_162m)
        |
   nn.py  Linear Norm CausalSelfAttention MLP TransformerBlock GPT       module builders, params are specs
        |  call(builder, weights)
   ir.py  Builder -> logical ops in program order (sequential semantics)
        |        autodiff.py   @vjp rules compose registered logical ops -> backward ops, in-place grad accumulation
        |        train.py      N steps unrolled, data -> fwd -> loss -> bwd -> AdamW/SGD -> weight-version publish
        |        serve.py      prefill + KV-cache decode per request      game.py   doom_tick -> doom_present row blocks
   lower.py   logical op -> tile tasks (device op + typed args + regions read / written), sizes from `Tiling`
        |
   plan.py    buffer lifetimes -> slot reuse (same shape), pinned buffers for weights / state / requests
   dataflow.py  regions -> RAW / WAR / WAW edges, tile granularity, no per-model edges anywhere
        |
   emit.py    workload JSON {variables, buffers, graphs[tasks{id, op, args, after, priority, deadline_us, publishes, reads}]}
        |                                   |
   runtime/compile_workload.py (bundle)   refvm.py  CPU golden VM (float64) = oracle for the GPU run
```

Tools (run from `frontend/`, or put it on `PYTHONPATH`):

```
python3 -m kvmfe.demo --all --out DIR                  # train windows + serving requests + Doom -> compiled bundles + MANIFEST.json
python3 -m kvmfe.run_manifest DIR --allow-opaque       # reference host runner: executes DIR on the CPU VM from MANIFEST.json alone
python3 -m kvmfe.inspect --model gpt_162m --steps 1 --tiling a100   # tasks, edges, critical path, memory plan (no JSON)
python3 -m kvmfe.check_defs                            # .def registry vs frontend references
python3 -I -m unittest discover -s tests/frontend      # from the repo root
```

`--model` is a preset name, a JSON file or a JSON string (`ModelConfig` fields). The 162M-parameter GPT is a config
(`gpt_162m`: 12 layers, d768, 12 heads, seq 1024, vocab 50304, batch 4, RoPE, untied head), not code.

## How the pieces answer "no framework, just a hardwired kernel"

| Need | Where it lives |
|---|---|
| any model from ops | `Builder` methods + `nn.py` modules; a new architecture is a new module, nothing else changes |
| backward pass | `autodiff.py`: one `@vjp` per differentiable logical op, composed of other registered ops |
| fast without a hand-written kernel | every task is a registered device op (`include/kvm_ops*.def`); the frontend only chooses tiles and edges |
| fine-grain overlap | edges come from tile regions, so a step's tile starts when the tiles it reads are done, across layers and steps |
| train + serve + game together | three builders, three graph names, one scheduler; they meet only at the `weights` and `frame` variables |

## Add an op in 3 places

1. `include/kvm_ops_*.def`: `X(name, id, "arg:type,...")` (ids append-only) and the CUDA implementation. The frontend reads
   signatures from the `.def`; it never repeats them.
2. `kvmfe/opset.py`: `@impl('name', flops=...)` with the float64 CPU reference (this is the golden path the GPU is diffed against).
3. `kvmfe/lower.py`: `@lowers('logical_kind')` yielding `_task('name', reads=[regions], writes=[regions], **args)` per tile; add a
   `Builder` method in `ir.py` for shape inference; if differentiable, `@vjp('logical_kind')` in `autodiff.py`.

`check_defs` flags ops the frontend references that the `.def` does not define. A wrong argument set is rejected at lowering,
an under-declared region shows up as a result change between program order and random topological order (tests do this).

## Tiling and dataflow rules

* Device-fixed tiles: `matmul_tile` is 64x64, attention is 32 query rows (`attn_bwd_dkv`: 32 keys). Everything else is a `Tiling`
  field (`rows`, `ew_rows`, `cols`, `vocab`, `opt_elems`); presets `cpu`, `a100`, `b200`.
* A task declares the rectangles it reads and writes (`lower.py`). `dataflow.py` keeps, per buffer, the last writer and the readers
  since, as disjoint rectangles (partial overwrites split them), so edges are exact for the declared regions and do not grow with
  history. RAW, WAR and WAW all become edges; there are no hand-written edges. When the same large predecessor set (>= 32
  tasks) is needed twice (e.g. 50k logits-GEMM tiles that read the LM-head weights, then 590 optimizer chunks that overwrite
  it) a no-op `join` task is inserted, turning R x W edges into R + W (it costs one dependency hop). Fuzz-tested against the
  naive all-pairs hazard set.
* Fusion is the device ops' own: bias, GELU (with the pre-activation kept for backward) and the residual add run in the
  `matmul_tile` epilogue. A residual connection carries one gradient buffer through the network (`res` forwards the output
  gradient; later contributions accumulate in place with `acc`).
* Memory: weights, AdamW state, step counter, request buffers are pinned. Other tensors live from first to last logical op and share
  a buffer slot with any same-shape tensor whose lifetime ended more than `--gap` ops earlier (`--reuse none` disables it). Reuse
  never decides order: dataflow adds the WAR/WAW edges, so it is always safe, and `gap` buys overlap between steps with memory.
* Multi-step: S steps are unrolled in one program (S = N for a single bundle, see windows below). Step s+1 depends on step s only
  through regions. Gradients are globally clipped (`gradnorm_partial/finish` -> `adamw` gscale) unless `TrainConfig.clip <= 0`.
* Weight versions (`weights`): the update is in place, so each step publishes twice, `pre` (about to overwrite) and `post`.
  Stable weights after k updates are version 2k. A serving request reading `exact` 2k is ordered before `pre` of step k, so
  the compiler's rule "an exact reader precedes the next publish" protects the buffers; `at_least` just waits. A one-cell token
  chains post(s-1) -> pre(s) -> updates(s) -> post(s). The cost is real: an exact reader can hold the trainer back.
* Serving: the first P-1 prompt tokens are one prefill batch (K/V appended to the cache), then every step is a one-row decode
  (`attn_decode`, `kv_append`, `counter_inc` on the cache length). Greedy only (`argmax`).
* Game: `doom_tick(f) -> doom_status(f) -> doom_present(f, row blocks)` per frame, tick f+1 after tick f through the state
  buffer, `deadline_us = (f+1)/35 s`, frames double-buffered and published as versions of `frame`. Ops come from
  `include/kvm_ops_game.def`; if absent, building fails with the op name. The Doom engine itself is the port in `include/`.

## Cost model (`inspect`)

Task time = max(FLOPs / 1.8e11, bytes / 1.4e10) + 5 us hop (one A100 SM's share of fp32 CUDA-core peak and HBM bandwidth;
a dependency hop through global memory, per AGENTS.md); opaque engine ops use their declared time (Doom tick 6.7 ms, v1's
measurement). Critical path = longest path with these times. "List schedule" = greedy simulation on 108 workers. It ranks
designs; it does not predict a GPU run, and it ignores tensor-core speed-ups and L2 locality.

## Windows, loops and the manifest

Everything that differs between training steps lives in device buffers: `select_window` picks step s's tokens from a stream
buffer using the AdamW step counter (`counter_inc`), AdamW reads the same counter, the global-norm clip coefficient is a buffer.
So the text of a window of S steps is identical for every window: `--window S` emits ONE program and admits it N/S times
(S=1 is a loop template; a remainder window gets its own program). Only the version numbers a window publishes move, which the
manifest records as `base_versions` per admission. Compare with a fully unrolled graph: task count is S/N of it, the frontend runs once.

What the executor supports (feat/executor, 50405a4): bundles are admitted at runtime into a bounded task-slot pool with slot
recycling, and one huge bundle can be streamed through it in windows; bundles share state only through version variables and
named buffers. There is no loop opcode: the host re-admits the template bundle (compile with the new `base_versions`, or have the
executor add the version offset). Overlap between consecutive windows is therefore limited to what the executor streams within one
bundle; across bundles the manifest's `after_bundles` is a full barrier.

`MANIFEST.json` (version 1) is all a host runner needs; `kvmfe.run_manifest` is the reference implementation:

```
workloads{name: file, tasks, edges, priority, deadline_us[min,max], variables, graphs}
buffers{name: rows, cols, dtype, owner}        allocate once; initialize from the owner workload; others share it
shared_buffers[...]                            names declared by more than one workload (weights, optimizer state, counters)
admissions[{name, workload, steps, at_us, after_bundles[], when[{var, min, max?}], base_versions{}, publishes{},
            compile{workload, data_dir, base_versions}, bundle_dir|null}]
```

An admission starts when its time/after/when conditions hold (`when` = live version window of a variable); training windows chain
by `after_bundles`; a serving request at step k is `when weights >= 2k` (policy `at_least`, concurrent with the next window) or,
with `--serve-policy exact`, `when weights == 2k` plus the next window waiting for it (a host-level barrier: the compiler only
isolates `exact` readers from publishers in the same bundle). A request at step k must fall on a window boundary.

## Compiler integration

`emit.py` output is compiled by `runtime/compile_workload.py` unchanged (i32 index buffers, `null` optional args, all `.def`
packs, `seeded` parameter init, `file` token streams). Weights use `seeded` init (`runtime/initgen.py`, seed = crc32 of the
parameter name), so even 162M parameters need no JSON values; the stream of tokens is a `file` when it exceeds 4096 elements.
`tests/frontend/test_manifest.py` compiles gpt_tiny, runs the manifest on the compiler's float32 CPU reference and checks the loss
curve against the independent float64 oracle (rel. 2e-5) and served tokens against its greedy decode at each snapshot.

## Honest limits

* Not run on a GPU yet. The golden path is the CPU VM; the GPU must match it (fp32/TF32 tolerance), not bit-for-bit.
* `gather_rows` is unused: `embedding_fwd` already gathers; batches come from `select_window` over a host-provided token stream
  (sized total_steps x tokens; no on-device data loading beyond that).
* The learning rate is a constant baked into each task (a schedule would break template reuse; it needs an lr buffer in `adamw`).
* Per-step losses of a window are pinned buffers `loss.<i>` reused by the next window: a host reads them between windows.
* Absolute learned positions are not supported (RoPE only); no GQA, no dropout, no mixed precision, SGD does not clip.
* Serving assumes `attn_fwd` clips a final partial 32-row block when P-1 is not a multiple of 32; greedy decoding only.
* Doom ops have no CPU reference (opaque; `--allow-opaque` runs them as no-ops); the real engine is the port.
* The CPU VM is pure Python: golden runs are for tiny configs. The 162M config is compiled and inspected, not executed.
