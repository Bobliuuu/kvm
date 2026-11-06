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

Beyond GPT, `models.py` holds a Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU, tied or untied head, optional top-k MoE feed-forward
with capacity, gather/scatter dispatch and a load-balance loss), an MLP classifier and a conv-free ViT. They are built only from `Builder`
ops, so autodiff, fusion, rematerialization, gradient accumulation, LR schedules, loss scaling and bf16 apply to all of them
(section "Model zoo" below).

Any traced module can be trained, not only the shipped ones: `compile_training(module, loss=MSE(), data=ArrayData(..),
optimizer=OptimConfig(..), steps=N)` (section "Generic training" below). `ModelConfig` stays the GPT description; the generic path never touches it.

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

## Status

| Piece | State | Evidence |
|---|---|---|
| Builder, autodiff, tiling, dataflow, memory plan, emit (GPT, Llama/GQA, MoE, MLP, ViT) | implemented + tested: 78 tests (about 35 s) against independent numpy float64 oracles, finite-difference gradients, any-topological-order bitwise state | `tests/frontend/`, `results/2026-11-04-a100-models` |
| Fusion, remat, gradient accumulation, device LR schedule, loss scaling, bf16 | implemented + measured on the A100 (fusion -3% Llama / -11% MoE time, remat pool -41 to -59% for +36% tasks); bf16 is emulated in fp32 buffers | `results/2026-11-04-a100-models` |
| GPT bundles on the executor | implemented + measured; 1xTF32 GEMMs are compared on losses and step-1 buffers, 3xTF32 (flag 16, `gen_bundle.py --precise`) to fp32 level | `results/2026-11-04-a100-gpt-golden` |
| Generic training (`generic.py`): any traced module + `CrossEntropy`/`MSE` + `ArrayData`/`GeneratedData` + SGD/AdamW | implemented + CPU-tested: 4 never-shipped models (regression MLP, residual RMSNorm MLP, 1-D CNN, transformer encoder) vs numpy float64 oracles. **A100 run NOT yet done** (`tests/generic/run.sh`: bundles vs CPU golden at 3xTF32, 2 MANIFEST demos through `demo/kvm_run`; the box's GPU lock was held by an idle session for 3+ h); until it passes, the MSE device path (`scale_flat`/`row_dot`/`colsum`) and the gather-based conv are verified only against the CPU reference | `tests/frontend/test_generic.py`, `tests/generic/` |
| Windows / MANIFEST runner | CPU reference runner only; the GPU CLI runs single bundles | `kvmfe.run_manifest`, `results/2026-11-04-a100-exec-recycling` |
| `gpt_162m` | code-only: compiled (318k-task template) and inspected, never run on a GPU | this file |
| Not built | dynamic shapes, MoE serving, dropout, fp16, a frontend switch for 3xTF32, 128x128 / 256x128 matmul tile stepping in the lowering (the device tiles exist) | below, `results/2026-11-04-a100-ops-perf` |
| Open test item | `norm_bwd_params` wants >= 64 tasks (16 columns each) to use the GPU; the partition here does not yet give that | `results/2026-11-04-a100-ops-perf` |

Run the frontend tests: `python3 -I -m unittest discover -s tests/frontend` (repo root). Everything else: [RUNNING_TESTS.md](RUNNING_TESTS.md).

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

## Model zoo and generality (nothing here is a hand-written kernel)

`models.py` implements the training protocol `train.build_training` consumes: `cfg`, `specs`, `batch_streams(micros)`,
`losses(b, batch, w, grad_scale)`. The extra generic op pack `include/kvm_ops_extra.def` (ids 96-121; `.cuh` device code,
`runtime/ref_extra.py` float32 golden, `opset.py` float64 oracle semantics) adds only shape-parametrized building blocks:
activations (relu/silu/tanh/sigmoid), SwiGLU, row softmax, `gather_rows_scaled` / `scatter_add_rows` / `row_dot` (each other's VJP),
MoE routing (`moe_route`, `moe_gates_*`, `moe_balance_*`), bidirectional attention, bf16 matmul and rounding, `adamw_lr` /
`lr_schedule`, `add_rowtile`, and three matmul-epilogue wrappers used by fusion.

| Model (preset) | Built from |
|---|---|
| `llama_tiny`, `llama_gqa_tied` | `rmsnorm`, `rope` (q and k with their own head counts), `attention(kv_heads=..)` (GQA), `silu_mul`, `linear(wt=True)` for the tied head |
| `moe_tiny` | `softmax` -> `moe_route` -> per-expert `gather` (idx0 = e*C) -> SwiGLU expert -> `scatter_add` with `moe_gates` as scale; `moe_balance` is a second backward root |
| `mlp_tiny` | `linear`, `act` (relu), `ce_loss` on Gaussian clusters |
| `vit_tiny` | patch embedding = `linear` over pre-cut patches, `add_rowtile` (learned positions), `attention(causal=False)`, mean pooling = `scatter_add` |

Checks (`tests/frontend/test_models.py`, `oracle_models.py`): the CPU VM's loss equals an independent numpy float64 forward
(rel 1e-9); the autodiff gradient of every parameter tensor (largest element plus a pseudo-random one) matches a central finite
difference of that oracle; any topological order gives bitwise identical state; KV-cache decode (GQA, RoPE offset) equals
full-forward greedy decode. GPU: `examples/models/run.sh` (bundles, one `flock` hold, `check.py`).

Frontend features added with them:

* **Gradient accumulation** (`TrainConfig.accum`): K microbatches per optimizer step; backward ops accumulate in place into the
  gradients of the previous microbatch (`autodiff.backward(accumulate=..)`), each loss root is scaled 1/K; the stream window
  of microbatch m is `select_window(stride=K*rows, off=m*rows)`.
* **Rematerialization** (`with b.remat():`, `LlamaConfig.remat`): the region's forward ops are re-emitted right before their
  backward with fresh tensors; the originals die at their last forward use, so the planner reuses their memory. Results are
  bitwise identical, the pool is smaller, the task count grows by the recomputed forward.
* **LR schedules** (`TrainConfig.schedule`): `lr` is a device buffer written each step by an `lr_schedule` task from the step
  counter; `adamw_lr` reads it. The program text of every step stays identical (window templates still work).
* **Loss scaling + bf16** (`precision='bf16'`, `TrainConfig.loss_scale`): matmuls run on bf16 operands with fp32 accumulation
  (`matmul_tile_bf16`), activations are stored rounded (`cast`), parameters stay fp32 masters; the backward starts from S*dL and the
  clip coefficient (AdamW's gscale) is multiplied by 1/S. bf16 values live in fp32 containers: the ABI v2 dtype ids (0 f32, 1 i32) are
  unchanged for that emulated path, so it saves no memory and has no fp16/overflow-skipping dynamic scaling.
* **Real bf16 storage** (`KvmBuf.dtype` 2, 2 bytes per element; `precision='bf16-real'` on `generic.MLP`, or `b.cast(x, 'bf16', real=True)` +
  `linear/mm(bf16=True)` by hand): weight copies (`Linear(bf16='real')` casts the fp32 master each call; the master gets the unrounded
  gradient) and between-layer activations live in bf16 buffers. Ops: `cast_f32_bf16`, `cast_bf16_f32`, `matmul_tile_bf16s` (A and/or B
  bf16 storage, flags 128/256; same tile code as `matmul_tile_bf16`, so results are bit-identical to the emulated path). Everything
  else (norms, softmax, optimizer, losses) stays fp32 and refuses bf16 buffers (`BAD_BUF`/compile error); `cast(x_bf16, 'f32')` widens.
  The memory plan counts 2 bytes per bf16 element. Not covered: bf16 parameters/optimizer state, bf16 attention, fp16.
* **Tied embeddings** (`linear(wt=True)`), **GQA** (`attention(kv_heads=)`, `kv_append/attn_decode` with cache heads), **multiple
  loss roots** (cross-entropy + auxiliary), **float input streams** (`select_window(cols=)`).
* **Fusion** (`fuse.py`, `compile_training(fuse=True)`): `gelu_bwd(mm(..))` -> `mm_dgelu`, SwiGLU up-projection + gating ->
  `linear_silu_mul`, SwiGLU backward -> `mm_silu_mul_bwd`. Only intermediates with a single reader are fused and a fused tile's
  reference is the unfused reference, so the CPU result is bitwise equal (tested). Bias+GELU and the residual add were already
  fused in the `matmul_tile` epilogue.

## Generic training (any traced module, pluggable loss and data)

Before this, `compile_training` took a GPT/zoo config, `build_training` hard-wired a token stream and softmax cross-entropy, and
`ModelConfig` rejected anything that was not attention-shaped (head dim 16/32/64/128, seq a multiple of 32) even for an MLP. The audit
(`results/2026-11-04-claims-audit`, gap 7) had to call `build_training` / `lower_program` / `to_workload` by hand. Now:

```python
from kvmfe.generic import MLP, MSE, CrossEntropy, ArrayData, GeneratedData, Conv1D, SequencePool
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
c = compile_training(MLP('reg', [11, 24, 13, 3], act='tanh'),            # any module: .specs + __call__(b, *inputs, w)
                     loss=MSE(), data=ArrayData(6, {'x': X, 'y': Y}),    # batch 6 per step, wraps around the samples
                     optimizer=OptimConfig(kind='sgd', lr=0.05), steps=20, tiling=Tiling.preset('a100'))
c.run()                                                                  # CPU golden;  c.spec -> workload JSON -> bundle
demo.write_training_demo('out/', module, loss=.., data=.., steps=200, window=10, precise=True)   # MANIFEST.json + bundles for demo/kvm_run
```

| Piece | Contract |
|---|---|
| module | `.specs` ({name: `ParamSpec`}) and `__call__(b, *inputs, w) -> Tensor`; inputs are the batch tensors named by `data.inputs`. `nn.Module` subclasses compose (`Linear`, `Norm`); `generic.MLP` (act, residual, norm), `Conv1D` (odd taps, zero 'same' padding: row `gather` per tap feeding a `linear` that accumulates the previous taps), `SequencePool` (mean per sequence = `scatter_add`) are provided |
| loss | `CrossEntropy` (i32 class column; ids range-checked on the host) or `MSE`. `MSE` is a new logical op built from existing device ops only (`scale_flat`, `residual_add`, `row_dot`, `colsum`; no new CUDA), VJP `mse_bwd` = `scale_flat` of the saved difference. Subclass `Loss` for others |
| data | `ArrayData(batch, arrays, dtypes)` (epochs by wrap-around) or `GeneratedData(make(micro), shapes)`; shapes, dtypes, NaN/inf and ragged rows are checked before anything is emitted. The batch of step s is selected on the device by the step counter, as for GPT |
| optimizer | `OptimConfig` (AdamW or SGD); `tc=TrainConfig(...)` still carries accum, schedule, clip, loss scale, windows |
| refusals | `FrontendError` naming the shape/dtype/argument: target shape, class ids out of range, ragged data, missing `loss=`/`data=`, generic-only arguments on a known model, shape mismatches inside a module (from the `Builder`) |

The GPT path is the same code: `compile_training(GPT(cfg), tc, tiling, loss=CrossEntropy(), data=<token ArrayData>)` equals the
built-in run bit for bit (`test_gpt_through_the_generic_api_equals_the_builtin_path`), and `compile_training(cfg, tc, tiling)` is
unchanged. Checks: `tests/frontend/test_generic.py` (19 tests: loss and finite-difference gradients of every parameter against numpy
float64 oracles, any dataflow order bitwise equal, accumulation == big batch, manifest through `kvmfe.run_manifest`, refusals) and
`tests/generic/run.sh` (A100, not yet run: 4 models x {cpu, a100} tiling, 3xTF32, golden_check precise; 2 MANIFEST demos through `demo/kvm_run` and
`demo/check_demo.py`).

### Reinforcement-learning losses and attention-free models (generality step 1)

| Piece | Contract |
|---|---|
| `PolicyGradient()` | REINFORCE: `mean(adv * -log softmax(out)[action])`. Targets `('act', 'adv')` = i32 action column + f32 advantage column (data: no gradient). New logical op `pg_loss` built from existing device ops (`softmax_ce_fwd/reduce/bwd`, `row_dot`, `colsum`, `gather_rows_scaled`); with `adv == 1` it equals `CrossEntropy` bit for bit (tested) |
| `QLearning()` | one-step TD / DQN: `mean((Q(s,a) - y)^2)`. Targets `('mask', 'y')` = one-hot f32 action mask (`generic.one_hot`) + bootstrapped target column (computed on the host from a frozen copy: data). New logical op `row_select` (`row_dot` of each Q row with its mask; VJP = scaled `gather` of the mask) feeding `mse_loss` |
| data | `DataSource.target` may be a tuple of stream names; `ArrayData` serves logged transitions, `GeneratedData(make(micro))` serves host-simulated rollouts |
| on-policy loops | one bundle never samples from its own policy: the loop is on the host (roll out with the latest weights, feed the batch, run a bundle window). The device program is the update half |
| `ModelConfig(layers=0)` | attention geometry (head dim, seq % 32) is validated only when there are attention blocks; the refusal for a bad head dim now points at the generic API. Models without attention are written as a `Module` and use `compile_training(module, loss=, data=)`, never a `ModelConfig` |

The audit's four models (`xfmr_odd`, `mlp_ce`, `resmlp_rms`, `bad_dims`) now run through `compile_training` with no workaround:
`xfmr_odd` as a `ModelConfig`, the three token MLPs as zoo cases (`tests/generic/zoo.py`: `mlp_ce`, `resmlp_rms`, `bad_dims`, plus
`pg_policy`, `dqn_q`). CPU checks: `tests/frontend/test_generic_rl.py` (loss and finite-difference gradients vs numpy float64, learning, refusals).

Limits of this layer (not fixed here): no auxiliary-loss roots from inside a generic module (use a `models.py`-style model for those),
one `loss` root per program, no dynamic shapes or control flow, no real conv/pool kernels (the CNN is gather + matmul: correct, not
fast; a 3-tap conv reads its input 3 times), attention shapes are still bound by the device ops (head dim 16/32/64/128, sequence a
multiple of 32), and nothing here made small models faster than PyTorch (the audit's 3.9-5.4x loss to a CUDA graph on small
dependent work is a property of the VM, not of the frontend).

## Cost model (`inspect`)

Task time = max(FLOPs / 1.8e11, bytes / 1.4e10) + 5 us hop (one A100 SM's share of fp32 CUDA-core peak and HBM bandwidth;
a dependency hop through global memory, per AGENTS.md); opaque engine ops use their declared time (Doom tick 6.7 ms, v1's
measurement). Critical path = longest path with these times. "List schedule" = greedy simulation on 108 workers. It ranks
designs; it does not predict a GPU run, and it ignores tensor-core speed-ups and L2 locality.

## Data-dependent control (predicated ops, branches, bounded while loops)

```python
fl = b.flags('p', 3)                                   # persistent i32 predicate elements
p = b.pred_cmp(x, '>', 0.0, index=0)                   # p = (x[0] > 0), computed on the device; AND/OR via prev=, comb='and'|'or'
with b.when(p):            b.scale_scalar(y, 2.0)      # branch arms: exactly one runs; the task graph is the same for every input
with b.when(p, negate=True): b.scale_scalar(y, -1.0)
flags = b.while_loop(16, lambda b: (x, 0, '>', 1.0),   # while x[0] > 1.0 (device-evaluated), at most 16 iterations
                     lambda b, i: b.scale_scalar(x, 0.5))
```

`when` stamps every op emitted inside with a predicate; lowering turns each tile task into a predicated task (FRAMEWORK.md "Data-dependent
control"), adds the predicate element as a read and the op's outputs as reads too (a skipped task leaves them unchanged, so they are
read-modify-write for dataflow). `while_loop` unrolls the bound, and returns the `i32 1 x (max_iters+1)` flags tensor (`flags[i]` = iteration
i ran; `flags[max_iters]` = cap hit with the condition still true). Predicated ops are not differentiable (clear `FrontendError`), are not fused,
cannot nest, and write nothing when skipped: use persistent (`state`) tensors or in-place updates for what the region changes, since a fresh
activation tensor of a skipped op holds whatever its (possibly reused) buffer held. Checks: `tests/frontend/test_control.py` (CPU, program
and random dataflow order, compiler ABI encoding, CPU runtime VM) and `tests/ctl/run.sh` (A100, bundles with data-dependent trip counts
and branches against the CPU VM golden, executor skip counts).

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

* **Static shapes and a static DAG, by design.** A program is a fixed set of tasks: no dynamic shapes, no data-dependent control flow
  at the frontend level. What is data-dependent lives in buffers: `select_window` (batch picked by the step counter), `lens` in decode,
  `moe_route` (expert choice and capacity are computed on the device, but the expert matmuls always run `capacity` rows, padded with
  zero rows), `lr_schedule`. Loops are unrolled or re-admitted as windows; a `while`/`if` over device values is predicated (see
  "Data-dependent control"), but there is no early-exit decoding.
* MoE routing ops are single-task over all N tokens of a microbatch (O(N*E*k) serial work, fine up to a few thousand tokens); dropped tokens
  (over capacity) get no expert and keep only the residual, as in Switch Transformer.
* bf16 is *emulated storage*: values are bfloat16-representable but live in fp32 buffers (ABI v2 dtype ids unchanged). No fp16, no dynamic
  loss scaling (no overflow detection or step skipping).
* Fused tiles save a task and a dependency hop per tile, nothing more: the matmul epilogue cannot hold a full row, so norms are never
  fused into a matmul.
* The GPU matches the CPU golden to fp32 level only with 3xTF32 GEMMs (matmul flag 16; `gen_bundle.py --precise`); 1xTF32 runs are compared on
  step-1 activations/gradients and losses (see `results/2026-11-04-a100-gpt-golden`). The frontend still has no flag for flag 16.
* `gather_rows` (id 63) is unused; the extra pack's `gather_rows_scaled` replaces it (padding rows are zeros instead of errors).
* Per-step losses of a window are pinned buffers `loss.<i>` reused by the next window: a host reads them between windows.
* A generic training program has exactly one loss root (`loss`); extra roots need a `models.py`-style model. Dropout is not implemented (needs a counter-based RNG op); SGD does not clip; serving is greedy; MoE serving is not supported (routing
  capacity is defined per training microbatch).
* Doom ops have no CPU reference (opaque; `--allow-opaque` runs them as no-ops); the real engine is the port.
* The CPU VM is pure Python: golden runs are for tiny configs (the model tests take minutes). The 162M config is compiled and inspected,
  not executed.
