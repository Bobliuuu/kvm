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
python3 -m kvmfe.demo --train-steps N --serve-every K --game-frames F --out DIR   # train / serve_* / game / demo_combined .json
python3 -m kvmfe.inspect --model gpt_162m --steps 1 --tiling a100                  # tasks, edges, critical path, memory plan
python3 -m kvmfe.check_defs                                                        # .def registry vs frontend references
python3 -I -m unittest discover -s tests/frontend                                  # from the repo root
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
* Multi-step: the N steps are unrolled in one graph (window = the whole run). Step s+1 depends on step s only through regions.
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

## Honest limits

* Not run on a GPU yet. The golden path is the CPU VM; the GPU must match it (fp32/TF32 tolerance), not bit-for-bit.
* Needs from the compiler/ABI (emitted as shown, rejected by `compile_workload.py` until added): `dtype: "i32"` on index
  buffers; `null` for an absent optional buffer arg; loading `kvm_ops_gpt.def` and `kvm_ops_game.def` (it reads `kvm_ops.def`
  only); initial values too large for JSON (the 162M weights are emitted zero-filled with a warning: needs a host fill or an
  init-RNG buffer kind / op).
* No data-gather op: each step's token / target buffers are host-provided init values.
* No loop opcode: steps are unrolled (task count grows linearly with steps).
* Absolute learned positions are not supported (the registry's embedding ops pair with RoPE); no GQA, no dropout, no gradient
  clipping (`gradnorm_finish` emits the clip coefficient in `out[1]` while `adamw` reads `gscale[0]`), no mixed precision.
* Serving assumes `attn_fwd` clips a final partial 32-row block when P-1 is not a multiple of 32.
* Greedy decoding only; the `sample` op has no CPU reference here.
* The CPU VM is pure Python: golden runs are for tiny configs (20 steps of `gpt_tiny` take seconds). The 162M config is compiled to
  a DAG and inspected, not executed.
