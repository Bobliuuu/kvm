# KVM claims audit vs Flapping Airplanes (WIP draft, 2026-11-04)

Auditor X, branch review/claims-audit, merged with feat/integration 0d945bc. Status: WIP; the GPU benchmark groups are queued behind
other users on the shared A100 lock. Rows marked "pending" have no result yet.

## What FA publicly claims (no primary text beyond the tweet could be fetched; x.com returned 403)
- Tweet 4/5 (quoted in AGENTS.md, not re-fetched by me): a "kittens" VM that "takes over the whole GPU", fuses "entire training runs into a single kernel".
- TechCrunch 2026-10-21 (fetched): "the world's most complete megakernel framework"; "can perform entire training runs, serve inference, play video games, and even do all three simultaneously in a single kernel"; "can overlap memory-bound inference with compute-bound training".
- Interview snippet via a search result (MTS on X, not fetched, secondhand): co-founder says they do not use Torch, NCCL or CUDA's normal runtime; one kernel on one B200 trains, serves and plays Doom.
- No numbers anywhere. Everything else the coordinator listed (dynamic SM sharing, bandwidth arbitration, rebalancing, weight sync, multi-node without NCCL, trillion parameters) is INFERRED, not found in primary text.
- Context numbers (Hazy Research blogs, fetched): Llama-1B decode <680 us on B200, 78% of H100 bandwidth, >1.5x vs SGLang; Tokasaurus megakernel +22% tokens/s vs SGLang (Llama-70B TP, H100); launch 2.1 us / 1.3 us in a graph; global work queue +14.2%; NCCL replaced by in-kernel NVSwitch stores.

## Early findings (details to follow)
- New models through the frontend (a 3-layer 3-head rmsnorm transformer, a 2-layer MLP, a residual-rmsnorm MLP, an MLP with hidden 7) all compile and match the CPU golden on the A100 (weights rel L2 <= 1.4e-3, 3-step loss within 4e-4): logs/newmodel-*. Frontend limits: ModelConfig validates attention shape even for an MLP; build pipeline is LM-only (token stream + softmax CE); head dim must be 16/32/64/128, seq % 32 == 0.
- Executor, scheduler, ABI and compiler contain no model or game names (logs/hardcoding-grep.txt). Op registries are compile-time CUDA: every op is hand-written.
- Static SM split exists only as a baseline (--sched static / --shares; tests, doom/tests/run_sched.sh). Default CLI is scheduler off.
- Bandwidth arbiter is blind to GPT and game ops: include/kvm_ops_cost.inc has 11 entries (core ops only); every other op gets MIXED / 0 bytes and is never throttled.
- Python tests on 1d3c15a: 145 run, 1 failure + 2 errors, all because the comm registry (kvm_ops_comm.def) was not in that tree (re-check after merge).
