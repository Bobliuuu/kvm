# Litmus for the FIFO ready ring, and the two controls that never fired

GPU: NVIDIA GeForce RTX 3080 Ti (sm_86, 80 SMs), CUDA 12.8 (V12.8.93), every binary built for `sm_80`, the A100's target. Tree: this branch,
on the merge of `main` with #13 and #22. One run of `KVM_NO_LOCK=1 LIT_PREBUILT=1 tests/litmus/run_litmus.sh`, with one hold of the GPU.

Builds come from the tree's core headers: the gpt, extra, comm and game packs are left out, as `tests/gpu/build_core.sh` does. The litmus
kernels call the protocol functions directly, and the e2e scenarios use only `fill`, `spin`, `copy`, `add`, `scale` and `noop`, so the
code under test is the same at about a minute a build.

## What this adds

Every scheduler-off CLI run serves its ready tasks through one FIFO ring: `kvm_fifo_pop` on the consumer side, and `kvm_enqueue`'s
`fifo_only` publish on the producer side. A consumer's lane 0 takes a ticket on the head and watches that ticket's cell. A producer takes a
tail position and publishes the slot and its generation in one release store. The queue litmus covered only the 32 bucket rings.

The `e2e_breakchunk` and `e2e_breakcont` controls never failed on any scenario (`dag`, `bigrecord`, `versions`), so their mechanisms had no
test. Each now has a scenario that reaches the broken step, and a control that is not detected counts as a failure.

| Check | What it does | Production | Its negative |
|---|---|---|---|
| `queue(fifo)` | Tokens recirculate through the FIFO ring across CTAs: 2 to 1,024 tokens, 7 and 200 CTAs. Each token must end exactly once (retired, or still in the ring with generation pos + 1), and no pop may double or lose an update | 1,002,349 pops, 0 violations | `fifoticket` (the head ticket by a plain read-modify-write): 47,742 violations. `fifotail` (the tail position by a plain read-modify-write): 2,194 |
| `e2e lateadmit` | One record whose 60,000 successors each depend on one of its first 60,000 tasks. The admitter releases each decoded pass at once, so predecessors finish before their successors' indegrees are added, and add-then-check must release each successor exactly once | pass | `KVM_BREAK_CHUNK_RELEASE` (store instead of add): 58,651 failures |
| `e2e contver` | 16 pipelines: `spin` then `fill(z)` publishes v = 1; a quick `fill(x)` precedes `copy(z -> out)`, which waits for v >= 1. The continuation is offered the copy while its wait is unmet: it must park | pass | `KVM_BREAK_CONT` (the continuation ignores the wait): 192 failures |

Totals (`SUMMARY`): 17 properties, scenarios and probes passed (14 before), 12 mutants and controls detected (8 before), 1 failure.

The failure is `property mp`'s control `relaxed CAS no fence`, which this card does not show. Every run on these cards prints the same line
(an RTX 3090 read the same before this change); the A100 shows the class.

## Limits

- **Read on a 3080 Ti, not an A100.** The FIFO mutants are atomicity faults (two CTAs take one ticket or one tail position), so any GPU
  shows them. The `mp` and `weakpub` lines need an A100.
- **The FIFO ring never waits for a free cell.** It relies on its 4 x live_cap cells exceeding every position in flight: the ready tasks
  (live_cap without version waits, 3 x live_cap with them) plus one ticket per worker. `queuefifo` sizes live_cap to hold this, and so does
  every CLI run (live_cap >= 16,384 against a few hundred workers). Below it, waiting workers share cells, and a consumer that sees a later
  generation stops the run with `QUEUE_OVERFLOW`.

## Reproduce

```
tests/litmus/run_litmus.sh                    # builds litmus_prod, the 10 mutants and the five e2e/CLI binaries into out/, then runs them all
out/litmus_prod queuefifo 5                   # the property alone: 0 violations
out/litmus_fifoticket queuefifo 2             # a mutant alone: must report violations (exit 1)
out/e2e lateadmit 2 && out/e2e contver 2      # production: pass
out/e2e_breakchunk lateadmit 2; out/e2e_breakcont contver 2   # controls: must fail
```

File: `raw/run_litmus.log`, the run's output with the GPU lock's two lines removed.
