# The FIFO ready ring, used only where it fits

GPU: NVIDIA GeForce RTX 3080 Ti (sm_86, 80 SMs), CUDA 12.8, every binary built for `sm_80`. Builds come from the tree's core headers, as
`tests/gpu/build_core.sh` does. Tree: this branch, on the FIFO-ring litmus branch, on the merge of `main` with #13 and #22.

## The defect

The FIFO ring never waits for a free cell. `kvm_publish_cell` stores a slot and its generation in one release store, and each idle worker
watches the cell of its own ticket. That is safe while the ring's 4 x live_cap cells exceed every position in flight: the ready tasks plus
one ticket per worker. The ready tasks are at most 2 x live_cap, because admission caps the runnable window and the parked count at
live_cap each. (The header's own sum, `live_cap + (live_cap + n) + n <= 3 * live_cap`, was mis-added; this change corrects it.)

The executor never checked the bound. With live_cap below the worker count, waiting workers share cells, and one that finds a later
generation stops the run with `QUEUE_OVERFLOW`. The failure is loud, never silent: a cell hands its slot only to the ticket whose generation
it carries.

No measured run reaches it: the CLI's live_cap is at least 16,384 (the panel's MLP rows pass at least 131,072), against 107 workers on an
A100. A hand-set `--live-cap 2` on the occupancy grid reaches it, and `tests/litmus/fuzz.py` draws that value.

## The probe: litmus `queuefifo_cap`

The probe forces live_cap and puts 3 x live_cap tokens in the ring, more than the executor's 2 x live_cap ready tasks. Each point runs
400,000 pops:

| live_cap | 7 CTAs | 200 CTAs |
|---|---|---|
| 2 | 0 violations | **stopped after 42,124 pops: `lost=1 bad_cell=1 err=16(QUEUE_OVERFLOW)`** |
| 4 to 256 | 0 | 0 |

Three earlier runs stopped at the same point and nowhere else:
- two of this mode, after 5,710 and 13,458 pops (`lost=1 bad_cell=1` each; the earlier one printed the code as `err=16`);
- one of the same probe before it became a litmus mode, after 381,707 pops with 3 tokens lost.

Only live_cap 2 with 200 CTAs, 25 waiting CTAs a cell, overran. The guard's bound refuses every point with workers >= live_cap: at 200 CTAs
every live_cap up to 128, at 7 CTAs live_cap 2 and 4. So the guard is conservative.

The probe is a manual mode, outside `all` and `run_litmus.sh`. It finds an overrun only where one is reached at random, so it cannot gate.

## The fix

- **`KvmExec::fifo_fits()`** is `workers() < live_cap` (and `max_tasks <= 2^26`, below). With the 2 x live_cap ready tasks, the positions in
  flight then stay under 3 x live_cap. So a worker that stalls between taking its ticket and reading its cell is overrun only after more
  than live_cap further enqueues. If that happens, it reports `QUEUE_OVERFLOW`.
- **One place decides the ring.** `set_fifo()` runs wherever the FIFO ring is asked for: the config's `fifo_only` and `set_run_facts`. When
  the ring does not fit, the executor keeps the 32 bucket rings, which wait for a free cell, and logs a warning with the reason.
- **The request stays the contract.** `config().fifo_only` keeps the caller's request, and `submit()` rejects a priority or a deadline under
  it on every grid, whichever ring serves the run. `fifo_only()` reports the ring in use, and the CLI's "ready queue" line prints that, where
  it used to print the ring the bundles qualified for.
- **The slot must fit the cell's low 26 bits** (`KVM_FIFO_SHIFT`), so `fifo_fits` also asks `max_tasks <= 2^26`. The CLI already capped
  `--max-tasks` there; a `KvmConfig` did not.
- **Limit, documented:** a FIFO run has 2^38 enqueues before the generation in the cell's upper bits wraps. Past that, waiters never match,
  and the stall timeout reports it.

## Checks

- **`test_exec`** (core build): 8,623 checks, 0 failed. The `dag` suite's new checks:
  - the ring is refused at 8 workers with live_cap 8 (the edge: workers == live_cap) and kept at 7;
  - it is refused at 15 workers with live_cap 8 and kept at 3 with 4,096;
  - `set_run_facts` refuses it at 15 workers and keeps the request in the config;
  - a priority-3 task under a FIFO request is `INVALID` on both a refused ring and a kept one.

  A refused request then ran 640 tasks on the bucket rings, bitwise against the CPU reference. That run is a smoke test of the fallback, not a
  test of the decision: the decision is the `fifo_only()` checks.
- **`KVM_TEST_FIFO=1 test_exec dag`** (every dag run asks for the FIFO ring): 602 checks, 0 failed.
- **litmus `queuefifo`** (the FIFO ring's property): 1,002,367 pops, 0 violations. **litmus `queue`** (the bucket rings' property):
  1,000,063 pops, 0 violations.
- **The litmus detail lines** of both queue tests now print a device error by name (`err=16(QUEUE_OVERFLOW)`). An error counts once in the
  violation sum, instead of adding its code.

## Reproduce

```
tests/gpu/build_core.sh test_exec_core && ./test_exec_core dag && KVM_TEST_FIFO=1 ./test_exec_core dag
nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude tests/litmus/litmus.cu -o litmus_prod
./litmus_prod queue 5 && ./litmus_prod queuefifo 5    # the properties: 0 violations
./litmus_prod queuefifo_cap 20                        # the probe: violations only where workers >= live_cap (exit 1 when it finds any)
```

Files: `raw/suites-summary.txt` and `raw/dag-on-fifo-ring-summary.txt` (each suite's verdict, the new checks' line and their warnings), and
`raw/queue.log`, `raw/queuefifo.log` and `raw/queuefifo_cap.log` (whole).
