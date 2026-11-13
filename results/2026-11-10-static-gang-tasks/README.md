# Gang tasks on the static path: a fused op's blocks scheduled to run at once

**Verdict.** The static planner now places a *gang*, the tasks of one op that meet at barriers inside it, on distinct workers at one point of
every worker's program, so a fused step can span several blocks without deadlock. Bundles without gangs plan exactly as before. The gang's barrier
(`include/kvm_gang.cuh`) needs no reset between uses or runs, and a member that fails ends its gang instead of hanging it. No op in this change
uses gangs yet; the split MLP step (`fmlp_split`) and the fused GPT step (D2) are their first users.

## What changed

* **Interface** (`include/kvm_static.h`): a gang op names three `i32` arguments, `gang`, `member` and `members`. `kvm_static_gang_args` reads
  their positions from the op's registry signature. No ABI change.
* **Planner** (`kvm_static_plan`):
  - the DAG is contracted to units, where a unit is a task or a gang's members together, and the units are ranked and placed;
  - a gang takes the `members` workers that free up first, and each member goes to the chosen worker where its data arrives first;
  - all members start when the last can, and finish together;
  - refused: a member depending on another member (directly or through other tasks), a gang wider than the workers, and a member missing, given
    twice, or disagreeing on the count.
* **Why it cannot deadlock:** every worker's program is appended in rank order, and a gang's members are appended at one point of that order.
  So every record before that point, on any worker, waits only for records before it, and every member reaches the op's barrier.
* **Executor** (`kvm_exec.cu`): the registry views carry the gang arguments. A bundle with gang ops takes the static path, and the dynamic path
  refuses one with its reason, since its ready queue cannot promise that a gang's members run at once.
* **Barrier** (`include/kvm_gang.cuh`):
  - sense-reversing on two words: a member reads the generation, then adds itself to the count, and the last arrival zeroes the count and
    publishes the next generation;
  - the count is zero between crossings, so the words need no reset between runs;
  - a waiting member reads the run's error word every 4,096 polls and gives up once it is set.
* **CPU tests** (`tests/static/registry.h`): `test_static_plan` and `test_fuse_pass` take the operator registry the executor compiles
  (`include/kvm_ops_list.inc`, expanded at compile time). They read the `.def` files by a list of names relative to the working directory
  before, so a run outside the repository root loaded no operator and still passed.

## Tests

| test | where | result | log |
|---|---|---|---|
| `tests/static/test_static_gang.cpp`: 550 synthetic plans in 4 wirings (one a stress case: long tasks holding workers, members of unequal rank). The emitted records are replayed with the barrier's semantics: distinct workers, no deadlock, every edge honoured | CPU (g++) | all pass | `test_static_gang.log` |
| the same test against two mutant planners (one worker for every member; gangs planned task by task), built by editing one line of `kvm_static.h` each | CPU | 881 and 441 failures of the independent checks |  |
| 6 refusal cases (each malformed gang above) | CPU | all refused with their reason | `test_static_gang.log` |
| `tests/static/test_static_plan.cpp` on mlp72, mlp576 and mlp72 x 8 runs, at 8 worker counts, against the planner of `339c6b9` (this branch's base) | CPU | the same output, line for line | `planner_unchanged.log` |
| `tests/static/test_gang_device.cu`: gangs of 1, 2, 3, 5, 8, 13 and 48 blocks side by side, 2,000 rounds of slot write, barrier, read every slot, barrier; twice without re-zeroing | RTX 3080 Ti | 0 wrong reads; counts back at 0; generation +2 a round | `test_gang_device_3080ti.log` |
| the same under compute-sanitizer memcheck, racecheck, synccheck (50 rounds) | RTX 3080 Ti | 0 errors, 0 hazards | `test_gang_device_3080ti.log` |
| a barrier that does not wait (mutant) | RTX 3080 Ti | fails (311M wrong reads) | `test_gang_device_3080ti.log` |
| the abort path: the widest gang's first member quits at round 7 (with the clean launches, and under the three sanitizers) | RTX 3080 Ti | the launch ends; its 47 other members give up; 0 errors, 0 hazards | `test_gang_device_abort_3080ti.log` |
