# Root causes of three open flakes (recycle admission, T=1 attention backward, `versions`)

GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, CUDA 12.8. Shared box (other jobs ran during every run: load average about 4-5).
Build: `tests/gpu/build_core.sh` (suites) and `nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo` (test_ops). Tip tested: `fix/flake-rootcause` (this change on
top of `main` 1c2111b). Repetition harness: one `flock ~/.gpu.lock` hold per batch, the pre-fix and fixed binaries alternated run by run, so both
see the same box load.

## Verdicts

| Flake | Root cause | Fix (commit) | Evidence |
|---|---|---|---|
| `recycle` long-running admission | `window_done` tested `win_done[slot] == wid+1`. The table is a ring: once a newer window reuses the slot the mark is `wid+win_cap+1`, so a finished window read as "not done" for ever. A producer that polled late (here: producer 0 was at bundle 363 while producer 1 had already submitted all 50,000, so its polled window had been overwritten) spun to its 60 s timeout and returned silently. | 15994b4 (already on main): wrap-safe signed `>=`. This change adds the diagnostics. | 25 alternated runs: pre-fix 10 failed (40%), fixed 0 failed. Revert-the-fix control: the wrap regression test (a2) fails 3 of 3 (`window_done false for 256 of 320 finished windows`); fixed binary passes 3 of 3. |
| `test_ops` T=1 attention backward | Not a race. dq, dk are exactly 0 mathematically (p=1, o=v, dP-delta cancels); the kernel returns rounding noise, which is **bitwise deterministic per data set** (3 runs of a 2000-seed sweep: identical output hash). So a "flake" is a new data set or new kernel rounding crossing a fixed number. The old absolute bound 3e-5 is crossed on A100: max noise over 2000 seeds is 4.11e-5 (2.77e-5 over the 64 seeds in the suite: 8% margin). | This change: per-element bound derived from the data. | `attention/`: 2000 seeds x hd 16/32/64/128 x 3 runs, worst noise/bound 0.151 (6.6x margin), identical every run. |
| `versions` (scalar `at_least` reader) | Not a kernel bug. The publisher stores the new weights, fences, then publishes the version, so an `at_least` reader (no upper gate) can read `w[ver1+1]` while its sampled `ver1` is still `ver1`. The scalar checker allowed `[ver0, ver1]`, the vector one `[ver0, ver1+1]`. Timing-dependent, architecture-independent. | c4cc899 / f595945 (already on main): shared `at_least_hi()`. | 600 alternated runs on A100: pre-fix tree 15 failed (2.5%, seeds 5021 5058 5211 5262 5398, always "1 scalar readers violated their snapshot"), fixed 0 failed. (~10% was the A10G figure; A10G box not used.) |

## What changed in this commit

- `kvm_exec.cu`: `window_mark()`, `window_why_not_done()`; the `submit_stream` timeout error now carries the raw mark.
  A mark above `wid+1` (negative "windows behind") means the table wrapped over `wid`; below means the window really is unfinished.
- `tests/gpu/test_exec.cu` (`recycle`): producers record their first failure (producer, bundle, set, raw mark) into the verdict line; before, they
  returned silently and the only symptom was `submitted 50277 of 100000`.
- `tests/gpu_ops/test_ops.cu`: T=1 dq/dk checked elementwise against `eps * sum_d|dO_d v_d| * scale * |other operand|`, `eps=(hd+8) 2^-22`
  (hd terms of a 3xTF32 product, the fp32 delta dot, o's 2^-23 deviation from v), summed over the heads of a kv group for dk. The verdict prints the
  worst noise/bound ratio; a real bug (dropped term, wrong scale) is O(1) against a bound of ~1e-5, i.e. 1e5 times over. `KVM_T1_SEEDS=N` deepens the sweep.

## Reproduce

```
# recycle, alternated pre-fix (1e58c69) vs fixed, 25 rounds:   ./test_exec_core recycle
# revert control: main with window_done changed back to `== wid + 1`
# versions, alternated pre-fix (f595945^) vs fixed:             ./test_exec_core versions   (500 + 100 rounds)
# attention:  TEST_ONLY=attention KVM_T1_SEEDS=2000 ./test_ops
```
Files: `recycle/` (summary + one failing, one passing pre-fix log, one fixed log; `[D]` ring lines stripped), `recycle-negative/`,
`attention/`, `versions/summary-100.txt`, `versions/summary-500.txt` (columns: tree, run, exit code, FAIL lines).
Limits: the exact cause of producer 0's lag (unfair `mu_` acquisition vs thread descheduling on the loaded box) is not instrumented; the fix does not
depend on it because `window_done` is now exact for any lag. The 40% pre-fix rate is for this box under its usual load, not a general figure.
