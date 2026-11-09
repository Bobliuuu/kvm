# REVIEW: adversarial review of the KVM persistent-kernel executor

Reviewed `include/kvm_exec.cuh` + `kvm_exec.cu` at 083513e (this worktree's base: no slot recycling). Key results were
re-run against `feat/executor` @ 5b869aa (slot recycling, free ring, windows; section "Re-run on 5b869aa").
Hardware: one A100-SXM4-40GB (108 SMs, sm_80), CUDA 12.x. Everything here is reproducible from `tests/litmus/` (see the end).
Line numbers are for 083513e.

## Verdict

* The lock-free core (ready queue, park/publish, indegree wakeups, version visibility, admission ring) **held up**: 0
  violations in tens of millions of litmus rounds, and every harness has a mutant that fails, so the zeros mean something.
* The defects are **outside the memory model**: policy and robustness. Timeout semantics, a deadlock class the admission
  accounting cannot see, host-side integer overflow and null-dereference crashes on malformed input, an unbounded long task.
* Nothing found corrupts data through the normal CLI path. Two items (C3, C5) can crash the host CLI or stall the GPU for
  minutes from a corrupted input file.

## Findings (ranked)

| ID | Sev | Status | Where | One line |
|----|-----|--------|-------|----------|
| C1 | High | CONFIRMED | cuh:468, 568, 630; cu:214 | `timeout` is a kernel-LIFETIME cap, not a stall detector: a healthy, progressing executor is killed at `timeout_s` |
| C2 | Med-High | CONFIRMED | cuh:567 | Parked (version-waiting) tasks count against `live_cap`; a valid cross-bundle wait deadlocks admission until the timeout |
| C3 | Med | CONFIRMED | cu:82-83 (host), cuh:588 (device) | `off + cnt` int overflow passes the range check: host segfault on corrupt `tasks.bin`; device illegal address with host validation bypassed |
| C4 | Med | CONFIRMED | cu:404-438, 504, 542, 552, 555 | CLI segfaults on malformed `metadata.json` / `golden.json` (null derefs, OOB `trace[]`, unbounded JSON recursion) |
| C5 | Med | CONFIRMED | cuh:337-340; cuh:466 | No per-task watchdog and `spin` arg unvalidated: one task holds the GPU far past `timeout`, up to 35 min |
| C6 | Low-Med | CONFIRMED | cu:213-214, 461-467 | `atoll` CLI parsing: negative `--aging-us` inverts priorities or stalls; `--timeout-s 0` fails at once; `-1` disables the timeout |
| C7 | Low | CONFIRMED | cu:274-318 | `submit()` returns OK after the device died (unless the ring is full): accepted bundles silently never run |
| C8 | Low | CONFIRMED | cu:99-106 | Unsatisfiable / cyclic version waits are not rejected by host validation, only by the (slow) device timeout |
| S1 | Low | SUSPECTED | cu:176-177 | `8 * cfg_.max_tasks` int overflow (wraps at 2^29+1; OOM stops it first on 40 GB) |
| S2 | Low-Med | SUSPECTED | cuh:671-680, cu:184-185 | Grid is sized for the whole GPU and block 0 is the admitter: no co-residency guarantee if another process holds SMs |
| S3 | Low | SUSPECTED (5b869aa) | `kvm_admit_record` free-ring pop | `while (seq != fp+1) __nanosleep` has no error/timeout exit; hangs block 0 if a freeing worker bailed out |
| S4 | Info | SUSPECTED | cuh:384-389 | Release store of `cell->seq` is not empirically necessary on A100 (relaxed mutant never fails); correctness there rests on the PTX model alone |

## C1. `timeout` is a lifetime cap (High, CONFIRMED)

`kvm_should_exit` (cuh:468), the admitter idle loop (cuh:630) and the backpressure loop (cuh:568) compare `kvm_now() - t0`,
where `t0` is the block's launch time. Header text says "deadlock or stalled admission".

Scenario: a serving executor that idles 3.5 s between submits with `timeout_s=2`, or a steady 40-task stream that always
makes progress, dies with `KVM_ERR_TIMEOUT`. Default 60 s means no persistent service can outlive one minute.

Evidence (`e2e probe_timeout`, `probe_timeout_busy`):
```
probe_timeout:      healthy executor idle 3.5s with timeout_s=2 ... final err=TIMEOUT done=1/1
probe_timeout_busy: a 100ms task every 100ms (always progressing), timeout_s=2: submitted 40/40 over 4.0s, err=TIMEOUT done=20/20
```
Fix: time a stall, not the lifetime. Pending work and no counter movement for `timeout_ns`; idle never times out.
```diff
--- a/include/kvm_exec.cuh
+++ b/include/kvm_exec.cuh
+struct KvmWatch { unsigned long long since; int done, admitted; };
+// Stall = work is pending (admitted > done) and neither counter moved for timeout_ns. Idle waiting for the host is not a stall.
+__device__ __forceinline__ bool kvm_stalled(const KvmDevState& S, KvmWatch& w) {
+  const int d = ld_acq(&S.cnt->done), a = ld_acq(&S.cnt->admitted); const unsigned long long n = kvm_now();
+  if (d != w.done || a != w.admitted || a == d) { w.done = d; w.admitted = a; w.since = n; return false; }
+  return n - w.since > S.timeout_ns;
+}
@@ kvm_should_exit(S, t0)   ->   kvm_should_exit(S, KvmWatch& w)
-  if (kvm_now() - t0 > S.timeout_ns) {
+  if (kvm_stalled(S, w)) {
@@ kvm_admitter / kvm_admit_record: same substitution at cuh:568 and cuh:630 (thread-0 local KvmWatch, initialised to {kvm_now(),0,0}).
```
Caveat to document: `timeout` must then exceed the longest single task (see C5).

## C2. Parked tasks pin `live_cap`: cross-bundle wait deadlock (Med-High, CONFIRMED)

`admitted - done + nt <= live_cap` (cuh:567) counts a task parked on a version like a running one, but a parked task can only
be released by a publisher that may sit in a bundle the admitter cannot yet admit.

Scenario: bundle 1 = `live_cap` tasks all `wait v0>=1`; bundle 2 = the one task that publishes `v0=1`. All of bundle 1 parks,
`admitted - done == live_cap`, bundle 2 never fits. Evidence (`e2e probe_deadlock`, `live_cap=8`):
```
8 waiters then the publisher bundle: err=TIMEOUT done=0/8 after 3.0s
```
Also reproduced on 5b869aa (where it additionally pins the window's successor words, so `succ_cap`/`win_cap` fill too).
Valid per ABI ("bundles share state only through version variables"), so this is a liveness bug, not misuse.
Minimal patch = fail fast and name the cause instead of a 60 s timeout; a real fix needs a separate parked budget.
```diff
@@ struct KvmCounters   +  alignas(128) int parked;
@@ kvm_release_task, before the push   +  atomicAdd(&S.cnt->parked, 1);
@@ every winning atom_cas(S.state + n, 1, 2) == 1   +  atomicAdd(&S.cnt->parked, -1);
@@ kvm_admit_record backpressure loop (inside the while):
+        if (ld_acq(&c->admitted) - ld_acq(&c->done) == ld_acq(&c->parked)) {      // every live task is parked: only admission can unblock
+          if (!stall_since) stall_since = kvm_now();
+          else if (kvm_now() - stall_since > 1000000ull) { kvm_fail(S, KVM_ERR_PARK_STALL, -1, c->parked, nt); A.action = KVM_ADM_EXIT; break; }
+        } else stall_since = 0;   // 1 ms grace for a waiter whose own recheck is still in flight
```
Real fix (design): size the ready rings `live_cap + max_parked`, count parked tasks against `max_parked` instead of `live_cap`.
Workaround today: host submits publishers before waiters, or `live_cap` > max simultaneously parked.

## C3. Range check overflows (Med, CONFIRMED)

Host `validate` (cu:82-83): `t.args_off + t.args_words > size` and `t.succ_off + t.succ_cnt > size` are `int + int`. With
`args_off = INT_MAX` (+3 words) the sum wraps negative, passes, and the next line indexes `args[INT_MAX]`: segfault.
Same shape on the device (cuh:588 `so + sc > ne`, `ao + aw > na`) when host validation is bypassed (`submit(b, ms, false)`,
a corrupt ring record): illegal memory access, i.e. the GPU kernel faults.

Evidence: `fuzz.py --targeted`: T01 `succ_off=INT_MAX,succ_cnt=1` SIGSEGV; T03 `args_off=INT_MAX` SIGSEGV; T04 `args_off=INT_MAX-1` SIGSEGV
(T02 `succ_cnt=INT_MAX` and T05 `succ_off=-1` are correctly rejected). `e2e devfuzz` boundary sweep over 4 tasks x 4 fields x 3 values:
16/48 cases end in `cudaStreamSynchronize -> an illegal memory access`; 1 more in 299 random cases.
On 5b869aa the successor ring is masked, so `succ_off/succ_cnt` overflow no longer faults but silently follows other windows' words
(or loops 2^31 times: `succ_cnt=INT_MAX` hangs, 5 cases); `args_off` still faults (8 cases).
```diff
--- a/kvm_exec.cu
-      if (t.args_off < 0 || t.args_off + t.args_words > (int)args.size()) ...
+      if (t.args_off < 0 || (long long)t.args_off + t.args_words > (long long)args.size()) ...
-      if (t.succ_off < 0 || t.succ_cnt < 0 || t.succ_off + t.succ_cnt > (int)succ.size()) ...
+      if (t.succ_off < 0 || t.succ_cnt < 0 || (long long)t.succ_off + t.succ_cnt > (long long)succ.size()) ...
--- a/include/kvm_exec.cuh
-    else if (so < 0 || sc < 0 || so + sc > ne || ao < 0 || aw < 0 || ao + aw > na || ...
+    else if (so < 0 || sc < 0 || (long long)so + sc > ne || ao < 0 || aw < 0 || (long long)ao + aw > na || ...
```

## C4. CLI crashes on malformed metadata/golden (Med, CONFIRMED)

Nothing validates JSON shape before dereferencing. `fuzz.py` random: 8/150, 13/300, 17/300 cases SIGSEGV; deterministic probes:

| Probe | Site | Result |
|-------|------|--------|
| T06 `reads[0]` without `policy` (`--golden`) | cu:555 `pol->s` | SIGSEGV |
| T07 `reads[0]` without `name` | cu:555 `jr.get("name")->s` | SIGSEGV |
| T08 `reads[0].task = -1` | cu:552 `r.trace[slot]` | OOB read (reads garbage; reported MISMATCH) |
| T09 `reads[0].task = 1000000` | cu:552 | SIGSEGV |
| T10 `variables[0]` without `name` | cu:542 | SIGSEGV |
| T11 buffer `init:{kind:"values"}` without `values` | cu:504 `get("values")->a` | SIGSEGV |
| T12 `metadata.json` = 200000 x `[` | `JsonParser::value` recursion | stack overflow SIGSEGV |

All fail before any GPU work, so the GPU is untouched, but the contract says "reject cleanly".
```diff
--- a/kvm_exec.cu
@@ JsonParser
+  int depth = 0;
   Json value() {
-    ws(); if (i >= s.size()) bad("eof");
+    ws(); if (i >= s.size()) bad("eof");
+    struct D { int& d; D(int& x) : d(x) { ++d; } ~D() { --d; } } guard(depth);
+    if (depth > 64) bad("nesting too deep");
@@ buffers init
-            else if (kind && kind->s == "values") { for (auto& v : in->get("values")->a) init.push_back((float)v.n); }
+            else if (kind && kind->s == "values" && in->get("values")) { for (auto& v : in->get("values")->a) init.push_back((float)v.n); }
@@ golden
-      ... var_id[jv.get("name")->s] = ...
+      ... if (const Json* nm = jv.get("name")) var_id[nm->s] = ...
-        const KvmTrace& t = r.trace[slot]; ++nreads;
+        if (slot < 0 || slot >= (int)r.trace.size() || !jr.get("name") || !pol) { KLOG(0, "golden: malformed read entry"); ++golden_bad; continue; }
+        const KvmTrace& t = r.trace[slot]; ++nreads;
```

## C5. No per-task watchdog; `spin` unbounded (Med, CONFIRMED)

The timeout is only evaluated while a block is idle (`kvm_should_exit`). A running task cannot be interrupted, and `spin`
(cuh:337) accepts any `int` microseconds (INT_MAX us = 35.8 min).

Evidence: `e2e probe_spin`: one 9 s spin task, `timeout_s=2` -> `finish()` returns after 9.0 s. `fuzz.py` random seed 3:
`args[16]=2000000000` flips a spin argument -> HANG (killed by the driver at 40 s; kernel would run 33 min). T13 (`--timeout-s 3`, 100 s spin) the same.
The same holds for any op whose runtime scales with an unvalidated arg; ops that bound `n` by buffer size are fine.
```diff
 KVM_OP(spin) {
   int us = ARG_I32(0);
+  if (us < 0 || us > 10000000) { c.fail(KVM_ERR_BOUNDS, us); return; }   // 10 s
```
and document that `timeout` must exceed the longest task (the fix in C1 has the same limit).

## C6. CLI numeric parsing (Low-Med, CONFIRMED)

`atoll` accepts garbage as 0 and truncates to `int`; negative values reach the device.

| Input | Effect (measured) |
|-------|-------------------|
| `--aging-us -5` (cu:213 `aging_ns = -5000`) | priorities INVERTED: run order of priorities 0..11 on one worker is `0 1 2 ... 11` (correct is `11 ... 0`) |
| `--aging-us -2000000` | `done 0/13`, TIMEOUT after 3 s (scores < 0 mean "empty", `kvm_pop` never pops) |
| `--timeout-s 0` (T16) | TIMEOUT at 0.02 ms on an empty run |
| `--timeout-s -1` (cu:467 wraps to ~2^64 us) | timeout disabled: unsatisfiable wait hangs until the driver kills it (T15) |
| `--max-tasks 4294967297` | silently becomes 1 (by inspection of the `(int)` cast, not run) |
```diff
-    auto val = [&]() -> long long { if (i + 1 >= argc) { usage(); exit(2); } return atoll(argv[++i]); };
+    auto val = [&](long long lo, long long hi) -> long long {
+      if (i + 1 >= argc) { usage(); exit(2); }
+      char* e; errno = 0; long long v = strtoll(argv[++i], &e, 10);
+      if (e == argv[i] || *e || errno || v < lo || v > hi) { fprintf(stderr, "%s: integer in [%lld,%lld] expected, got '%s'\n", a.c_str(), lo, hi, argv[i]); exit(2); }
+      return v; };
-    if (a == "--aging-us") cfg.aging_us = (int)val();
+    if (a == "--aging-us") cfg.aging_us = (int)val(0, 1000000000);
-    else if (a == "--timeout-s") cfg.timeout_us = (unsigned long long)val() * 1000000ull;
+    else if (a == "--timeout-s") cfg.timeout_us = (unsigned long long)val(1, 86400) * 1000000ull;
   (and --blocks val(0,1<<20), --live-cap/--max-tasks/--ring-words val(1,1<<30))
```
Also validate `aging_us >= 0` in the `KvmExec` constructor for API users.

## C7. `submit()` after device death (Low, CONFIRMED)

The `herr_->code` check (cu:297) sits only on the ring-full path. After a device error with ring space free, `submit` returns OK.
Evidence: `probe_timeout_busy` submitted 40/40 OK, 20 ran. Patch: check once at the top of `submit` after the `closed_flag_` test:
```diff
+    if (herr_->code) { last_error_ = "device error"; return KvmSubmit::DEVICE_ERROR; }
```

## C8. Version-wait cycles are not rejected host-side (Low, CONFIRMED)

`validate` runs Kahn on successor edges only (cu:99-106). `wait_ver = INT_MAX` (T14) or two tasks waiting on each other's
publication is accepted; the device rejects it only after `timeout` (3.3 s here, 60 s by default), reported as TIMEOUT, not the cause.
The compiler already knows base versions and publish counts; a per-variable check `wait_ver <= base + publishes(var)` plus a cycle check
over the (edge + publish->wait) graph in `validate` makes this a clean, instant INVALID.

## Suspected

* **S1** `8 * cfg_.max_tasks` (cu:176-177) overflows `int` for `max_tasks >= 2^28`; at 2^29+1 it wraps to 8 (pools of 8 words). Not reached: `cudaMalloc` OOMs first on 40 GB (T19 rejected cleanly).
* **S2** `grid = SMs * occupancy` assumes the whole GPU. Blocks are not guaranteed co-resident with another kernel/process holding SMs; if block 0 (admitter) were scheduled late, workers spin until their timeout. Not reproduced (block 0 launched first every time, 540 blocks).
* **S3** (5b869aa) the admitter's `while (ld_acq64(&cell->seq) != fp + 1) __nanosleep(32);` has no error/timeout exit. If a worker took a free-ring ticket and then failed in `kvm_free_slot` (QUEUE_OVERFLOW path returns without publishing), block 0 spins forever and the kernel never ends. Only reachable after another failure.
* **S4** `qrelax` mutant (plain store instead of `st.release` for `cell->seq`) never failed in 1M pops: A100 does not expose the missing release, so that edge is untested empirically. The model requires it; keep it.
* The host NT-store claim ("seq_cst fence also orders non-temporal stores") is correct on x86 (MFENCE orders WC/NT stores) but the 6.8 MB record test may not have crossed glibc's NT-copy threshold, so the NT path itself was not shown to run. Arm hosts untested.

## Memory-model audit (task 1)

| # | Header claim | PTX justification | Attack | Result |
|---|--------------|-------------------|--------|--------|
| 1 | Publisher: all threads store; `__syncthreads`; thread 0 `fence.sc.gpu`; release CAS. Other threads' stores precede the CAS | `bar.sync` orders all participating threads' earlier operations before later ones (CTA scope). Causality order is transitive, so W_t -> bar -> F_0 -> release CAS(gpu) chains W_t before the CAS. The release is cumulative over what thread 0 has observed. Standard idiom (cooperative-groups grid sync). Sound; the `fence.sc` here is redundant for visibility (needed only for Dekker) | MP litmus: 256-thread writer CTA, plain stores, publish; reader CTA primes L1 with old data, spins, reads. 864 CTAs. **Mutant** relaxed CAS and no fence | PROD: 0 stale in 3.0 M rounds (3.07 G words checked). Mutant: 244 stale rounds / 3.0 M (control detected). `nofence` build (all `__threadfence` removed, CAS still `acq_rel`): 0 / 0.4 M |
| 2 | Reader: thread 0 `ld.acquire.gpu`, then `__syncthreads`, other threads read | acquire -> bar.sync -> other threads' loads is the mirror chain. Stale L1 lines are the hazard that makes the acquire necessary | **Mutant** reader polls with `ld.relaxed` | PROD 0 / 3.0 M; mutant 55,737 stale rounds, 42.5 M stale words (control detected: L1 staleness is real here) |
| 3 | Publish-before-data | n/a (negative control) | thread 0 publishes before the other threads store | 2,906,465 / 3.0 M stale (the harness fails when it should) |
| 4 | Dekker: waiter `push; fence.sc; load version`, publisher `store version; fence.sc; exchange` | `atom.acq_rel` RMWs on different addresses are NOT sequentially consistent in PTX; the two `fence.sc.gpu` totally order and forbid both-miss. Required by the model | SB shape with relaxed ops (no fence) vs fence vs RMW-only; then the REAL `kvm_release_task`/`kvm_publish` with random skew. Mutants: waiter never rechecks; publisher exchanges before storing; both fences removed | SB: relaxed/no-fence 19,222 both-miss / 6 M (harness sensitive); every fenced or RMW form 0 / 6 M. Real code: 0 lost, 0 double in 6 M (1 waiter) + 3 M (3 waiters, re-park path, 9 M waiters). Mutants caught: `norecheck` 23,020 lost / 0.5 M; `exchbefore` 11,800; **`nodekkerfence` and `nofence`: 0**, so on A100 the Dekker fences are not observably needed (acq_rel RMWs behave SC); SASS has `MEMBAR.SC.GPU`, so `__threadfence()` is the `fence.sc` the argument needs |
| 5 | Indegree `acq_rel` RMW + last-decrementer enqueue; cell release/acquire | release sequence over RMWs; last decrementer acquires every predecessor's release | 12 random layered DAG configs (24-node bundles, concurrent producers, ring 576/1024 words, live_cap 32/64, aging on/off): exact integer dataflow per node | 172,800 tasks, 144,238 edges: 0 wrong values, 0 exec_count != 1, 0 edges with `start < pred.end` |
| 6 | Ready queue: Vyukov cells, 64-bit tickets, recycle `seq = h + live_cap`; ABA after wrap | tickets never wrap (2^64); head CAS is monotonic | Real `kvm_enqueue`/`kvm_pop`, `cap` 2..1024 tokens recirculating (= the in-flight bound), 7 or 200 warps, random priorities, per-token plain counter + inQueue flag + drain | 2.0 M pops, up to 100,000 wraps per bucket: 0 double pops, 0 lost updates, 0 lost/dup tokens. **Mutant** plain-store head claim: fails in < 400 pops (control detected). Mutant relaxed `seq` store: NOT detected (S4) |
| 7 | Host doorbell: record via memcpy (NT stores possible), `atomic_thread_fence(seq_cst)`, release store; device `ld.acquire.sys` then relaxed reads by all threads after `bar.sync` | x86 TSO + MFENCE orders NT stores; mapped memory is write-back (no WC flag); device acquire at `.sys` scope. `closed` read before `doorbell` so FINAL implies the doorbell is final | 50,000 bundles through rings of 64/72/96/256/4096 words (pad record on nearly every wrap), 3 producer threads; one 6.8 MB record; 200 `close()` vs 4-thread submit races | 0 wrong/missing; 283,993 accepted bundles all ran; no mismatch between `admitted` and accepted submits |
| 8 | Version-ordered pipelines (the whole protocol, 2 vars/pipeline, readers admitted first so they park) | combination of 1-4 | 24 pipelines x 120 versions x 2048 floats, readers and writers ordered ONLY by versions, 9 runs | 0 reads != the exact version. **Negative control** (`-DKVM_BREAK_FENCE`): 1,184 stale reads |

## What I could NOT break, and how hard

| Property | Attempts | Zero-failure evidence |
|----------|----------|-----------------------|
| No lost wakeup (park vs publish), single waiter | 6.0 M rounds, random skew, 864 CTAs | 0 |
| No lost / double wakeup, 3 waiters + re-park (need 1, 2, 2; two publishers) | 3.0 M rounds (9.0 M waiters) | 0 |
| Version visibility: new version implies all of that version's weights | 3.0 M rounds x 1024 words (isolated) + 9 e2e runs x 24 x 120 versions | 0 |
| Queue exactly-once, wraparound, ABA | 2.0 M pops, 10 configs | 0 |
| Double execution / early execution | 172,800 DAG tasks, 144 k edges | 0 (`exec_count`, trace order, values) |
| Indegree decrement vs root decision | same DAG runs + `devfuzz` indegree mutations (rejected or TIMEOUT, never a double run) | 0 |
| Admission ring wrap, pad records, backpressure, record straddling | 50,000 bundles, 5 ring sizes down to the 64-word minimum, 3 producers | 0 |
| `close()` vs `submit()` races | 200 races, 284 k bundles | 0 |
| Device error paths leave no block spinning | `devfuzz`: 299 random + 48 boundary cases: every non-crash ends in a clean error or OK; sanity run after each fuzz batch | only the C3 faults and C5 long task |
| Malformed bundle through the CLI (valid JSON) | 750 random + 22 targeted cases, GPU sanity after each batch | no GPU hang from a malformed bundle other than C5; only C3/C4 host crashes |
| compute-sanitizer | memcheck, initcheck, synccheck, racecheck on two compiled bundles (`--blocks 5`) | 0 errors, 0 hazards (smoke only: tiny grid) |

Controls that the harness can fail (all detected): SB without fence (19,222), publish-before-stores (2.9 M), reader without acquire (55,737),
relaxed CAS (244), waiter without recheck (23,020), publisher exchange-first (11,800), plain-store head claim (< 400 pops),
`KVM_BREAK_FENCE` e2e (1,184).

## Re-run on 5b869aa (slot recycling)

* Litmus (park/publish/scan, MP, queue) against the new header: dekker 1.0 M + 0.5 M rounds, MP 0.5 M, queue 1.0 M pops: 0 violations; `KVM_BREAK_PARK_RECHECK` mutant: 12,834 lost wakeups / 0.3 M (detected).
* E2E with a 128-slot pool (constant recycling), 115,200 DAG tasks, versions (34,560 tasks), ring (35,000 bundles), 6.8 MB record, 200 close races: all PASS; `KVM_BREAK_FENCE` control: 829 stale reads.
* `fuzz.py --targeted` against the 5b869aa CLI: identical outcome to 083513e (same 11 findings: T01 T03 T04 SIGSEGV from C3, T06 T07 T09 T10 T11 T12 SIGSEGV from C4, T13 and T15 HANG from C5/C6). `devfuzz` 48 boundary + 200 random cases: `args_off` overflow faults (8 cases + the random one), `succ_cnt` overflow hangs (5 cases, killed at 30 s).
* C1, C2, C5, C6 reproduce unchanged (`probe_timeout`, `probe_timeout_busy`, `probe_deadlock`, `probe_spin`, `probe_aging`).
* C3 changes shape (see C3). New concern S3. One new design observation: a task parked on a version pins its window's successor words and, because regions are reclaimed in order, every later window's words: a deadlock class larger than C2 once `succ_cap` or `win_cap` is small.
* The late `d->wait_ver` read after the push in `kvm_release_task` can observe a recycled slot (the task may already have run); the effect is only an extra or skipped `kvm_scan_parked`, and a skipped scan cannot lose a wakeup because the node was already detached. Benign.

## Not covered

Other GPUs/architectures (only A100 sm_80), multi-GPU, the gpt/game op packs, NT-store host path, Arm hosts, proofs beyond the
PTX argument above. Litmus cannot prove absence of reorderings the A100 happens not to exhibit (rows 1, 4, 6): those rest on the model.

## Reproduce

```
KVM_HOST=user@host tests/litmus/run_remote.sh build      # rsync + nvcc: litmus binaries + mutants, e2e, kvm_exec
KVM_HOST=... tests/litmus/run_remote.sh litmus           # prod litmus (LK/LL/LQ env scale it)
KVM_HOST=... tests/litmus/run_remote.sh mutants          # every mutant: each must report VIOLATIONS
KVM_HOST=... tests/litmus/run_remote.sh e2e | probes | devfuzz | sanitize
KVM_HOST=... FC=300 FS=2 tests/litmus/run_remote.sh fuzz        # random corpus; FZ=--targeted for the T01..T22 probes
KVM_HOST=... tests/litmus/run_remote.sh buildnew         # same against a branch exported to ~/kvm-review-new
```
Files: `tests/litmus/litmus.cu` (real device functions, SB/Dekker/MP/queue + controls), `build.sh` (sed-built header mutants),
`e2e.cu` (host-driven scenarios and probes), `fuzz.py` (corpus generator from `runtime/compile_workload.py` bundles + driver).
