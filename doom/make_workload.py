#!/usr/bin/env python3
"""Generate a workload JSON for runtime/compile_workload.py: real Doom frames as scheduler tasks, optionally beside a compute load.

  make_workload.py --frames 700 [--chains 128 --rounds 24] [--doom-priority 10] --out wl.json

graph "doom": per frame f   tick_f (doom_tick) -> present_f_0..3 (doom_present, 50 rows each) and status_f (doom_status, hash + state);
              tick_f+1 waits for all of them (tick_f+1 overwrites the screen they read). The 35 Hz pacing is deadline_us metadata only.
graph "compute": --chains independent chains of (matmul 256^3 then spin) x --rounds. Every chain computes the same A@B, so the result is
                 checkable; the spins keep every block busy for the whole run (chains > worker blocks, so the ready queue is never empty).
"""
import argparse
import json
import random

FB_WORDS = 16256          # KVM_DOOM_FB_WORDS: 64000 palette indices + 256 palette words
STATE_REC = 16            # KVM_DOOM_ST_REC
SLICES = 4
N = 256                   # matmul size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=700)
    ap.add_argument("--instances", type=int, default=1, help="independent Doom engines (1..8), each its own chain, buffers and state; needs a build with DOOM_INSTANCES>=N")
    ap.add_argument("--chains", type=int, default=0, help="compute chains (0: no competing load)")
    ap.add_argument("--rounds", type=int, default=40, help="rounds per chain; each round is one matmul then one spin")
    ap.add_argument("--spin-us", type=int, default=200000, help="a spin task holds a block this long (sustains the load for the whole run)")
    ap.add_argument("--burst", action="store_true", help="bursty load: each spin lasts 5, 40 or 200 ms (seeded), chains start staggered")
    ap.add_argument("--doom-priority", type=int, default=10)
    ap.add_argument("--compute-priority", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    N = a.instances
    if not 1 <= N <= 8:
        ap.error("--instances must be 1..8")
    # N == 1: the original ids/ops (byte-identical workload). N > 1: instance k uses the *_n ops (inst arg) and ids tick_{k}_{f} etc.
    sfx = lambda k: f"_{k}" if N > 1 else ""
    buffers, tasks = [], []
    for k in range(N):
        buffers += [{"name": f"d_input{sfx(k)}", "rows": 1, "cols": 1},       # zero: no input magic -> the scripted player plays
                    {"name": f"d_state{sfx(k)}", "rows": 1, "cols": 16 + STATE_REC * a.frames}]
        buffers += [{"name": f"fb{sfx(k)}_{f}", "rows": 1, "cols": FB_WORDS} for f in range(a.frames)]
        op = (lambda n: n + "_n") if N > 1 else (lambda n: n)
        ex = (lambda d: {**d, "inst": k}) if N > 1 else (lambda d: d)
        prev = []
        for f in range(a.frames):
            dl = int((f + 1) * 1e6 / 35)
            tick = {"id": f"tick{sfx(k)}_{f}", "op": op("doom_tick"), "priority": a.doom_priority, "deadline_us": dl, "after": prev,
                    "args": ex({"frame": f, "input": f"d_input{sfx(k)}", "state": f"d_state{sfx(k)}"})}
            tasks.append(tick)
            prev = []
            for s_ in range(SLICES):
                rows = 200 // SLICES
                tasks.append({"id": f"present{sfx(k)}_{f}_{s_}", "op": op("doom_present"), "priority": a.doom_priority, "deadline_us": dl, "after": [tick["id"]],
                              "args": ex({"fb": f"fb{sfx(k)}_{f}", "row0": s_ * rows, "rows": rows})})
                prev.append(f"present{sfx(k)}_{f}_{s_}")
            tasks.append({"id": f"status{sfx(k)}_{f}", "op": op("doom_status"), "priority": a.doom_priority, "deadline_us": dl, "after": [tick["id"]],
                          "args": ex({"frame": f, "state": f"d_state{sfx(k)}"})})
            prev.append(f"status{sfx(k)}_{f}")
    graphs = [{"name": "doom", "tasks": tasks}]

    if a.chains:
        rng = random.Random(7)
        buffers += [{"name": "mm_a", "rows": N, "cols": N, "init": {"kind": "values", "values": [rng.randint(-8, 8) / 8 for _ in range(N * N)]}},
                    {"name": "mm_b", "rows": N, "cols": N, "init": {"kind": "values", "values": [rng.randint(-8, 8) / 8 for _ in range(N * N)]}}]
        buffers += [{"name": f"mm_out_{c}", "rows": N, "cols": N} for c in range(a.chains)]
        ct = []
        for c in range(a.chains):
            for r in range(a.rounds):
                ct.append({"id": f"mm_{c}_{r}", "op": "matmul", "priority": a.compute_priority, "after": [f"spin_{c}_{r - 1}"] if r else [],
                           "args": {"a": "mm_a", "b": "mm_b", "out": f"mm_out_{c}", "m": N, "n": N, "k": N}})
                us = rng.choice([5000, 40000, 200000]) if a.burst else a.spin_us
                ct.append({"id": f"spin_{c}_{r}", "op": "spin", "priority": a.compute_priority, "after": [f"mm_{c}_{r}"], "args": {"us": us}})
        graphs.append({"name": "compute", "tasks": ct})
    json.dump({"buffers": buffers, "graphs": graphs}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(tasks)} doom tasks, {len(graphs[1]['tasks']) if a.chains else 0} compute tasks, {len(buffers)} buffers")


if __name__ == "__main__":
    main()
