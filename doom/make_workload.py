#!/usr/bin/env python3
"""Generate a workload JSON for runtime/compile_workload.py: real Doom frames as scheduler tasks, optionally beside a compute load.

  make_workload.py --frames 700 [--chains 128 --rounds 24] [--doom-priority 10] --out wl.json

graph "doom": per frame f   tick_f (doom_tick) -> present_f_0..3 (doom_present, 50 rows each) and status_f (doom_status, hash + state);
              tick_f+1 waits for all of them (tick_f+1 overwrites the screen they read). The 35 Hz pacing is deadline_us metadata only.
graph "compute": --chains independent chains of --rounds matmul tasks (128x128x128, the same A@B each round, so the result is checkable).
"""
import argparse
import json
import random

FB_WORDS = 16256          # KVM_DOOM_FB_WORDS: 64000 palette indices + 256 palette words
STATE_REC = 16            # KVM_DOOM_ST_REC
SLICES = 4
N = 128                   # matmul size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=700)
    ap.add_argument("--chains", type=int, default=0, help="compute chains (0: no competing load)")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--doom-priority", type=int, default=10)
    ap.add_argument("--compute-priority", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    buffers = [{"name": "d_input", "rows": 1, "cols": 1},                     # zero: no input magic -> the scripted player plays
               {"name": "d_state", "rows": 1, "cols": 16 + STATE_REC * a.frames}]
    buffers += [{"name": f"fb_{f}", "rows": 1, "cols": FB_WORDS} for f in range(a.frames)]
    tasks, prev = [], []
    for f in range(a.frames):
        dl = int((f + 1) * 1e6 / 35)
        tick = {"id": f"tick_{f}", "op": "doom_tick", "priority": a.doom_priority, "deadline_us": dl, "after": prev,
                "args": {"frame": f, "input": "d_input", "state": "d_state"}}
        tasks.append(tick)
        prev = []
        for k in range(SLICES):
            rows = 200 // SLICES
            tasks.append({"id": f"present_{f}_{k}", "op": "doom_present", "priority": a.doom_priority, "deadline_us": dl, "after": [tick["id"]],
                          "args": {"fb": f"fb_{f}", "row0": k * rows, "rows": rows}})
            prev.append(f"present_{f}_{k}")
        tasks.append({"id": f"status_{f}", "op": "doom_status", "priority": a.doom_priority, "deadline_us": dl, "after": [tick["id"]],
                      "args": {"frame": f, "state": "d_state"}})
        prev.append(f"status_{f}")
    graphs = [{"name": "doom", "tasks": tasks}]

    if a.chains:
        rng = random.Random(7)
        buffers += [{"name": "mm_a", "rows": N, "cols": N, "init": {"kind": "values", "values": [rng.randint(-8, 8) / 8 for _ in range(N * N)]}},
                    {"name": "mm_b", "rows": N, "cols": N, "init": {"kind": "values", "values": [rng.randint(-8, 8) / 8 for _ in range(N * N)]}}]
        buffers += [{"name": f"mm_out_{c}", "rows": N, "cols": N} for c in range(a.chains)]
        ct = []
        for c in range(a.chains):
            for r in range(a.rounds):
                ct.append({"id": f"mm_{c}_{r}", "op": "matmul", "priority": a.compute_priority, "after": [f"mm_{c}_{r - 1}"] if r else [],
                           "args": {"a": "mm_a", "b": "mm_b", "out": f"mm_out_{c}", "m": N, "n": N, "k": N}})
        graphs.append({"name": "compute", "tasks": ct})
    json.dump({"buffers": buffers, "graphs": graphs}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(tasks)} doom tasks, {len(graphs[1]['tasks']) if a.chains else 0} compute tasks, {len(buffers)} buffers")


if __name__ == "__main__":
    main()
