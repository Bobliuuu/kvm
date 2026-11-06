#!/usr/bin/env python3
"""Generate a workload JSON for runtime/compile_workload.py: real Doom frames as scheduler tasks, optionally beside a compute load.

  make_workload.py --frames 700 [--chains 128 --rounds 24] [--doom-priority 10] --out wl.json

graph "doom": per frame f   tick_f (doom_tick) -> present_f_0..3 (doom_present, 50 rows each) and status_f (doom_status, hash + state);
              tick_f+1 waits for all of them (tick_f+1 overwrites the screen they read). The 35 Hz pacing is deadline_us metadata only.
graph "compute": --chains independent chains of (matmul 256^3 then spin) x --rounds. Every chain computes the same A@B, so the result is
                 checkable; the spins keep every block busy for the whole run (chains > worker blocks, so the ready queue is never empty).
--tenant train:  instead of spin tasks, each chain is a training-step workload with REAL compute and bandwidth pressure: per round two
                 tiled TF32 GEMMs (fwd Y=X@W, wgrad G=X^T@Y, dgrad D=Y@W^T; 64x64 matmul_tile tasks, a join task between GEMMs) and an AdamW
                 sweep over the W-sized parameters (p,g,m,v; 7 streams) in --adam-chunk element tasks. No timing task at all.
--split T:       Doom as the split task graph (doom_sim -> T x doom_draw -> doom_post -> present/status; include/kvm_ops_game.cuh), frame-exact
                 against the serial doom_tick. T = 0 (default) keeps the serial doom_tick chain.
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
    ap.add_argument("--split", type=int, default=0, help="split-tick Doom with this many draw tiles per frame (0: serial doom_tick)")
    ap.add_argument("--planes", type=int, default=4, help="split-tick Doom: tasks of the flat-plane stage (doom_planes), run before the column tiles")
    ap.add_argument("--tenant", choices=("spin", "train"), default="spin", help="competing load: matmul+spin chains (default) or GEMM+AdamW training chains")
    ap.add_argument("--gemm-m", type=int, default=1024, help="train tenant: GEMM output is M x M (multiple of 64)")
    ap.add_argument("--gemm-k", type=int, default=16384, help="train tenant: GEMM reduction depth (K-deep 64x64 tiles ~0.15 ms: long enough for the 0.5 M tasks/s admitter)")
    ap.add_argument("--adam-elems", type=int, default=8 << 20, help="train tenant: AdamW parameter elements per chain")
    ap.add_argument("--adam-chunk", type=int, default=2 << 20, help="train tenant: AdamW elements per task")
    ap.add_argument("--doom-priority", type=int, default=10)
    ap.add_argument("--compute-priority", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    N = a.instances
    if not 1 <= N <= 64:
        ap.error("--instances must be 1..64")
    # N == 1: the original ids/ops (byte-identical workload). N > 1: instance k uses the *_n ops (inst arg) and ids tick_{k}_{f} etc.
    sfx = lambda k: f"_{k}" if N > 1 else ""
    SPLIT = a.split
    buffers, tasks = [], []
    for k in range(N):
        buffers += [{"name": f"d_input{sfx(k)}", "rows": 1, "cols": 1},       # zero: no input magic -> the scripted player plays
                    {"name": f"d_state{sfx(k)}", "rows": 1, "cols": 16 + STATE_REC * a.frames}]
        buffers += [{"name": f"fb{sfx(k)}_{f}", "rows": 1, "cols": FB_WORDS} for f in range(a.frames)]
        op = (lambda n: n + "_n") if (N > 1 or SPLIT) else (lambda n: n)
        ex = (lambda d: {**d, "inst": k}) if (N > 1 or SPLIT) else (lambda d: d)
        prev = []
        for f in range(a.frames):
            dl = int((f + 1) * 1e6 / 35)
            tick = {"id": f"tick{sfx(k)}_{f}", "op": "doom_sim" if SPLIT else op("doom_tick"), "priority": a.doom_priority, "deadline_us": dl, "after": prev,
                    "args": ex({"frame": f, "input": f"d_input{sfx(k)}", "state": f"d_state{sfx(k)}"})}
            tasks.append(tick)
            prev = []
            frame_ready = tick["id"]      # the task after which the screen of frame f is complete
            if SPLIT:
                pl = [f"planes{sfx(k)}_{f}_{t}" for t in range(a.planes)]
                for t, tid_ in enumerate(pl):
                    tasks.append({"id": tid_, "op": "doom_planes", "priority": a.doom_priority, "deadline_us": dl, "after": [tick["id"]],
                                  "args": {"frame": f, "tile": t, "ntiles": a.planes, "inst": k}})
                tiles = [f"draw{sfx(k)}_{f}_{t}" for t in range(SPLIT)]
                for t, tid_ in enumerate(tiles):
                    tasks.append({"id": tid_, "op": "doom_draw", "priority": a.doom_priority, "deadline_us": dl, "after": pl,
                                  "args": {"frame": f, "tile": t, "ntiles": SPLIT, "inst": k}})
                tasks.append({"id": f"post{sfx(k)}_{f}", "op": "doom_post", "priority": a.doom_priority, "deadline_us": dl, "after": tiles,
                              "args": {"frame": f, "inst": k}})
                frame_ready = f"post{sfx(k)}_{f}"
            for s_ in range(SLICES):
                rows = 200 // SLICES
                tasks.append({"id": f"present{sfx(k)}_{f}_{s_}", "op": op("doom_present"), "priority": a.doom_priority, "deadline_us": dl, "after": [frame_ready],
                              "args": ex({"fb": f"fb{sfx(k)}_{f}", "row0": s_ * rows, "rows": rows})})
                prev.append(f"present{sfx(k)}_{f}_{s_}")
            tasks.append({"id": f"status{sfx(k)}_{f}", "op": op("doom_status"), "priority": a.doom_priority, "deadline_us": dl, "after": [frame_ready],
                          "args": ex({"frame": f, "state": f"d_state{sfx(k)}"})})
            prev.append(f"status{sfx(k)}_{f}")
    graphs = [{"name": "doom", "tasks": tasks}]

    if a.chains and a.tenant == "train":
        M, K, E, ch = a.gemm_m, a.gemm_k, a.adam_elems, a.adam_chunk
        buffers.append({"name": "tr_step", "rows": 1, "cols": 1, "dtype": "i32", "init": {"kind": "fill", "value": 1}})
        ct = []
        for c in range(a.chains):
            for nm, rows, cols, init in (("X", M, K, 1), ("X2", M, K, 2), ("W", K, M, 3), ("Y", M, M, 0), ("Y2", M, M, 0),
                                         ("P", 1, E, 4), ("G", 1, E, 5), ("Mo", 1, E, 0), ("Vo", 1, E, 0)):
                b_ = {"name": f"tr{c}_{nm}", "rows": rows, "cols": cols}
                if init: b_["init"] = {"kind": "seeded", "seed": 1000 * c + init, "scale": 0.02}
                buffers.append(b_)
            prev = []
            for r in range(a.rounds):
                ids = []
                for tag, X, Y in (("g1", "X", "Y"), ("g2", "X2", "Y2")):          # two forward GEMMs, K-deep tiles (compute-bound)
                    for m0 in range(0, M, 64):
                        for n0 in range(0, M, 64):
                            tid_ = f"{tag}_{c}_{r}_{m0}_{n0}"; ids.append(tid_)
                            ct.append({"id": tid_, "op": "matmul_tile", "priority": a.compute_priority, "after": prev,
                                       "args": {"a": f"tr{c}_{X}", "b": f"tr{c}_W", "out": f"tr{c}_{Y}", "bias": None, "res": None, "pre": None,
                                                "m0": m0, "n0": n0, "M": M, "N": M, "K": K, "flags": 0, "alpha": 1.0}})
                j = f"join_g_{c}_{r}"
                ct.append({"id": j, "op": "noop", "priority": a.compute_priority, "after": ids, "args": {}})
                ids = []
                for e0 in range(0, E, ch):                                         # AdamW sweep: 7 streams over the parameters (bandwidth-bound)
                    tid_ = f"adam_{c}_{r}_{e0}"; ids.append(tid_)
                    ct.append({"id": tid_, "op": "adamw", "priority": a.compute_priority, "after": [j],
                               "args": {"p": f"tr{c}_P", "g": f"tr{c}_G", "m": f"tr{c}_Mo", "v": f"tr{c}_Vo", "gscale": None, "step": "tr_step", "e0": e0, "e1": min(e0 + ch, E),
                                        "lr": 1e-4, "b1": 0.9, "b2": 0.999, "eps": 1e-8, "wd": 0.01}})
                jn = f"join_adam_{c}_{r}"
                ct.append({"id": jn, "op": "noop", "priority": a.compute_priority, "after": ids, "args": {}})
                prev = [jn]
        graphs.append({"name": "compute", "tasks": ct})
    elif a.chains:
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
