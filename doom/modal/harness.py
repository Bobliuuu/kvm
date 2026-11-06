#!/usr/bin/env python3
"""In-container job runner (see app.py). python3 harness.py JOB OUTDIR NINST [ARGS]. Prints RESULT <json> lines, appends them to OUTDIR/results.jsonl."""
import csv
import json
import os
import statistics
import subprocess
import sys
import time

JOB, OUT, NINST = sys.argv[1], sys.argv[2], int(sys.argv[3])
ARGS = sys.argv[4] if len(sys.argv) > 4 else ""
WAD = "doom/third_party/doom1.wad"
DYN = "--sched dynamic --classes 2 --min 1,0 --urgent-us 20000,0"
t_start = time.time()


def emit(d):
    d["t"] = round(time.time() - t_start, 1)
    line = json.dumps(d)
    print("RESULT", line, flush=True)
    open(f"{OUT}/results.jsonl", "a").write(line + "\n")


def sh(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0.0


def oracle(frames):
    d = f"{OUT}/cpu"
    if not os.path.exists(d + "/summary.json"):
        rc, o = sh(f"python3 doom/cpu_run.py --frames {frames} --lib doom/build/libdoom_cpu.so --out {d} --log info")
        emit({"step": "oracle", "frames": frames, "rc": rc})
    return d


def phases(trace_csv, split):
    """Per-frame critical path of the Doom graph from the executor trace (workload 0)."""
    rows = [r for r in csv.DictReader(open(trace_csv)) if r["workload"] == "0" and float(r["end_us"]) > 0]
    frames, cur = [], None
    for r in rows:
        if r["op"] in ("doom_tick", "doom_tick_n", "doom_sim"):
            cur = []; frames.append(cur)
        if cur is not None: cur.append(r)
    out = {"sim_us": [], "draw_wall_us": [], "post_us": [], "frame_us": [], "draw_run_max_us": [], "hop_us": []}
    for fr in frames[2:]:
        by = lambda *ops: [r for r in fr if r["op"] in ops]
        sim = by("doom_tick", "doom_tick_n", "doom_sim")[0]
        out["sim_us"].append(float(sim["run_us"]))
        dr = by("doom_draw")
        if dr:
            out["draw_wall_us"].append(max(float(r["end_us"]) for r in dr) - min(float(r["start_us"]) for r in dr))
            out["draw_run_max_us"].append(max(float(r["run_us"]) for r in dr))
        po = by("doom_post")
        if po: out["post_us"].append(float(po[0]["run_us"]))
        out["frame_us"].append(max(float(r["end_us"]) for r in fr) - float(sim["start_us"]))
        # idle gap between the sim ending and the next stage starting (dispatch + wait)
        nxt = [r for r in fr if r is not sim and r["op"] in ("doom_draw", "doom_present", "doom_present_n")]
        if nxt: out["hop_us"].append(min(float(r["start_us"]) for r in nxt) - float(sim["end_us"]))
    return {k: {"p50": round(pct(v, .5), 1), "p99": round(pct(v, .99), 1), "mean": round(statistics.mean(v), 1)} for k, v in out.items() if v}


def run(name, frames, wl="", ex="", binary="bin/doom_exec_n1", inst=1, split=0, timeout=900, wrap="", cmp=True):
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    a = f"--frames {frames} --doom-priority 0 --compute-priority 0 {wl}" + (f" --split {split}" if split else "") + (f" --instances {inst}" if inst > 1 else "")
    rc, o = sh(f"python3 doom/make_workload.py {a} --out {d}/wl.json && python3 runtime/compile_workload.py {d}/wl.json {d}/bundle --allow-opaque 2>&1 | tail -1")
    if rc: emit({"name": name, "error": "compile", "log": o[-400:]}); return None
    t0 = time.time()
    rc, o = sh(f"{wrap} {binary} {d}/bundle --wad {WAD} --out {d}/run --timeout-s {timeout} {ex}", timeout=timeout + 120)
    res = {"name": name, "frames": frames, "split_tiles": split, "instances": inst, "exec": ex, "wl": wl, "rc": rc, "wall_s": round(time.time() - t0, 1)}
    open(f"{d}/exec.log", "w").write(o)
    if wrap:
        res["sanitizer"] = [l for l in o.splitlines() if "ERROR SUMMARY" in l or "RACECHECK SUMMARY" in l or "SYNCCHECK" in l or "Error" in l][:6]
    if os.path.exists(f"{d}/run/metrics.json"):
        m = json.load(open(f"{d}/run/metrics.json"))
        res["metrics"] = {k: m[k] for k in ("achieved_fps", "tick_gap_us", "frame_latency_us", "tick_gap_over_28571us", "ticks_flagged_missed", "sms_used_by_doom", "kernel_ms", "boot_tick_ms", "mean_block_utilization", "matmul_ok", "matmul_bad")}
        res["compute_finished_ms"] = None
    inst_lines = [l for l in o.splitlines() if l.startswith("[") and "instance " in l and "fps" in l] or [l for l in o.splitlines() if "instance " in l and "fps" in l]
    if inst_lines:
        fp = []
        for l in inst_lines:
            try: fp.append(float(l.split("instance")[1].split(":")[1].split("fps")[0]))
            except Exception: pass
        res["per_instance_fps"] = {"n": len(fp), "min": min(fp), "max": max(fp), "sum": round(sum(fp), 1)} if fp else None
    for l in o.splitlines():
        if "workload " in l and ("makespan" in l or "done" in l or "span" in l): res.setdefault("workload_lines", []).append(l.strip()[:160])
    if cmp and rc == 0:
        cp = f"{OUT}/cpu"
        dirs = [f"{d}/run/inst{k}" for k in range(inst)] if inst > 1 else [f"{d}/run"]
        outs = []
        for dd in dirs:
            r2, o2 = sh(f"python3 doom/compare.py {cp} {dd} --frames {frames}")
            outs.append(o2.strip().splitlines()[0] if o2.strip() else "no output")
        ok = all("raw-identical %d/%d" % (frames, frames) in x and "fingerprint-identical %d/%d" % (frames, frames) in x and "state-identical %d/%d" % (frames, frames) in x for x in outs)
        res["frame_exact"] = ok
        res["compare_first"] = outs[0]
        if not ok: res["compare_all"] = outs[:4]
    if os.path.exists(f"{d}/run/trace.csv") and rc == 0 and inst == 1:
        try: res["phases"] = phases(f"{d}/run/trace.csv", split)
        except Exception as e: res["phases_error"] = str(e)
    # keep the volume small: drop raw frames and bundles after comparing
    sh(f"rm -rf {d}/bundle {d}/run/frames.bin {d}/run/inst*/frames.bin {d}/wl.json")
    emit(res)
    return res


def job_verify():
    F = 300
    oracle(700)
    run("serial", F)
    for t in (1, 2, 4, 5, 8, 16, 32, 64): run(f"split{t}", F, split=t)
    run("split8_700", 700, split=8)
    run("serial_700", 700)
    load = "--chains 3 --rounds 40 --tenant train"
    run("serial_trainload_dyn", F, wl=load, ex=DYN + " --classes 2")
    run("split8_trainload_dyn", F, wl=load, split=8, ex=DYN)
    run("split8_trainload_global", F, wl=load, split=8, ex="--sched global --classes 2")


def job_sanitize():
    oracle(700)
    for tool in ("memcheck", "racecheck", "synccheck"):
        for nm, sp in (("serial", 0), ("split4", 4)):
            run(f"{tool}_{nm}", 12, split=sp, wrap=f"compute-sanitizer --tool {tool} --error-exitcode 9", timeout=1500)


def job_speed():
    oracle(700)
    for rep in range(3):
        run(f"serial_r{rep}", 700, cmp=(rep == 0))
        for t in (1, 2, 4, 8, 16, 32, 64): run(f"split{t}_r{rep}", 700, split=t, cmp=(rep == 0))
        run(f"serial_dyn_r{rep}", 700, ex=DYN, cmp=False)
        run(f"split16_dyn_r{rep}", 700, split=16, ex=DYN, cmp=False)


def job_instances():
    F = 120
    oracle(700)
    for n in [1, 2, 4, 8, 16, 32, 64]:
        if n > NINST: break
        b = "bin/doom_exec_nN" if n > 1 else "bin/doom_exec_n1"
        run(f"inst{n}_serial", F, binary=b, inst=n)
        run(f"inst{n}_split4", F, binary=b, inst=n, split=4)
    if NINST >= 16:
        load = "--chains 3 --rounds 40 --tenant train"
        run("inst16_split4_trainload", F, binary="bin/doom_exec_nN", inst=16, split=4, wl=load, ex=DYN)
        run("inst16_serial_trainload", F, binary="bin/doom_exec_nN", inst=16, wl=load, ex=DYN)


def job_prof():
    rc, o = sh("bin/doomprof " + WAD + f" 700 {OUT}/prof", timeout=600)
    emit({"step": "classic_phase_profile", "log": [l for l in o.splitlines() if "[prof]" in l or "tick time" in l]})


def job_study():
    """Doom (serial vs split) beside the GEMM+AdamW training tenant, across scheduler modes. NO spin task anywhere."""
    F = int(os.environ.get("STUDY_FRAMES", "300"))
    oracle(700)
    load = ARGS or "--chains 4 --rounds 400 --tenant train"
    modes = [("alone", None, ""), ("orig", "load", ""), ("global", "load", "--sched global --classes 2"), ("static10", "load", "--sched static --classes 2 --shares 1,9"),
             ("static25", "load", "--sched static --classes 2 --shares 1,3"), ("dyn", "load", DYN), ("dyn_noresv", "load", "--sched dynamic --classes 2 --min 0,0 --urgent-us 0,0")]
    for rep in range(int(os.environ.get("STUDY_REPS", "3"))):
        for dm, dsplit in (("serial", 0), ("split8", 8), ("split16", 16)):
            for mname, kind, ex in modes:
                if mname == "alone" and rep: continue
                run(f"{dm}_{mname}_r{rep}", F, wl=(load if kind else ""), split=dsplit, ex=ex, cmp=(rep == 0), timeout=300)


{"verify": job_verify, "sanitize": job_sanitize, "speed": job_speed, "instances": job_instances, "prof": job_prof, "study": job_study,
 "all": lambda: [j() for j in (job_prof, job_verify, job_speed, job_instances, job_study, job_sanitize)]}[JOB]()
emit({"step": "done", "job": JOB})
