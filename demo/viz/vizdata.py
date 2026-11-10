"""Loaders for the demo's real outputs (trace.csv from the executor, run.json, report.json, observations.jsonl).

A "run dir" holds those files, plain or trace.csv.gz. Everything the charts show is read from them: nothing is typed in or simulated
(the one exception, scaleout.svg, is a chart of the committed cost-model tables and says so on the figure).
"""
import csv
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results" / "2026-11-05-showcase"        # committed copies of the raw run outputs (traces gzipped)
ASSETS = REPO / "demo" / "assets"
MODES = ("dynamic", "static", "global")                   # scheduler modes collected by collect.sh
# Okabe-Ito colour-blind-safe palette; one colour per scheduler class everywhere.
COLORS = {"train": "#0072B2", "serve": "#E69F00", "game": "#009E73", "idle": "#e6e6e6"}


def _open(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def load_run(mode, root=DATA):
    return load_run_dir(Path(root) / mode, mode)


def load_run_dir(d, mode="run"):
    """-> dict(run=run.json, report=report.json, obs=[observations], trace=[row dicts], cls=[class name per workload id])."""
    d = Path(d)
    run = json.loads((d / "run.json").read_text())
    report = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else {}
    obs = [json.loads(line) for line in (d / "observations.jsonl").read_text().splitlines() if line.strip()]
    tp = d / "trace.csv.gz" if (d / "trace.csv.gz").exists() else d / "trace.csv"
    with _open(tp) as fh:
        rows = list(csv.DictReader(fh))
    trace = []
    for r in rows:
        if float(r["end_us"]) < 0:                        # never ran
            continue
        trace.append(dict(op=r["op"], wl=int(r["workload"]), t0=float(r["start_us"]), t1=float(r["end_us"]), sm=int(r["smid"]),
                          v0=int(r["ver_start"]), v1=int(r["ver_end"]), wait=float(r["queue_wait_us"])))
    # the trace's `workload` column is the scheduler class index, in run.json's `classes` order (admission `wids` are a different numbering)
    cls = list(run["classes"])
    assert set(x["wl"] for x in trace) <= set(range(len(cls))), "trace workload ids outside run.json classes"
    return dict(mode=mode, run=run, report=report, obs=obs, trace=trace, cls=cls)


def available_modes(root=DATA):
    return [m for m in MODES if (Path(root) / m / "run.json").exists()]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def doom_ticks(run):
    """Start/end (us) of every doom_tick task in order: frame f is tick f."""
    t = sorted((x for x in run["trace"] if x["op"] == "doom_tick"), key=lambda x: x["t0"])
    return [(x["t0"], x["t1"]) for x in t]


def tick_gaps_ms(run):
    """Gap between consecutive Doom tick starts in ms, without the boot frame (frame 0's tick contains the engine boot)."""
    s = [a for a, _ in doom_ticks(run)]
    return [(s[i] - s[i - 1]) / 1e3 for i in range(2, len(s))]


def loss_series(run):
    """[(t_done_us, step, loss)] from the train window observations (loss.N buffers, first value is the loss)."""
    out, step = [], 0
    for o in run["obs"]:
        if not o["adm"].startswith("train"):
            continue
        for k in sorted((k for k in o["buffers"] if k.startswith("loss.")), key=lambda k: int(k.split(".")[1])):
            out.append((o["t_done_us"], step, o["buffers"][k]["values"][0]))
            step += 1
    return out


def serve_series(run):
    """[(t_done_us, request name, [token ids])] for each served request."""
    out = []
    for o in run["obs"]:
        if o["adm"].startswith("serve"):
            toks = [o["buffers"][k]["values"][0] for k in sorted(o["buffers"], key=lambda k: int(k.rsplit("cur", 1)[1]))]
            out.append((o["t_done_us"], o["adm"], toks))
    return out
