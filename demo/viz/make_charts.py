#!/usr/bin/env python3
"""Static charts (SVG) from the real demo outputs. Usage: make_charts.py [--out DIR]

  sm_timeline.svg     SM x time, every task as a bar coloured by class, dynamic vs static split (same demo, same code, --sched differs)
  sm_occupancy.svg    busy SMs over the whole training burst per class, same two runs
  doom_frame_gaps.svg histogram of the gap between consecutive Doom ticks, per scheduler, with the 35 Hz period
  weight_versions.svg weight version published by training vs the version each serve request read, over time
  loss_tokens.svg     training loss per step and the tokens each serve request produced
  scaleout.svg        scale-out cost model (a SIMULATION; reads results/2026-11-04-scaleout-sim/tables.md)
  scoreboard.svg      where the VM wins and where it loses (numbers from results/, cited on the figure)
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import vizdata as V

matplotlib.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                            "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 100, "svg.hashsalt": "kvm"})
PERIOD_MS = 1000 / 35


def anchor_window(run, before_ms=1.0, span_ms=12.0):
    """A short window in which all three classes run: among windows starting just before the first task of a serve request, the one whose
    count of the scarcer of (serve, train) tasks is largest (Doom ticks run throughout after the boot). Returns (t0_us, t1_us)."""
    boot_end = V.doom_ticks(run)[0][1]
    srv = [x["t0"] for x in run["trace"] if run["cls"][x["wl"]] == "serve" and x["op"] != "noop"]
    trn = [x["t0"] for x in run["trace"] if run["cls"][x["wl"]] == "train" and x["op"] != "noop"]
    starts = sorted(x["t0"] for x in run["trace"] if run["cls"][x["wl"]] == "serve" and x["op"] == "embedding_fwd" and x["t0"] > boot_end)
    best, best_n = boot_end, -1
    for t in starts or [boot_end]:
        a, b = t - before_ms * 1e3, t + (span_ms - before_ms) * 1e3
        n = min(sum(1 for u in srv if a <= u <= b), sum(1 for u in trn if a <= u <= b))
        if n > best_n:
            best, best_n = t, n
    t0 = best - before_ms * 1e3
    return t0, t0 + span_ms * 1e3


def bars(run, t0, t1, ax, nsm):
    polys, cols = [], []
    for x in run["trace"]:
        if x["t1"] < t0 or x["t0"] > t1 or x["op"] == "noop":
            continue
        a, b = (max(x["t0"], t0) - t0) / 1e3, (min(x["t1"], t1) - t0) / 1e3
        polys.append([(a, x["sm"]), (b, x["sm"]), (b, x["sm"] + 0.9), (a, x["sm"] + 0.9)])
        cols.append(V.COLORS[run["cls"][x["wl"]]])
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="none", linewidths=0, rasterized=True))
    ax.set_xlim(0, (t1 - t0) / 1e3)
    ax.set_ylim(0, nsm)
    return len(polys)


def chart_sm_timeline(runs, out):
    modes = [m for m in ("dynamic", "static") if m in runs]
    fig, axs = plt.subplots(len(modes), 1, figsize=(10, 3.6 * len(modes)), sharex=True, squeeze=False)
    for ax, m in zip(axs[:, 0], modes):
        r = runs[m]
        t0, t1 = anchor_window(r)
        n = bars(r, t0, t1, ax, r["run"]["sms"])
        ax.set_ylabel("SM id")
        ax.set_title(f"{m} scheduler" + ("  (SMs split 70/10/20 train/serve/game, fixed)" if m == "static" else "  (classes share all SMs by weight)")
                     + f"   window starts at {t0 / 1e3:.0f} ms, {n} tasks", loc="left")
    axs[-1, 0].set_xlabel("ms (executor clock; window start differs per run)")
    axs[0, 0].legend(handles=[Patch(color=V.COLORS[c], label=c) for c in ("train", "serve", "game")], loc="lower right", ncol=3, frameon=True, fontsize=8)
    fig.suptitle("Who runs on which SM: real %smid / %globaltimer trace, 3 workloads in one persistent kernel (A100, 108 SMs)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "sm_timeline.svg", dpi=220)
    plt.close(fig)


def occupancy(run, bin_us=1000.0, t_end=None):
    """-> times_ms, {class: busy SMs averaged over each bin} (task-seconds / bin)."""
    t_end = t_end or max(x["t1"] for x in run["trace"])
    nb = int(t_end // bin_us) + 1
    acc = {c: [0.0] * nb for c in ("train", "serve", "game")}
    for x in run["trace"]:
        c = run["cls"][x["wl"]]
        if c not in acc or x["op"] == "noop":
            continue
        a, b = x["t0"], x["t1"]
        i = int(a // bin_us)
        while a < b and i < nb:
            e = min(b, (i + 1) * bin_us)
            acc[c][i] += (e - a) / bin_us
            a, i = e, i + 1
    return [(i + 0.5) * bin_us / 1e3 for i in range(nb)], acc


def chart_occupancy(runs, out):
    modes = [m for m in ("dynamic", "static") if m in runs]
    span = max(max(x["t1"] for x in runs[m]["trace"] if runs[m]["cls"][x["wl"]] != "game") for m in modes) * 1.03
    fig, axs = plt.subplots(len(modes), 1, figsize=(10, 2.6 * len(modes)), sharex=True, sharey=True, squeeze=False)
    for ax, m in zip(axs[:, 0], modes):
        r = runs[m]
        t, acc = occupancy(r, t_end=span)
        ax.stackplot(t, [acc[c] for c in ("train", "serve", "game")], colors=[V.COLORS[c] for c in ("train", "serve", "game")], step="mid")
        ax.set_ylabel("busy SMs (1 ms avg)")
        ax.set_title(f"{m}: mean busy SMs per 1 ms (the GPU has {r['run']['sms']}; this demo keeps only a few busy)", loc="left")
    axs[-1, 0].set_xlabel("ms")
    axs[0, 0].legend(handles=[Patch(color=V.COLORS[c], label=c) for c in ("train", "serve", "game")], ncol=3, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "sm_occupancy.svg", dpi=200)
    plt.close(fig)


def chart_doom_gaps(runs, out):
    modes = [m for m in V.MODES if m in runs]
    fig, axs = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 3.4), sharey=True, sharex=True, squeeze=False)
    gmax = max(max(V.tick_gaps_ms(runs[m])) for m in modes)
    bins = [i * gmax / 40 for i in range(41)]
    for ax, m in zip(axs[0], modes):
        g = V.tick_gaps_ms(runs[m])
        ax.hist(g, bins=bins, color=V.COLORS["game"])
        p50, p99 = V.pct(g, .5), V.pct(g, .99)
        ax.axvline(p99, color="k", ls="--", lw=1)
        ax.axvline(PERIOD_MS, color="#CC79A7", lw=1.5)
        ax.set_yscale("log")
        ax.set_title(f"{m}\np50 {p50:.2f}  p99 {p99:.2f}  max {max(g):.2f} ms", loc="left")
        ax.set_xlabel("gap between Doom ticks, ms")
    axs[0, 0].set_ylabel("frames (log)")
    axs[0, -1].legend(handles=[Line2D([], [], color="k", ls="--", label="p99"), Line2D([], [], color="#CC79A7", lw=1.5, label=f"35 Hz period {PERIOD_MS:.1f} ms")],
                      loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Doom tick gaps while training and serving run beside it (700 frames each, real trace; Doom is not paced here: it free-runs)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "doom_frame_gaps.svg")
    plt.close(fig)


def chart_versions(runs, out):
    r = runs["dynamic"]
    pub = sorted((x["t1"] / 1e3, x["v1"]) for x in r["trace"] if r["cls"][x["wl"]] == "train" and x["op"] == "noop" and x["v1"] >= 0)
    reads = {}
    for x in r["trace"]:
        if r["cls"][x["wl"]] == "serve" and x["v0"] >= 0:
            reads.setdefault(x["v0"], []).append(x["t0"] / 1e3)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.step([0] + [t for t, _ in pub], [0] + [v for _, v in pub], where="post", color=V.COLORS["train"], label="weights version published by training (end of each window)")
    ax.scatter([min(ts) for v, ts in sorted(reads.items())], [v for v, ts in sorted(reads.items())], color=V.COLORS["serve"], zorder=3, s=26,
               label="version read by a serve request (first task of the request)")
    ax.set_xlabel("ms"); ax.set_ylabel("weights version")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Training publishes, serving reads: every version-tagged task read the version the DAG required "
                 f"({r['run']['version_check']['exact']}/{r['run']['version_check']['checked']} exact, {r['run']['version_check']['violations']} violations)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "weight_versions.svg")
    plt.close(fig)


def chart_loss_tokens(runs, out):
    r = runs["dynamic"]
    ls, ss = V.loss_series(r), V.serve_series(r)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [3, 2]})
    a.plot([s for _, s, _ in ls], [l for _, _, l in ls], color=V.COLORS["train"])
    a.set_xlabel("training step"); a.set_ylabel("loss"); train_status = next((k["status"] for k in r["report"].get("checks", []) if k["name"] == "training"), "not checked")
    a.set_title(f"loss {ls[0][2]:.2f} -> {ls[-1][2]:.4f} over {len(ls)} steps (CPU golden check: {train_status})", loc="left")
    for t, _, toks in ss:
        for k, tok in enumerate(toks):
            b.text(t / 1e3, k, str(tok), ha="center", va="center", fontsize=8, color=V.COLORS["serve"], fontweight="bold")
    b.set_xlim(-20, max(t for t, _, _ in ss) / 1e3 + 20); b.set_ylim(-0.7, 3.7); b.set_yticks(range(4)); b.set_yticklabels([f"token {k}" for k in range(4)])
    b.set_xlabel("ms (request finished)"); serving = next((k["status"] for k in r["report"].get("checks", []) if k["name"] == "serving"), "not checked")
    b.set_title(f"{len(ss)} served requests, token ids (check vs golden decode: {serving})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "loss_tokens.svg")
    plt.close(fig)


def parse_md_table(text, header_start):
    """Rows of the first markdown table after the line starting with header_start."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header_start))
    rows = []
    for l in lines[i:]:
        if l.startswith("|"):
            rows.append([c.strip() for c in l.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[0], [r for r in rows[2:]]


def chart_scaleout(out):
    text = (V.REPO / "results" / "2026-11-04-scaleout-sim" / "tables.md").read_text()
    h, rows = parse_md_table(text.split("## 1. Headline", 1)[1], "| GPUs")
    pts = {}
    for r in rows:
        m = re.match(r"(\d+) \(WAN (\d+) Gb/s\)", r[0])
        pts.setdefault(int(m.group(2)), []).append((int(m.group(1)), float(r[1]), float(r[2].rstrip("%"))))
    h2, rows2 = parse_md_table(text.split("## 2. WAN sensitivity", 1)[1], "| one-way")
    bw = [int(c) for c in h2[1:]]
    row20 = next(r for r in rows2 if r[0] == "20 ms")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for bwv, col in ((100, "#D55E00"), (400, "#0072B2")):
        a.plot([p[0] for p in pts[bwv]], [p[1] for p in pts[bwv]], "o-", color=col, label=f"WAN {bwv} Gb/s per DC pair")
    a.set_xscale("log", base=2); a.set_xlabel("GPUs"); a.set_ylabel("step time, s"); a.legend(frameon=False)
    a.set_title("1T dense model, pipeline across 4 datacenters", loc="left")
    b.plot(bw, [float(x) for x in row20[1:]], "o-", color="#0072B2")
    b.set_xscale("log", base=2); b.set_xlabel("WAN Gb/s per datacenter pair"); b.set_ylabel("step time, s"); b.set_title("4096 GPUs, 20 ms one-way, plain 1F1B window", loc="left")
    fig.suptitle("SIMULATION (DES + cost model, no GPU, NIC or network was measured; uncalibrated): results/2026-11-04-scaleout-sim", x=0.01, ha="left", fontsize=9, color="#b00020")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "scaleout.svg")
    plt.close(fig)


def chart_scoreboard(out):
    """Numbers typed from committed result files (each cited); this is the only chart not computed from a run directory."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    # kvm.cu MLP, README table (A100): us/step/run, 1 run
    names = ["sequential\nlaunches", "sequential\nCUDA graphs", "batched\ngraph", "KVM"]
    v = [45.1, 38.0, 37.9, 25.3]
    a.bar(names, v, color=["#999", "#999", "#999", V.COLORS["train"]])
    for i, y in enumerate(v):
        a.text(i, y + 0.8, f"{y}", ha="center", fontsize=8)
    a.set_ylabel("us per step"); a.set_title("WIN: kvm.cu MLP step, 1 run: 1.50x over best baseline", loc="left", fontsize=9)
    # executor vs whole-step CUDA graph, claims audit table 5 (A100)
    labs = ["MLP (2 hidden)", "3-layer transformer"]
    graph, vm = [132.5, 981], [721, 3810]
    x = [0, 1]
    b.bar([i - 0.2 for i in x], graph, 0.4, color="#999", label="PyTorch whole-step CUDA graph")
    b.bar([i + 0.2 for i in x], vm, 0.4, color="#D55E00", label="KVM executor")
    for i in x:
        b.text(i - 0.2, graph[i] + 50, f"{graph[i]:g}", ha="center", fontsize=8); b.text(i + 0.2, vm[i] + 50, f"{vm[i]:g}", ha="center", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(labs); b.set_ylabel("us per step"); b.legend(frameon=False, fontsize=8, loc="upper left")
    b.set_title("LOSS: executor vs CUDA graph: 3.9-5.4x slower", loc="left", fontsize=9)
    fig.suptitle("Honest scoreboard (A100). Sources: README.md results table; results/2026-11-04-claims-audit/README.md gap 1. Ops are also 0.26-0.39x cuBLAS at 4096^3.",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "scoreboard.svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V.ASSETS))
    ap.add_argument("--data", default=str(V.DATA))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    modes = V.available_modes(a.data)
    if "dynamic" not in modes:
        sys.exit(f"no run data under {a.data}: run demo/viz/collect.sh then demo/viz/make_all.sh (or use the committed results/2026-11-05-showcase)")
    runs = {m: V.load_run(m, a.data) for m in modes}
    for m, r in runs.items():
        print(f"{m}: {len(r['trace'])} tasks")
    chart_sm_timeline(runs, out); chart_occupancy(runs, out); chart_doom_gaps(runs, out); chart_versions(runs, out); chart_loss_tokens(runs, out)
    chart_scaleout(out); chart_scoreboard(out)
    print("wrote", sorted(p.name for p in out.glob("*.svg")))


if __name__ == "__main__":
    main()
