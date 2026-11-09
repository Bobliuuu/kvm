#!/usr/bin/env python3
"""SM-occupancy animation from the real trace: demo/assets/sm_occupancy.gif. Usage: make_anim.py [--data DIR] [--out DIR]

One window of a few ms (chosen as in make_charts.anchor_window: all three classes active), dynamic and static side by side, same demo and
code, only --sched differs. Each cell is one SM (%smid), coloured by the class of the task it is executing at that instant, with a short fading
trail of what it ran in the last 60 us. The curves below count busy SMs per class. The animation is slowed down by the printed factor.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).parent))
import vizdata as V
from make_charts import anchor_window

COLS, TRAIL_US, NFRAMES, FPS = 12, 60.0, 100, 20
ORDER = ("train", "serve", "game")


def window_tasks(run, t0, t1):
    """Tasks overlapping [t0, t1] as (start_us, end_us, sm, class index), times relative to t0; noop bookkeeping tasks excluded."""
    return [(x["t0"] - t0, x["t1"] - t0, x["sm"], ORDER.index(run["cls"][x["wl"]])) for x in run["trace"]
            if x["t1"] >= t0 and x["t0"] <= t1 and x["op"] != "noop" and run["cls"][x["wl"]] in ORDER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(V.DATA))
    ap.add_argument("--out", default=str(V.ASSETS))
    a = ap.parse_args()
    modes = [m for m in ("dynamic", "static") if m in V.available_modes(a.data)]
    runs = {m: V.load_run(m, a.data) for m in modes}
    wins = {m: anchor_window(runs[m], 0.4, 3.0) for m in modes}
    span = wins[modes[0]][1] - wins[modes[0]][0]
    tasks = {m: window_tasks(runs[m], *wins[m]) for m in modes}
    nsm = max(r["run"]["sms"] for r in runs.values())
    rows = -(-nsm // COLS)
    times = np.linspace(0, span, NFRAMES)
    cols = np.array([to_rgb(V.COLORS[c]) for c in ORDER])
    idle = np.array(to_rgb(V.COLORS["idle"]))

    def state(m, t):
        """grid image and busy counts at time t: running task -> class colour; ran within TRAIL_US -> blended trail."""
        img = np.tile(idle, (rows * COLS, 1)); cnt = np.zeros(3)
        for s, e, sm, c in tasks[m]:
            if s <= t < e:
                img[sm] = cols[c]; cnt[c] += 1
            elif e <= t < e + TRAIL_US and tuple(img[sm]) == tuple(idle):
                w = 1 - (t - e) / TRAIL_US
                img[sm] = idle * (1 - 0.6 * w) + cols[c] * 0.6 * w
        return img.reshape(rows, COLS, 3), cnt

    # busy-count curves over the whole window (for the lower panels)
    grid = np.linspace(0, span, 300)
    curves = {m: np.array([state(m, t)[1] for t in grid]) for m in modes}
    ymax = max(c.sum(axis=1).max() for c in curves.values()) + 1

    fig = plt.figure(figsize=(9, 6), dpi=90)
    gs = fig.add_gridspec(2, len(modes), height_ratios=[3, 1.6], hspace=0.35)
    ims, cursors, lbls, names = {}, {}, {}, {}
    for j, m in enumerate(modes):
        ax = fig.add_subplot(gs[0, j]); ax.axis("off")
        ims[m] = ax.imshow(state(m, 0)[0], interpolation="nearest", aspect="equal")
        names[m] = m + (" (fixed 70/10/20 SM split)" if m == "static" else " (shared by weight)")
        lbls[m] = ax
        bx = fig.add_subplot(gs[1, j])
        bx.stackplot(grid / 1e3, curves[m].T, colors=[V.COLORS[c] for c in ORDER], step="mid")
        bx.set_xlim(0, span / 1e3); bx.set_ylim(0, ymax); bx.set_xlabel("ms in window", fontsize=8); bx.set_ylabel("busy SMs", fontsize=8)
        bx.tick_params(labelsize=7); bx.spines[["top", "right"]].set_visible(False)
        cursors[m] = bx.axvline(0, color="k", lw=1)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=V.COLORS[c]) for c in ORDER], labels=ORDER, loc="upper center", ncol=3, frameon=False, fontsize=8)
    slow = (NFRAMES / FPS) / (span / 1e6)
    fig.text(0.01, 0.005, f"Real A100 trace (%smid, %globaltimer): {span / 1e3:.0f} ms of GPU time played over {NFRAMES / FPS:.0f} s, i.e. slowed {slow:.0f}x. One cell = one SM.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    def update(i):
        t = times[i]
        for m in modes:
            img, cnt = state(m, t)
            ims[m].set_data(img); cursors[m].set_xdata([t / 1e3, t / 1e3])
            lbls[m].set_title(f"{names[m]}\nt = {t:5.0f} us   busy SMs: train {int(cnt[0])}  serve {int(cnt[1])}  game {int(cnt[2])}", fontsize=9, fontweight="bold")
        return []

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gif = out / "sm_occupancy.gif"
    w, h = (int(v) for v in fig.get_size_inches() * fig.dpi)
    # ffmpeg's two-pass palette keeps the text crisp and the file small (Pillow's frame-delta GIFs lose the white background)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                           "-vf", "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none", "-loop", "0", str(gif)], stdin=subprocess.PIPE)
    for i in range(NFRAMES):
        update(i); fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    ff.stdin.close(); ff.wait()
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.2f} MB), window {span / 1e3:.1f} ms, tasks { {m: len(t) for m, t in tasks.items()} }")


if __name__ == "__main__":
    main()
