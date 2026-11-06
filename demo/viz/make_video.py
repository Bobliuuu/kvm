#!/usr/bin/env python3
"""Render demo/assets/doom_demo.mp4 (and a small .gif) from the REAL outputs of one demo run.

  make_video.py [--run DIR] [--out demo/assets/doom_demo.mp4] [--step N]

DIR (default logs/showcase-dynamic, from collect.sh / run_demo.sh --fetch-frames) must hold remote/out/{run.json,trace.csv,observations.jsonl},
remote/out/doom_candidate/frames.bin (frames the GPU executor produced) and remote/oracle/frames.bin (the CPU engine's frames). Left: the GPU
frame f. A line under it states, per frame, whether its bytes equal the CPU oracle's (sha256 computed here, not copied from a log). Right: training
loss and served tokens as of the executor time at which Doom tick f ended (from the trace). One video frame = one Doom tick; the video plays 35
ticks/s, the run itself did ~170/s, and training ends within the first tenth of the run: the rest of the video is Doom alone, as it was on the GPU.
Nothing is generated or interpolated: no frames.bin, no video.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import vizdata as V

FB, PIX = 64000 + 1024, 64000


def decode(raw):
    """65,024 raw bytes -> 200x320x3 uint8. Palette words are little-endian b,g,r,a (doom/glue/doom_glue.c)."""
    idx = np.frombuffer(raw, np.uint8, PIX).reshape(200, 320)
    pal = np.frombuffer(raw, np.uint8, 1024, PIX).reshape(256, 4)[:, [2, 1, 0]]
    return pal[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(V.REPO / "logs" / "showcase-dynamic"))
    ap.add_argument("--out", default=str(V.ASSETS / "doom_demo.mp4"))
    ap.add_argument("--fps", type=int, default=35)
    ap.add_argument("--step", type=int, default=1, help="render every Nth tick (default 1: all)")
    a = ap.parse_args()
    d = Path(a.run) / "remote"
    cand_p, orc_p = d / "out" / "doom_candidate" / "frames.bin", d / "oracle" / "frames.bin"
    if not (cand_p.exists() and orc_p.exists()):
        sys.exit(f"no frames under {d}: run demo/viz/collect.sh (it passes --fetch-frames). Skipping the video; not faking it.")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found")
    cand, orc = cand_p.read_bytes(), orc_p.read_bytes()
    n = len(cand) // FB
    assert len(orc) // FB >= n, "oracle has fewer frames than the candidate"
    r = V.load_run_dir(d / "out")
    ticks = V.doom_ticks(r)
    assert len(ticks) >= n, f"{len(ticks)} ticks in the trace for {n} frames"
    loss, serve = V.loss_series(r), V.serve_series(r)
    train_end_us = max(t for t, _, _ in loss)
    train_end_tick = next(f for f in range(n) if ticks[f][1] >= train_end_us)
    run_fps = (n - 2) / ((ticks[n - 1][0] - ticks[1][0]) / 1e6)
    same = [hashlib.sha256(cand[f * FB:(f + 1) * FB]).digest() == hashlib.sha256(orc[f * FB:(f + 1) * FB]).digest() for f in range(n)]
    print(f"{n} frames, byte-identical to the CPU oracle: {sum(same)}/{n}")
    if not all(same):
        print("WARNING: not all frames identical; the video says so per frame")

    fig, (al, at) = plt.subplots(2, 1, figsize=(6.4, 4.8), gridspec_kw={"height_ratios": [3, 1.3]}, dpi=100)
    lmax = max(l for _, _, l in loss)
    al.set_xlim(0, len(loss)); al.set_ylim(0, lmax * 1.05)
    al.set_xlabel("training step"); al.set_ylabel("loss"); al.set_title("training loss (GPU, same kernel)", loc="left", fontsize=9)
    (line,) = al.plot([], [], color=V.COLORS["train"])
    at.axis("off")
    txt = at.text(0, 1, "", va="top", family="monospace", fontsize=8)
    fig.tight_layout()
    frames_ix = list(range(0, n, a.step))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x500", "-r", str(a.fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for f in frames_ix:
        t_us = ticks[f][1]
        shown = [x for x in loss if x[0] <= t_us]
        line.set_data([s for _, s, _ in shown], [l for _, _, l in shown])
        served = [s for s in serve if s[0] <= t_us]
        lines = [f"executor time {t_us / 1e3:8.1f} ms   Doom tick {f:3d}", f"loss {shown[-1][2]:.4f} (step {shown[-1][1] + 1}/{len(loss)})" if shown else "loss: not yet reported",
                 f"serve requests answered: {len(served)}"]
        lines += [f"  {nm}: tokens {toks}" for _, nm, toks in served[-3:]]
        txt.set_text("\n".join(lines))
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        left = np.kron(decode(cand[f * FB:(f + 1) * FB]), np.ones((2, 2, 1), np.uint8))           # 640x400, nearest neighbour
        img = Image.new("RGB", (1280, 500), (255, 255, 255))
        img.paste(Image.fromarray(left), (0, 0))
        img.paste(Image.fromarray(right).resize((640, 480)), (640, 0))
        dr = ImageDraw.Draw(img)
        dr.text((8, 408), f"GPU frame {f}: {'BYTE-IDENTICAL' if same[f] else 'DIFFERS'} to the CPU oracle (sha256 compared)", fill=(0, 100, 0) if same[f] else (180, 0, 0))
        dr.text((8, 428), f"{sum(same[:f + 1])}/{f + 1} frames identical so far. Real frames from demo/run_demo.sh; video plays {a.fps} ticks/s,", fill=(60, 60, 60))
        dr.text((8, 444), f"the run did {run_fps:.0f}. Training ends at {train_end_us / 1e3:.0f} ms (tick {train_end_tick}); after that only Doom runs, as on the GPU.", fill=(60, 60, 60))
        ff.stdin.write(img.tobytes())
    ff.stdin.close(); ff.wait()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
