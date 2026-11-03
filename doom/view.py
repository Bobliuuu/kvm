#!/usr/bin/env python3
"""Turn dumped Doom frames (frames.bin: N x [64000 palette indices + 256 palette words]) into something you can watch.

  view.py frames.bin --png DIR [--every 5] [--scale 2]      PNG sequence (stdlib only: zlib)
  view.py frames.bin --video out.mp4|out.gif [--fps 35]     needs imageio (+ imageio-ffmpeg for mp4); numpy
  view.py frames.bin --frame 700 --png DIR                  a single frame
"""
import argparse
import logging
import os
import struct
import sys
import zlib

W, H = 320, 200
FB = W * H + 1024
log = logging.getLogger("view")


def rgb_rows(frame, scale):
    """Yield scanlines (bytes, RGB) of one raw frame. The palette word is b,g,r,a in memory (i_video.c's struct color)."""
    pal = frame[W * H:]
    lut = [bytes((pal[4 * i + 2], pal[4 * i + 1], pal[4 * i])) for i in range(256)]
    for y in range(H):
        row = b"".join(lut[p] * scale for p in frame[y * W:(y + 1) * W])
        for _ in range(scale):
            yield row


def png_bytes(frame, scale=1):
    raw = b"".join(b"\0" + r for r in rgb_rows(frame, scale))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W * scale, H * scale, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--png", default="", help="directory for a PNG sequence")
    ap.add_argument("--video", default="", help=".mp4 or .gif output")
    ap.add_argument("--frame", type=int, default=-1, help="only this frame")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=35)
    ap.add_argument("--log", default="info")
    a = ap.parse_args()
    logging.basicConfig(level=a.log.upper(), format="[view] %(levelname)s %(message)s")
    data = open(a.frames, "rb").read()
    if len(data) % FB:
        sys.exit(f"{a.frames}: size {len(data)} is not a multiple of the {FB}-byte frame")
    n = len(data) // FB
    idx = [a.frame] if a.frame >= 0 else range(0, n, a.every)
    log.info("%d frames in %s; writing %d", n, a.frames, len(idx))
    if a.png:
        os.makedirs(a.png, exist_ok=True)
        for f in idx:
            open(os.path.join(a.png, f"frame_{f:05d}.png"), "wb").write(png_bytes(data[f * FB:(f + 1) * FB], a.scale))
        log.info("PNG sequence in %s", a.png)
    if a.video:
        try:
            import imageio
            import numpy as np
        except ImportError:
            sys.exit("--video needs: pip install imageio imageio-ffmpeg numpy")
        w = imageio.get_writer(a.video, fps=a.fps) if a.video.endswith(".gif") else imageio.get_writer(a.video, fps=a.fps, codec="libx264", quality=5, macro_block_size=8)
        for f in idx:
            fr = data[f * FB:(f + 1) * FB]
            pal = np.frombuffer(fr[W * H:], np.uint8).reshape(256, 4)[:, [2, 1, 0]]
            img = pal[np.frombuffer(fr[:W * H], np.uint8)].reshape(H, W, 3)
            w.append_data(np.repeat(np.repeat(img, a.scale, 0), a.scale, 1))
        w.close()
        log.info("wrote %s", a.video)


if __name__ == "__main__":
    main()
