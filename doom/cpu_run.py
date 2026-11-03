#!/usr/bin/env python3
"""Run the CPU oracle (cpu_build.sh) for N tics and write what the GPU runs are compared against.

  cpu_run.py --frames 700 --out DIR     DIR/frames.bin (N x 65024 raw frames), DIR/summary.json (hashes, state, game outcome)

Frame f is the screen after doom_tick(frame=f), i.e. after f+1 game tics (frame 0 includes the boot). Same format as doomtest.
"""
import argparse
import ctypes
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_BYTES = 64000 + 1024
STATE_NAMES = ["booted", "tics", "gamestate", "health", "armor", "kills", "monsters", "items", "secrets", "deaths",
               "level_finished_tic", "x", "y", "waypoint"]
log = logging.getLogger("cpu_run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=700)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wad", default=os.path.join(HERE, "third_party", "doom1.wad"))
    ap.add_argument("--lib", default=os.path.join(HERE, "build", "libdoom_cpu.so"))
    ap.add_argument("--input", default="", help="optional file of little-endian u32 input words, one per tic (default: scripted player)")
    ap.add_argument("--log", default="debug")
    a = ap.parse_args()
    logging.basicConfig(level=a.log.upper(), format="[cpu_run %(asctime)s] %(levelname)s %(message)s", datefmt="%T")
    if not os.path.exists(a.lib):
        sys.exit(f"{a.lib} missing: run doom/cpu_build.sh")
    inputs = []
    if a.input:
        raw = open(a.input, "rb").read()
        inputs = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
    os.makedirs(a.out, exist_ok=True)

    lib = ctypes.CDLL(a.lib)
    lib.doom_create.argtypes = [ctypes.c_char_p]
    lib.doom_tic.argtypes = [ctypes.c_uint32]
    saved = os.dup(1)                                  # the engine prints its start-up log to stdout
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    lib.doom_create(a.wad.encode())
    st = (ctypes.c_int32 * 16)()
    hs = (ctypes.c_uint32 * 2)()
    raw = ctypes.create_string_buffer(FRAME_BYTES)
    hashes, states = [], []
    with open(os.path.join(a.out, "frames.bin"), "wb") as fb:
        for f in range(a.frames):
            lib.doom_tic(inputs[f] if f < len(inputs) else 0xFFFFFFFF)
            lib.doom_state(st)
            lib.doom_frame_hash(hs)
            lib.doom_frame_raw(raw)
            fb.write(raw.raw)
            hashes.append([hs[0], hs[1]])
            states.append(list(st)[:14])
    ctypes.CDLL(None).fflush(None)
    os.dup2(saved, 1)
    last = dict(zip(STATE_NAMES, states[-1]))
    summary = {"frames": a.frames, "hashes": hashes, "state_names": STATE_NAMES, "states": states, "final": last}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"))
    log.info("%d frames; final %s", a.frames, json.dumps(last))


if __name__ == "__main__":
    main()
