#!/usr/bin/env python3
"""Compare two Doom runs (cpu_run.py / doomtest / the scheduler run) frame by frame: raw bytes (sha256) and the 64-bit fingerprints,
then the game outcome. Exit 0 only when every compared frame is identical and the final state matches.

  compare.py REF_DIR CAND_DIR [--frames N]
"""
import argparse
import hashlib
import json
import os
import sys

FB = 64000 + 1024


def load(d):
    s = json.load(open(os.path.join(d, "summary.json")))
    p = os.path.join(d, "frames.bin")
    return s, (open(p, "rb").read() if os.path.exists(p) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--frames", type=int, default=0)
    a = ap.parse_args()
    (rs, rb), (cs, cb) = load(a.ref), load(a.cand)
    n = min(rs["frames"], cs["frames"], a.frames or 1 << 30)
    bad_raw, bad_hash, bad_state = [], [], []
    for f in range(n):
        if rb is not None and cb is not None and hashlib.sha256(rb[f * FB:(f + 1) * FB]).digest() != hashlib.sha256(cb[f * FB:(f + 1) * FB]).digest():
            bad_raw.append(f)
        if rs["hashes"][f] != cs["hashes"][f]:
            bad_hash.append(f)
        if rs["states"][f] != cs["states"][f]:
            bad_state.append(f)
    names = rs["state_names"]
    print(f"compared {n} frames: raw-identical {n - len(bad_raw) if rb and cb else 'n/a'}/{n}, fingerprint-identical {n - len(bad_hash)}/{n}, "
          f"state-identical {n - len(bad_state)}/{n}")
    for label, bad in (("raw", bad_raw), ("fingerprint", bad_hash), ("state", bad_state)):
        if bad:
            f = bad[0]
            print(f"FIRST {label} MISMATCH at frame {f} ({len(bad)} total)")
            if label == "state":
                print("  ref ", dict(zip(names, rs["states"][f])), "\n  cand", dict(zip(names, cs["states"][f])))
    print("final ref ", dict(zip(names, rs["states"][n - 1])), "\nfinal cand", dict(zip(names, cs["states"][n - 1])))
    ok = not (bad_raw or bad_hash or bad_state)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
