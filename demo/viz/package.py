#!/usr/bin/env python3
"""Copy the raw outputs of collect.sh (logs/showcase-<mode>/remote/) into results/2026-11-05-showcase/<mode>/ so the charts regenerate on a
clean clone without a GPU: run.json, report.json, observations.jsonl and trace.csv (gzipped). Frames are NOT copied (45 MB each)."""
import gzip
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vizdata as V

found = 0
for m in V.MODES:
    src = V.REPO / "logs" / f"showcase-{m}" / "remote"
    if not (src / "out" / "trace.csv").exists():
        print(f"skip {m}: no {src}/out/trace.csv")
        continue
    dst = V.DATA / m
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("run.json", "observations.jsonl"):
        shutil.copy(src / "out" / f, dst / f)
    shutil.copy(src / "report.json", dst / "report.json")
    with open(src / "out" / "trace.csv", "rb") as i, gzip.GzipFile(dst / "trace.csv.gz", "wb", mtime=0) as o:
        shutil.copyfileobj(i, o)
    print(f"packaged {m} -> {dst}")
    found += 1
sys.exit(0 if found else "nothing packaged: run demo/viz/collect.sh first")
