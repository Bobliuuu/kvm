#!/usr/bin/env bash
# Copy a fetched adaptive run directory into results/: every summary.json, status, compare.txt, run.log and the small per-event CSVs; the large
# timeline/controller/Doom-task CSVs only for repetition 1 of each configuration (gzipped). Usage: pack_results.sh FETCHED_DIR DEST_DIR
set -euo pipefail
SRC="${1:?fetched runs dir}"; DST="${2:?destination}"
mkdir -p "$DST"
for f in "$SRC"/*.txt "$SRC"/batch.log; do [ -f "$f" ] && cp "$f" "$DST/"; done
for cfg in "$SRC"/*/; do
  [ -f "$cfg/rep1/summary.json" ] || continue
  name="$(basename "$cfg")"
  for rep in "$cfg"/rep*/; do
    r="$(basename "$rep")"; d="$DST/$name/$r"; mkdir -p "$d"
    for f in summary.json status run.log doom_frames.csv serve.csv train.csv; do [ -f "$rep/$f" ] && cp "$rep/$f" "$d/"; done
    for k in "$rep"/doom[0-9]*/; do [ -f "$k/compare.txt" ] && cp "$k/compare.txt" "$d/compare_$(basename "$k").txt"; done
    if [ "$r" = rep1 ]; then for f in timeline.csv controller.csv doom_tasks.csv; do [ -f "$rep/$f" ] && gzip -c "$rep/$f" > "$d/$f.gz"; done; fi
  done
done
du -sh "$DST"
