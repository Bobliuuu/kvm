#!/usr/bin/env bash
# Regenerate every demo asset from real run data. No GPU needed unless you want a new run.
#   demo/viz/make_all.sh              charts + animation + demo/index.html from results/2026-11-05-showcase (committed raw data)
#                                     + the Doom video when logs/showcase-dynamic holds frames (else the committed video is kept)
#   demo/viz/make_all.sh --package    first copy logs/showcase-*/ (from collect.sh) into results/2026-11-05-showcase/
# New GPU data: KVM_HOST=user@host demo/viz/collect.sh && demo/viz/make_all.sh --package
# Needs: python3 with matplotlib, numpy, pillow; ffmpeg (GIF + MP4).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ "${1:-}" = "--package" ] && python3 demo/viz/package.py
command -v ffmpeg >/dev/null || { echo "ffmpeg missing (needed for the GIF and MP4)" >&2; exit 1; }
python3 demo/viz/make_charts.py
python3 demo/viz/make_anim.py
if [ -f logs/showcase-dynamic/remote/out/doom_candidate/frames.bin ]; then python3 demo/viz/make_video.py
else echo "no frames under logs/showcase-dynamic: keeping the committed demo/assets/doom_demo.mp4 (run demo/viz/collect.sh to rebuild it)"; fi
python3 demo/viz/make_html.py
ls -la demo/assets demo/index.html
