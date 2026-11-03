#!/usr/bin/env bash
# Fetch the Doom engine source and the shareware WAD into doom/third_party/ (git-ignored: GPL source and the WAD are never committed).
# Idempotent; pinned engine commit and WAD md5. Needs git/curl only.   LOG_LEVEL=debug|info|warn (default debug)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
ENGINE_REPO="${ENGINE_REPO:-https://github.com/ozkl/doomgeneric.git}"
ENGINE_COMMIT=e257a09527062bfb986b6e5e0780839b6cbd5b71
WAD_MD5=f0cefca49926d00903cf57551d901abe     # shareware doom1.wad v1.9
WAD_URLS=("https://github.com/Akbar30Bill/DOOM_wads/raw/master/doom1.wad")

mkdir -p "$TP"
if [ -d "$TP/doomgeneric/.git" ] && [ "$(git -C "$TP/doomgeneric" rev-parse HEAD)" = "$ENGINE_COMMIT" ]; then
  info "engine already at $ENGINE_COMMIT"
else
  info "fetching doomgeneric $ENGINE_COMMIT"
  rm -rf "$TP/doomgeneric"
  git init -q "$TP/doomgeneric"
  git -C "$TP/doomgeneric" fetch -q --depth 1 "$ENGINE_REPO" "$ENGINE_COMMIT"
  git -C "$TP/doomgeneric" checkout -q FETCH_HEAD
  [ "$(git -C "$TP/doomgeneric" rev-parse HEAD)" = "$ENGINE_COMMIT" ] || die "engine commit mismatch"
fi
wad_ok() { [ -s "$TP/doom1.wad" ] && [ "$(md5_of "$TP/doom1.wad")" = "$WAD_MD5" ]; }
if wad_ok; then info "WAD ok (md5 $WAD_MD5)"; else
  for u in "${WAD_URLS[@]}"; do
    debug "downloading $u"
    curl -fsSL -m 120 -o "$TP/doom1.wad" "$u" || true
    wad_ok && break || rm -f "$TP/doom1.wad"
  done
  wad_ok || die "doom1.wad missing or wrong md5: place the shareware doom1.wad (md5 $WAD_MD5) at $TP/doom1.wad"
  info "WAD fetched and verified"
fi
