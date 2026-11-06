#!/usr/bin/env bash
# Run a command (typically one nvcc-heavy build script) only when >= ${KVM_MIN_FREE_GB:-45} GB RAM is free; retry up to ${KVM_TRIES:-6}
# times if it is OOM-killed (exit 137) on a loaded shared box.   tools/box_serial.sh cmd args...
tries="${KVM_TRIES:-6}"
for i in $(seq 1 "$tries"); do
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "${KVM_MIN_FREE_GB:-45}" ]; do sleep 20; done
  "$@"; rc=$?
  [ "$rc" != 137 ] && exit "$rc"
  echo "box_serial: killed (OOM?), retry $i/$tries" >&2
done
exit 137
