# sourced by jobs. build_cached NAME FLAGS...  -> /tmp/ke_NAME (binary cached in the volume, keyed by source hash + flags)
FL="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177"
srchash() { (cd /src && cat kvm_exec.cu include/*.cuh include/*.h include/*.inc include/*.def 2>/dev/null | md5sum | cut -c1-12); }
build_cached() {  # name flags
  local n=$1; shift; local key; key=$(srchash)-$(echo "$*" | md5sum | cut -c1-6); mkdir -p /out/bin
  if [ -x /out/bin/ke_$key ]; then cp /out/bin/ke_$key /tmp/ke_$n; echo "cache hit $n $key"; return; fi
  ( cd /src && nvcc $FL "$@" kvm_exec.cu -o /tmp/ke_$n 2>&1 | grep -i error; cp /tmp/ke_$n /out/bin/ke_$key; echo "built $n $key" ) &
}
genbundles() {
  python3 -I results/2026-11-04-claims-audit/newmodel/gen.py /tmp/w/t_mlp mlp_ce --steps 200 --no-golden > $OUT/gen_mlp.log 2>&1 &
  python3 -I results/2026-11-04-claims-audit/newmodel/gen.py /tmp/w/t_xfmr xfmr_odd --steps 50 --no-golden > $OUT/gen_xfmr.log 2>&1 &
}
