#!/bin/bash
# the head of #48's chain) with KVM_FGPT_NT made overridable. Built 13:19Z and 13:25Z (fgpt_kernel<false> / <true>, the plain and the clocked
# instance): 182/178 registers at 256 threads, 160/157 at 384, 128/128 at 512; spills 0 at 256 and 384, 8 B / 0 at 512.
# resync at 64), then 5 rounds of the three arms, order rotating, at each shape's rule width (phase clock, 20 steps). Not bitwise by construction:
# the column sums' chunks, the block reductions' partials and the squares' per-thread sums follow NT; the hashes say whether they differ.
set -uo pipefail
T=/root/work/tree PY=/root/py/bin/python3; F=/root/work/fgsbed/run-20261111T074845Z; B=$F/bundles
W=/root/work/fgs384; RUN=$W/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $RUN/vm
for nt in 256 384 512; do cp $W/bed-$nt $RUN/bed-NT$nt; cp $W/build-$nt.log $RUN/; done
cp $F/vm/vm-*.json $RUN/vm/
cat > $RUN/gpu.sh <<EOG
#!/bin/bash
chk() { local arm=\$1 name=\$1-\$2 wl=\$3 steps=\$4 gangs=\$5; shift 5
  mkdir -p $RUN/work/\$name; ln -sf $RUN/vm/vm-*.json $RUN/work/\$name/
  $PY -I $T/bench/fused_gpt/fgpt_bed.py $B/\$wl --bed $RUN/bed-\$arm --steps \$steps --gangs \$gangs --work $RUN/work/\$name "\$@" > $RUN/check-\$name.log 2>&1
  local rc=\$?
  echo "CHECK \$name: \$([ \$rc = 0 ] && echo ok || echo "FAIL (rc \$rc)"); \$(grep -cE ': ok\$' $RUN/check-\$name.log) ok, \$(grep -cE ': FAIL' $RUN/check-\$name.log) FAIL"
  grep -E ': FAIL|no row chunk|exited [0-9]|Traceback|error' $RUN/check-\$name.log | head -3
}
for a in NT384 NT512; do
  chk \$a tiny-p-k2-g16 w_tiny_p 2 16; chk \$a small-p-k2-g32 w_small_p 2 32
  chk \$a d128-p-k1-g64 w_d128_precise 1 64; chk \$a d128-p-resync-g64 w_d128_precise 1 64 --resync 1
  chk \$a ho128-p-k1-g64 w_ho128_precise 1 64; chk \$a ho128-p-resync-g64 w_ho128_precise 1 64 --resync 1
done
for r in 1 2 3 4 5; do
  for wg in w_tiny:16 w_small:32 w_d128_tf32_k30:64 w_ho128_tf32_k30:64; do
    w=\${wg%%:*}; g=\${wg#*:}
    case \$((r % 3)) in 1) order="NT256 NT384 NT512";; 2) order="NT384 NT512 NT256";; 0) order="NT512 NT256 NT384";; esac
    for arm in \$order; do
      wd=$RUN/time-\$arm-\$w; lg=$RUN/phase-\$arm-\$w-r\$r.log
      $PY -I $T/bench/fused_gpt/fgpt_bed.py $B/\$w --bed $RUN/bed-\$arm --time-only --bed-args clock --steps 20 --gangs \$g --work \$wd > \$lg 2>&1
      echo "\$w round \$r \$arm \$(grep -oE '[0-9.]+ us a step in all; the SMs ran at [0-9]+ MHz|error word [1-9][0-9]*|Traceback' \$lg | head -1)"
      echo "HASH \$w g\$g r\$r \$arm \$(cat \$wd/out-G\$g-clock/*.bin 2>/dev/null | sha256sum | cut -c1-16)"
    done
  done
done
echo AW_GPU_END
EOG
chmod +x $RUN/gpu.sh
grep -E "^(CHECK|AW_)|FAIL|error|Traceback" $RUN/run.log
$PY - $RUN/run.log <<'EOP'
import sys, collections
h = collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(sys.argv[1]):
    p = l.split()
    if p and p[0] == 'HASH' and len(p) == 6: h[(p[1], p[2])][p[4]].add(p[5])
for (w, g), d in sorted(h.items()):
    each = {a: (next(iter(v)) if len(v) == 1 else f'{len(v)} values') for a, v in d.items()}
    same = len(set(each.values())) == 1
    print(f"HASHES {w} {g}: each arm's 5 rounds {'agree' if all(len(v) == 1 for v in d.values()) else 'DIFFER'}; across arms {'equal' if same else 'differ'} {each}")
EOP
for w in w_tiny w_small w_d128_tf32_k30 w_ho128_tf32_k30; do
  grep "^$w round" $RUN/run.log > $RUN/rounds-$w.log
  echo "== $w: $(for a in NT256 NT384 NT512; do echo -n "$a $(grep -m1 -oE 'rows in chunks of [0-9]+ of [0-9]+, shared memory [0-9]+ B' $RUN/phase-$a-$w-r1.log); "; done)"
  for c in NT384 NT512; do $PY /root/work/slowmode.py $RUN/rounds-$w.log NT256 $c 2>&1 | head -2; done
done
