"""Everything-green runner (Modal). One command builds every binary once (CPU containers, cached by a hash of the source tree),
runs every suite on its own A100 in parallel, and prints ONE PASS/FAIL table. Entry: scripts/run_all_green.sh.

  modal run --detach scripts/green_all.py --tag NAME [--only job1,job2] [--rebuild]
  results:  volume kvm-green:/NAME/{table.txt,table.json,<job>.log}   ->   modal volume get kvm-green /NAME results/<dir>/raw

Nothing here names a host or a key. Binaries live in kvm-green:/bin/<tree-hash>/ so a re-run of an unchanged tree skips the ~40 min nvcc builds.
A job PASSES only if its script's last line is `RESULT: PASS ...`; any other ending (including a hang killed by the job timeout) is FAIL.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import time

import modal

_p = pathlib.Path(__file__).resolve().parents
ROOT = _p[1] if len(_p) > 1 else pathlib.Path('/')
IGN = [".tmp", "results", "logs", ".git", ".claude", "docs", "assets", "**/__pycache__", "doom/third_party", "doom/build", "out", "demo/kvm_run"]
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("rsync", "git", "curl", "ca-certificates", "clang", "llvm", "lld", "build-essential", "time")
         .pip_install("numpy", "torch", "matplotlib")
         .add_local_dir(str(ROOT), "/src", copy=True, ignore=IGN))
app = modal.App("kvm-green", image=image)
vol = modal.Volume.from_name("kvm-green", create_if_missing=True)

NV = ('nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177')
HEAD = r'''
set -u
export PATH=$PATH:/usr/local/cuda/bin KVM_ARCH=sm_80 NVCC_ARCH=sm_80 KVM_NO_LOCK=1
touch ~/.gpu.lock
trap 'mkdir -p /logs/$TAG/$JOB.d; cp /tmp/*.txt /logs/$TAG/$JOB.d/ 2>/dev/null; gzip -f /logs/$TAG/$JOB.d/*.txt 2>/dev/null' EXIT
B=/logs/bin/$KEY; BZ=/logs/bin/$KEYZ
echo "# job=$JOB tag=$TAG key=$KEY $(date -u +%FT%TZ)"; nvcc --version | tail -1
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader 2>/dev/null || echo "(no GPU in this container)"
'''

# ---------------------------------------------------------------- builds (CPU containers; outputs cached in /logs/bin/<key>)
OPS = ('cd tests/gpu_ops && nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. %s.cu -o %s')
OPSF = ('cd tests/gpu_ops && nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. %s.cu -o %s')
BUILDS = {
    "test_exec": ('bash tests/gpu/build.sh', ["test_exec"]),
    "test_exec_bf": ('bash tests/gpu/build.sh -DKVM_BREAK_FENCE && mv test_exec test_exec_bf', ["test_exec_bf"]),
    "test_exec_bw": ('bash tests/gpu/build.sh -DKVM_BREAK_WINDOW_DONE && mv test_exec test_exec_bw', ["test_exec_bw"]),
    "test_exec_bp": ('bash tests/gpu/build.sh -DKVM_BREAK_PARK_RECHECK && mv test_exec test_exec_bp', ["test_exec_bp"]),
    "kvm_exec": (NV + ' kvm_exec.cu -o kvm_exec', ["kvm_exec"]),
    "test_sched": ('bash tests/gpu/build_sched.sh', ["test_sched"]),
    "test_ops": ('(' + OPS % ("test_ops", "test_ops") + ' && nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. test_e2e.cu -o test_e2e && mv test_ops test_e2e ../..)', ["test_ops", "test_e2e"]),
    "test_extra": ('(' + OPSF % ("test_extra", "test_extra") + ' && mv test_extra ../..)', ["test_extra"]),
    "test_bf16": ('(' + OPSF % ("test_bf16", "test_bf16") + ' && mv test_bf16 ../..)', ["test_bf16"]),
    "litmus": ('mkdir -p out && bash tests/litmus/build.sh "$PWD/out" && '
               'nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -lineinfo -DKVM_EXEC_NO_MAIN -DKVM_NEW tests/litmus/e2e.cu -o out/e2e && '
               'nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -lineinfo -DKVM_EXEC_NO_MAIN -DKVM_NEW -DKVM_BREAK_FENCE tests/litmus/e2e.cu -o out/e2e_breakfence && '
               'mkdir -p litmus_bins && cp out/litmus_* out/e2e out/e2e_breakfence litmus_bins/ && rm -rf out/hdr_*', ["litmus_bins"]),
    "kvm_run_nogame": (NV + ' -I. demo/kvm_run.cu -o kvm_run_nogame', ["kvm_run_nogame"]),
    "kvm_run_game": ('bash doom/setup.sh && bash doom/build.sh && bash doom/cpu_build.sh && ' + NV.replace('-Iinclude', '-rdc=true -DKVM_WITH_GAME -I. -Iinclude') +
                     ' demo/kvm_run.cu doom/build/doom_dev.o -o kvm_run_game && tar czf doom_assets.tgz doom/build doom/third_party', ["kvm_run_game", "doom_assets.tgz"]),
    "zoo": ('JOBS=12 bash examples/models/prebuild.sh zoo_bundles examples/models/cases.txt && JOBS=8 bash examples/models/prebuild.sh zoo_bundles_bf16s examples/models/cases_bf16s.txt', ["zoo_bundles", "zoo_bundles_bf16s"]),
}


def _hash(paths):
    h = hashlib.sha256()
    for top in paths:
        top = os.path.join("/src", top)
        if os.path.isfile(top):
            h.update(top.encode()); h.update(open(top, "rb").read()); continue
        for dp, dn, fn in os.walk(top):
            dn[:] = sorted(d for d in dn if d not in ("__pycache__", "out", "third_party", "build"))
            for f in sorted(fn):
                p = os.path.join(dp, f)
                h.update(p.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def tree_key():
    """(binary key, bundle key): binaries depend only on the CUDA/C++ sources; generated bundles also on the Python generators."""
    k = _hash(["include", "kvm_exec.cu", "tests/gpu", "tests/gpu_ops", "tests/litmus", "demo/kvm_run.cu", "doom"])
    return k, _hash(["frontend", "runtime", "examples/models", "include/kvm_abi.h"]) + k


def _drive(script, tag, job, key, logname, env=None, timeout=None, keyz=""):
    """Run a bash script in /src with a live log committed to the volume every 30 s. Returns (rc, tail)."""
    out = f"/logs/{tag}"; os.makedirs(out, exist_ok=True)
    path = f"{out}/{logname}"
    f = open(path, "w")
    e = dict(os.environ, TAG=tag, JOB=job, KEY=key, KEYZ=keyz, **(env or {}))
    p = subprocess.Popen(["bash", "-c", "cd /src && " + HEAD + script], stdout=f, stderr=subprocess.STDOUT, text=True, env=e, start_new_session=True)
    t0 = time.time(); rc = None
    while True:
        try:
            rc = p.wait(timeout=30); break
        except subprocess.TimeoutExpired:
            f.flush(); vol.commit()
            if timeout and time.time() - t0 > timeout:
                import signal
                os.killpg(p.pid, signal.SIGKILL); rc = -9; f.write(f"\nRESULT: FAIL job timeout {timeout}s (killed)\n"); break
    f.close(); vol.commit()
    last = [l for l in open(path, errors="replace").read().splitlines() if l.startswith("RESULT:")]
    return rc, (last[-1] if last else "RESULT: FAIL no result line (script died or hung)")


@app.function(cpu=16, memory=98304, timeout=4 * 3600, volumes={"/logs": vol})
def build(name: str, key: str, keyz: str, tag: str):
    cmd, outs = BUILDS[name]
    dest = f"/logs/bin/{keyz if name == 'zoo' else key}"; os.makedirs(dest, exist_ok=True)
    vol.reload()
    if all(os.path.exists(f"{dest}/{o}") for o in outs):
        return name, True, "cached"
    t0 = time.time()
    script = f'{cmd}\nrc=$?\nif [ $rc -eq 0 ]; then for o in {" ".join(outs)}; do cp -r "$o" {dest}/; done; fi\necho "build rc $rc"; exit $rc'
    rc, _ = _drive(script, tag, "build-" + name, key, f"build-{name}.log", keyz=keyz)
    return name, rc == 0, f"{time.time() - t0:.0f}s rc {rc}"


# ---------------------------------------------------------------- test jobs (each ends with a RESULT: line)
# helper: pass if rc==0 and the output has the given pattern
PASSFN = r'''
res() { # rc pattern file label
  if [ "$1" -eq 0 ] && grep -qE "$2" "$3"; then echo "RESULT: PASS $4: $(grep -E "$2" "$3" | tail -1)"; else echo "RESULT: FAIL $4: exit $1; $(tail -n 3 "$3" | tr '\n' ' ' | cut -c1-300)"; fi; }
'''
JOBS = {}
JOBS["exec"] = ('cp $B/test_exec .; python3 tests/gpu/make_initgen_vectors.py --check || { echo "RESULT: FAIL initgen vectors"; exit 1; }\n'
                'timeout -k 10 2400 ./test_exec > /tmp/o.txt 2>&1; rc=$?; gzip -c /tmp/o.txt > /logs/$TAG/exec_full.txt.gz; tail -n 25 /tmp/o.txt\n'
                'res $rc "^[0-9]+ checks, 0 failed, 0 suites failed" /tmp/o.txt "executor suite"', "A100-40GB")
JOBS["recycle5"] = ('cp $B/test_exec .\nok=0; for i in 1 2 3 4 5; do timeout -k 10 900 ./test_exec recycle > /tmp/r$i.txt 2>&1; rc=$?; tail -n 2 /tmp/r$i.txt; '
                    'grep -qE "^[0-9]+ checks, 0 failed" /tmp/r$i.txt && [ $rc -eq 0 ] && ok=$((ok+1)); done\n'
                    '[ $ok -eq 5 ] && echo "RESULT: PASS recycle suite (3M tasks via 4096 slots + window wrap) 5/5 runs" || echo "RESULT: FAIL recycle $ok/5 runs passed"', "A100-40GB")
JOBS["negative"] = ('cp $B/test_exec_bf $B/test_exec_bp $B/test_exec_bw .\ntimeout -k 10 2400 ./test_exec_bf --expect-fail versions > /tmp/n1.txt 2>&1; r1=$?; tail -n 3 /tmp/n1.txt\n'
                    'timeout -k 10 2400 ./test_exec_bp --expect-fail litmus > /tmp/n2.txt 2>&1; r2=$?; tail -n 3 /tmp/n2.txt\n'
                    'timeout -k 10 900 ./test_exec_bw recycle > /tmp/n3.txt 2>&1; r3=$?; grep -E "window_done false|checks," /tmp/n3.txt | tail -n 3\n'
                    '[ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -ne 0 ] && grep -q "window_done false" /tmp/n3.txt && echo "RESULT: PASS all 3 broken builds are detected (KVM_BREAK_FENCE, KVM_BREAK_PARK_RECHECK, KVM_BREAK_WINDOW_DONE = the recycle flake)" || echo "RESULT: FAIL negatives rc $r1 $r2 $r3"', "A100-40GB")
JOBS["sanitize"] = ('cp $B/test_exec .\nbad=0; for t in memcheck racecheck synccheck; do echo "== $t"; timeout -k 10 3000 compute-sanitizer --tool $t --error-exitcode 9 ./test_exec smoke > /tmp/s_$t.txt 2>&1; rc=$?; tail -n 6 /tmp/s_$t.txt; echo "sanitizer exit $rc"; [ $rc -eq 0 ] || bad=$((bad+1)); done\n'
                    '[ $bad -eq 0 ] && echo "RESULT: PASS test_exec smoke under memcheck, racecheck, synccheck: 0 errors" || echo "RESULT: FAIL $bad sanitizer(s) reported errors"', "A100-40GB")
JOBS["litmus"] = ('KVM_LITMUS_PREBUILT=$B/litmus_bins bash tests/litmus/run_litmus.sh > /tmp/l.txt 2>&1; rc=$?; cat /tmp/l.txt | cut -c1-200\n'
                  'res $rc "^LITMUS PASS" /tmp/l.txt "litmus: $(grep ^SUMMARY /tmp/l.txt)"', "A100-40GB")
JOBS["gpt_modes"] = ('KVM_EXEC_BIN=$B/kvm_exec bash tests/gpu/run_gpt.sh > /tmp/g.txt 2>&1; rc=$?; tail -n 20 /tmp/g.txt | cut -c1-200\nres $rc "^GPT PASS" /tmp/g.txt "gpt_tiny 544 tasks, 6 scheduler modes, identical buffers, sanitizers"', "A100-40GB")
JOBS["ops"] = ('cp $B/test_ops $B/test_e2e .\nbad=0; for t in test_ops test_e2e; do timeout -k 10 900 ./$t > /tmp/$t.txt 2>&1; rc=$?; echo "$t: $(tail -n 1 /tmp/$t.txt)"; [ $rc -eq 0 ] || { bad=$((bad+1)); tail -n 20 /tmp/$t.txt; }; done\n'
               'echo "== memcheck on test_ops attention"; TEST_ONLY=attention timeout -k 10 1800 compute-sanitizer --tool memcheck --error-exitcode 9 ./test_ops 2>&1 | tail -n 5; [ ${PIPESTATUS[0]} -eq 0 ] || bad=$((bad+1))\n'
               '[ $bad -eq 0 ] && echo "RESULT: PASS test_ops ($(tail -n1 /tmp/test_ops.txt)), test_e2e ($(tail -n1 /tmp/test_e2e.txt)), attention memcheck clean" || echo "RESULT: FAIL ops $bad"', "A100-40GB")
JOBS["extra"] = ('cp $B/test_extra .\ntimeout -k 10 1500 ./test_extra > /tmp/x.txt 2>&1; rc=$?; tail -n 4 /tmp/x.txt\nbad=0; for t in memcheck racecheck synccheck; do echo "== $t"; timeout -k 10 3000 compute-sanitizer --tool $t --error-exitcode 9 ./test_extra > /tmp/xs_$t.txt 2>&1; r=$?; tail -n 4 /tmp/xs_$t.txt; [ $r -eq 0 ] || bad=$((bad+1)); done\n'
                 '[ $rc -eq 0 ] && [ $bad -eq 0 ] && echo "RESULT: PASS test_extra ($(tail -n1 /tmp/x.txt)); sanitizers clean" || echo "RESULT: FAIL test_extra rc $rc, $bad sanitizer failures"', "A100-40GB")
JOBS["bf16_ops"] = ('cp $B/test_bf16 .\ntimeout -k 10 3000 ./test_bf16 > /tmp/b.txt 2>&1; rc=$?; tail -n 4 /tmp/b.txt\nbad=0; for t in memcheck racecheck synccheck; do echo "== $t"; TEST_BF16_SMALL=1 timeout -k 10 3000 compute-sanitizer --tool $t --error-exitcode 9 ./test_bf16 > /tmp/bs_$t.txt 2>&1; r=$?; tail -n 3 /tmp/bs_$t.txt; [ $r -eq 0 ] || bad=$((bad+1)); done\n'
                    '[ $rc -eq 0 ] && [ $bad -eq 0 ] && echo "RESULT: PASS test_bf16 ($(tail -n1 /tmp/b.txt)); sanitizers clean" || echo "RESULT: FAIL test_bf16 rc $rc, $bad sanitizer failures"', "A100-40GB")
JOBS["sched"] = ('cp $B/test_sched .\ntimeout -k 10 3000 ./test_sched > /tmp/sc.txt 2>&1; rc=$?; tail -n 12 /tmp/sc.txt\nres $rc "^[0-9]+ checks, 0 failed, 0 suites failed" /tmp/sc.txt "scheduler suite"', "A100-40GB")
JOBS["zoo"] = ('cp -r $BZ/zoo_bundles /tmp/zb; cp -r $BZ/zoo_bundles_bf16s /tmp/zb16\n'
               'KVM_EXEC_BIN=$B/kvm_exec bash examples/models/run_box.sh /tmp/zb /tmp/zoo_out > /tmp/z1.txt 2>&1; r1=$?; grep -E "exit|PASS|FAIL|OK|mismatch" /tmp/z1.txt | tail -n 60 | cut -c1-220\n'
               'KVM_EXEC_BIN=$B/kvm_exec bash examples/models/run_box.sh /tmp/zb16 /tmp/zoo16_out --cases examples/models/cases_bf16s.txt > /tmp/z2.txt 2>&1; r2=$?; grep -E "exit|PASS|FAIL|OK|mismatch" /tmp/z2.txt | tail -n 30 | cut -c1-220\n'
               'mkdir -p /logs/$TAG/zoo; cp /tmp/z1.txt /tmp/z2.txt /logs/$TAG/zoo/; for d in /tmp/zoo_out/bundles/* /tmp/zoo16_out/bundles/*; do cp $d/check.txt /logs/$TAG/zoo/$(basename $d).check.txt 2>/dev/null; cp $d/check.json /logs/$TAG/zoo/$(basename $d).check.json 2>/dev/null; done\n'
               'n1=$(grep -c "^PASS\\|PASS " /tmp/z1.txt); echo "zoo rc $r1 / bf16s rc $r2"\n'
               '[ $r1 -eq 0 ] && [ $r2 -eq 0 ] && echo "RESULT: PASS model zoo: all cases of cases.txt and cases_bf16s.txt pass check.py (sanitizers on san_* cases)" || echo "RESULT: FAIL zoo rc $r1/$r2: $(grep -iE "FAIL|mismatch" /tmp/z1.txt /tmp/z2.txt | head -n 6 | tr "\\n" " " | cut -c1-400)"', "A100-40GB")
JOBS["gptdiff"] = (r'''cp $B/kvm_exec .
fail=0; mkdir -p /tmp/gd
for spec in "cpu 1" "cpu 2" "cpu 5" "a100 2"; do read -r tiling steps <<< "$spec"; for mode in tf32 precise; do
  c="${tiling}_s${steps}_$mode"; python3 tests/gptdiff/gen_bundle.py /tmp/gd/$c --steps $steps --tiling $tiling $([ $mode = precise ] && echo --precise) > /dev/null 2>&1
  ./kvm_exec --blocks 4 --quiet --dump-buffers /tmp/gd/$c/bundle > /tmp/gd/$c/dump.txt 2> /tmp/gd/$c/err.txt; echo "$c exit $?"
  python3 tests/gptdiff/golden_check.py /tmp/gd/$c/workload.json /tmp/gd/$c/bundle /tmp/gd/$c/dump.txt --mode $mode | tail -n 3 | tee /tmp/gd/$c/check.txt; [ ${PIPESTATUS[0]} -eq 0 ] || fail=$((fail+1))
done; done
for c in cpu_s2_tf32 cpu_s5_precise; do for t in memcheck racecheck synccheck; do compute-sanitizer --tool $t --error-exitcode 9 ./kvm_exec --blocks 4 --quiet /tmp/gd/$c/bundle > /dev/null 2>&1 || { echo "sanitizer $t $c FAILED"; fail=$((fail+1)); }; done; done
[ $fail -eq 0 ] && echo "RESULT: PASS gpt_tiny AdamW bundles (1/2/5 steps, 1xTF32 and 3xTF32) match the CPU golden; sanitizers clean" || echo "RESULT: FAIL gptdiff $fail failures"''', "A100-40GB")
JOBS["bf16_bundles"] = (r'''cp $B/kvm_exec .
python3 tests/bf16/gen.py /tmp/b16 > /tmp/b16gen.txt 2>&1; fail=0
for c in f32_s50 bf16s_s50 bf16s_s1 bf16s_s5; do ./kvm_exec --blocks 4 --quiet --dump-buffers /tmp/b16/$c/bundle > /tmp/b16/$c/dump.txt 2> /tmp/b16/$c/err.txt; echo "$c exit $?"
  g=""; case "$c" in bf16s_s1|bf16s_s5) g="--golden";; esac
  python3 tests/bf16/check.py /tmp/b16/$c /tmp/b16/$c/dump.txt $g | tail -n 4; [ ${PIPESTATUS[0]} -eq 0 ] || fail=$((fail+1)); done
[ $fail -eq 0 ] && echo "RESULT: PASS bf16 storage bundles (f32/bf16s, 1-50 steps) check against the golden" || echo "RESULT: FAIL bf16 bundles $fail"''', "A100-40GB")
JOBS["grads162"] = (r'''cp $B/kvm_run_nogame demo/kvm_run; chmod +x demo/kvm_run; fail=0
for P in bf16s f32; do PRECISION=$P KVM_COMMIT=$KEY bash bench/vs_pytorch/grads162.sh /tmp/num_$P /logs/$TAG/grads162_$P > /tmp/g_$P.txt 2>&1; echo "== $P"; grep -E "worst tensor|loss" /tmp/g_$P.txt | tail -n 4 | cut -c1-260
  python3 - $P /tmp/g_$P.txt <<'PY' || fail=$((fail+1))
import re, sys
P, txt = sys.argv[1], open(sys.argv[2]).read()
m = re.search(r"all \d+ tensors: max exec ([0-9.e+-]+), max torch-tf32 ([0-9.e+-]+)", txt)
if not m: print("no grads line"); sys.exit(1)
ex, tf = float(m.group(1)), float(m.group(2))
lim = 2.5e-2 if P == "bf16s" else max(1.5 * tf, 2e-3)
print(f"{P}: worst exec-vs-fp64 {ex:.2e} (limit {lim:.1e}, torch-tf32 worst {tf:.2e})"); sys.exit(0 if ex <= lim else 1)
PY
done
[ $fail -eq 0 ] && echo "RESULT: PASS 162M step-1 gradients vs PyTorch float64: bf16s and f32 within bound ($(grep -h 'worst exec' /tmp/g_*.txt | tr '\n' ' '))" || echo "RESULT: FAIL grads162 $fail"''', "A100-80GB")
JOBS["learn300"] = (r'''cp $B/kvm_run_nogame demo/kvm_run; chmod +x demo/kvm_run; mkdir -p /work
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"data_vocab":256,"data_noise":0.1}'
echo "$M" > /logs/$TAG/learn_model.json
KVM_SCRATCH=/work/scr bash tests/gpt162/learning.sh /logs/$TAG/learn "$M" 300 3e-4 5 8 > /tmp/learn.txt 2>&1; rc=$?; tail -n 12 /tmp/learn.txt | cut -c1-220
python3 tests/gpt162/learning_plot.py /logs/$TAG/learn --model-json "$M" --png /logs/$TAG/learn/loss.png --csv /logs/$TAG/learn/loss.csv > /tmp/plot.txt 2>&1; tail -n 5 /tmp/plot.txt
python3 - <<'PY'
import json, sys
s = json.load(open("/logs/%s/learn/summary.json" % __import__("os").environ["TAG"]))
g = s["max_rel_gap_vs_tf32"]["value"]; nf = s["max_rel_gap_tf32_vs_fp32 (noise floor)"]["value"]
le, lt = s["last_mean"]["executor"], s["last_mean"]["torch_tf32"]
ok = g < 1e-3 and abs(le - lt) < 1e-3 and le < 1.0
print(f"max gap vs torch-tf32 {g:.2e} (noise floor {nf:.2e}); last-30 mean exec {le:.4f} torch {lt:.4f}")
print("RESULT: %s 300-step learning curve vs torch: max per-step gap %.2e (bound 1e-3), last-30 mean %.4f vs %.4f" % ("PASS" if ok else "FAIL", g, le, lt))
PY''', "A100-80GB")
JOBS["demo"] = (r'''tar xzf $B/doom_assets.tgz; cp $B/kvm_run_game demo/kvm_run; chmod +x demo/kvm_run; touch demo/kvm_run
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"precision":"bf16s"}'
DEMO_TAG=big MODEL="$M" TILING=a100 STEPS=6 WINDOW=1 SERVE_EVERY=2 FRAMES=700 PRECISE=0 LR=3e-4 WARMUP=2 MAXB=8 SANITIZE=0 GOLDEN=0 RUN_ARGS="" LOCK_WAIT=60 \
  bash demo/remote_demo.sh > /tmp/demo.txt 2>&1; rc=$?; mkdir -p /logs/$TAG/demo; cp runs/big/check.log runs/big/report.json runs/big/kvm_run.log /logs/$TAG/demo/ 2>/dev/null; grep -E "^(PASS|FAIL|SKIP|INFO|ALL)" runs/big/check.log | cut -c1-240
grep -E "window sizing|windows|records" runs/big/kvm_run.log | head -n 5
if [ $rc -eq 0 ] && grep -q "ALL CHECKS PASSED" runs/big/check.log && grep -q "raw-identical 700/700" runs/big/check.log; then echo "RESULT: PASS demo (162M bf16s training + serving + Doom): $(grep -h 'game: compared' runs/big/check.log | cut -c1-120); $(grep -h 'cadence' runs/big/check.log | cut -c1-120)"; else echo "RESULT: FAIL demo rc $rc"; fi''', "A100-80GB")
JOBS["multigpu"] = (r'''nvidia-smi -L; KVM_STAGE_TIMEOUT=900 bash dist/run_multigpu_tests.sh > /tmp/mg.txt 2>&1; rc=$?; grep -E "^(PASS|FAIL|SKIP) |^RESULT|passed" /tmp/mg.txt | cut -c1-200; mkdir -p /logs/$TAG/multigpu; cp /tmp/mg.txt /logs/$TAG/multigpu/run.txt; cp -r logs/*-dist /logs/$TAG/multigpu/ 2>/dev/null
nsk=$(grep -c "^SKIP " /tmp/mg.txt); nsk_ok=$(grep "^SKIP " /tmp/mg.txt | grep -c "multinode")
if [ $rc -eq 0 ] && grep -q "^RESULT: PASS" /tmp/mg.txt && [ "$nsk" -le "$nsk_ok" ] && grep -q "PASS multigpu-p2p-pingpong" /tmp/mg.txt; then echo "RESULT: PASS distributed suite on 2 physical A100s: $(grep -E '[0-9]+ passed' /tmp/mg.txt | tail -1) (only the multi-node stage skipped)"; else echo "RESULT: FAIL multigpu rc $rc, $(grep -E '^(FAIL|SKIP)' /tmp/mg.txt | tr '\n' ';' | cut -c1-300)"; fi''', "A100-40GB:2")
JOBS["unit"] = (r'''fail=0
python3 -I -m unittest discover -s tests > /tmp/u1.txt 2>&1 || fail=$((fail+1)); tail -n 3 /tmp/u1.txt
python3 -I -m unittest discover -s tests/frontend > /tmp/u2.txt 2>&1 || fail=$((fail+1)); tail -n 3 /tmp/u2.txt
python3 -I -m unittest discover -s tests/dist -p 'test_*.py' > /tmp/u3.txt 2>&1 || fail=$((fail+1)); tail -n 3 /tmp/u3.txt
python3 -I -m unittest discover -s examples/models -p 'test_*.py' > /tmp/u4.txt 2>&1 || fail=$((fail+1)); tail -n 3 /tmp/u4.txt
c++ -std=c++17 -Iinclude tests/host/test_adapt.cpp -o /tmp/t && /tmp/t | tail -n 2 || fail=$((fail+1))
for f in $(find . -name '*.sh' -not -path './doom/third_party/*'); do bash -n "$f" || fail=$((fail+1)); done
[ $fail -eq 0 ] && echo "RESULT: PASS CPU unit tests ($(grep -h '^Ran' /tmp/u1.txt /tmp/u2.txt /tmp/u3.txt /tmp/u4.txt | tr '\n' ' ')), host adaptive test, shell syntax" || echo "RESULT: FAIL unit $fail"''', None)

NEEDS = {  # job -> builds it needs
    "exec": ["test_exec"], "recycle5": ["test_exec"], "negative": ["test_exec_bf", "test_exec_bp", "test_exec_bw"], "sanitize": ["test_exec"], "litmus": ["litmus"],
    "gpt_modes": ["kvm_exec"], "ops": ["test_ops"], "extra": ["test_extra"], "bf16_ops": ["test_bf16"], "sched": ["test_sched"], "zoo": ["zoo", "kvm_exec"],
    "gptdiff": ["kvm_exec"], "multigpu": [], "bf16_bundles": ["kvm_exec"], "grads162": ["kvm_run_nogame"], "learn300": ["kvm_run_nogame"], "demo": ["kvm_run_game"], "unit": [],
}


def _job(name, tag, key, keyz):
    vol.reload()   # builds finished in other containers after this one's mount snapshot
    script, _ = JOBS[name]
    rc, last = _drive(PASSFN + script, tag, name, key, f"{name}.log", timeout=3.4 * 3600, keyz=keyz)
    return name, last


@app.function(gpu="A100-40GB", cpu=8, memory=65536, timeout=4 * 3600, volumes={"/logs": vol})
def job40(name: str, tag: str, key: str, keyz: str):
    return _job(name, tag, key, keyz)


@app.function(gpu="A100-40GB:2", cpu=8, memory=98304, timeout=4 * 3600, volumes={"/logs": vol})
def job40x2(name: str, tag: str, key: str, keyz: str):
    return _job(name, tag, key, keyz)


@app.function(gpu="A100-80GB", cpu=8, memory=98304, timeout=4 * 3600, volumes={"/logs": vol})
def job80(name: str, tag: str, key: str, keyz: str):
    return _job(name, tag, key, keyz)


@app.function(cpu=8, memory=32768, timeout=2 * 3600, volumes={"/logs": vol})
def job_cpu(name: str, tag: str, key: str, keyz: str):
    return _job(name, tag, key, keyz)


@app.function(cpu=2, memory=8192, timeout=14 * 3600, volumes={"/logs": vol})
def orchestrate(tag: str, only: str, rebuild: bool):
    key, keyz = tree_key()
    names = [n for n in JOBS if (not only or n in only.split(","))]
    need = sorted({b for n in names for b in NEEDS[n]})
    print(f"tree {key}; jobs {names}; builds {need}", flush=True)
    os.makedirs(f"/logs/{tag}", exist_ok=True)
    if rebuild:
        import shutil
        shutil.rmtree(f"/logs/bin/{key}", ignore_errors=True)
    t0 = time.time()
    # builds and the CPU job start now; each GPU job starts as soon as ITS builds finished
    bcalls = {b: build.spawn(b, key, keyz, tag) for b in need}
    calls = {}; done = {}; bres = {}
    pending = set(names)
    while pending or calls:
        for b, c in list(bcalls.items()):
            try:
                bres[b] = c.get(timeout=0); del bcalls[b]; print("build", bres[b], flush=True)
            except Exception as ex:
                if "imeout" in type(ex).__name__: continue
                bres[b] = (b, False, f"build crashed: {ex}"); del bcalls[b]; print("build", bres[b], flush=True)
        for n in sorted(pending):
            if all(b in bres for b in NEEDS[n]):
                pending.discard(n)
                if not all(bres[b][1] for b in NEEDS[n]):
                    done[n] = "RESULT: FAIL build failed: " + ", ".join(f"{b}:{bres[b][2]}" for b in NEEDS[n] if not bres[b][1])
                    continue
                gpu = JOBS[n][1]
                fn = job_cpu if gpu is None else (job80 if gpu == "A100-80GB" else job40x2 if gpu.endswith(":2") else job40)
                calls[n] = fn.spawn(n, tag, key, keyz); print("started", n, f"+{time.time() - t0:.0f}s", flush=True)
        for n, c in list(calls.items()):
            try:
                done[n] = c.get(timeout=0)[1]; del calls[n]; print("done", n, done[n][:160], flush=True)
            except Exception as ex:
                if "imeout" in type(ex).__name__: continue
                done[n] = f"RESULT: FAIL job crashed: {ex}"; del calls[n]
        time.sleep(15)
    rows = []
    for n in names:
        r = done[n]; ok = r.startswith("RESULT: PASS")
        rows.append((n, "PASS" if ok else "FAIL", r[len("RESULT: "):][5:].strip() if ok else r[len("RESULT: "):][5:].strip()))
    w = max(len(r[0]) for r in rows)
    table = [f"kvm green run {tag}  tree {key}  {time.strftime('%FT%TZ', time.gmtime())}  wall {time.time() - t0:.0f}s", ""]
    table += [f"{n:<{w}}  {s}  {d}" for n, s, d in rows]
    table += ["", f"OVERALL: {'ALL GREEN' if all(r[1] == 'PASS' for r in rows) else 'RED: ' + ', '.join(r[0] for r in rows if r[1] != 'PASS')}"]
    txt = "\n".join(table)
    print(txt, flush=True)
    open(f"/logs/{tag}/table.txt", "w").write(txt + "\n")
    json.dump({"tag": tag, "tree": key, "rows": rows, "builds": bres}, open(f"/logs/{tag}/table.json", "w"), indent=1)
    vol.commit()
    return txt


@app.local_entrypoint()
def main(tag: str = "green", only: str = "", rebuild: bool = False):
    print(orchestrate.remote(tag, only, rebuild))   # blocks; with --detach the run survives this client
