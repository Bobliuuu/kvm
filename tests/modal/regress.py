"""Merged-executor regression A/B (A100 via Modal): executor of the pre-merge main vs the current tree, same bundle, interleaved reps.
  git archive <pre-merge main> | tar -x -C /tmp/pre_tree
  PRE_TREE=/tmp/pre_tree modal run --detach tests/modal/regress.py --tag NAME [--variants now,now72,now80]
Builds on CPU containers (cached in volume kvm-green:/regress/<hash>), then one GPU container runs interleaved rounds of every variant.
Variant "now72" is the current executor launched with 72 KiB of dynamic smem (env KVM_DYN_SMEM; the f32 bundle runs no bf16 op).
"""
import hashlib
import os
import pathlib
import subprocess
import time

import modal

_p = pathlib.Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) > 2 else pathlib.Path('/')
PRE = os.environ.get("PRE_TREE", "/tmp/pre_tree")
IGN = [".tmp", "results", "logs", ".git", ".claude", "docs", "assets", "**/__pycache__", "doom/third_party", "doom/build", "out"]
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("rsync", "git").pip_install("numpy")
         .add_local_dir(str(ROOT), "/src", copy=True, ignore=IGN)
         .add_local_dir(PRE, "/pre", copy=True, ignore=IGN))
app = modal.App("kvm-regress", image=image)
vol = modal.Volume.from_name("kvm-green", create_if_missing=True)
F = "-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"


def _h(path):
    h = hashlib.sha256()
    for f in ["kvm_exec.cu"] + sorted(str(p.relative_to(path)) for p in (path / "include").rglob("*")):
        q = path / f
        if q.is_file():
            h.update(q.read_bytes())
    return h.hexdigest()[:12]


@app.function(cpu=16, memory=98304, timeout=4 * 3600, volumes={"/logs": vol})
def build(which: str):
    src = "/src" if which == "now" else "/pre"
    d = f"/logs/regress/{which}-{_h(pathlib.Path(src))}"
    os.makedirs(d, exist_ok=True)
    vol.reload()
    if os.path.exists(d + "/kvm_exec"):
        return d, "cached"
    t0 = time.time()
    r = subprocess.run(f"cd {src} && export PATH=$PATH:/usr/local/cuda/bin && nvcc {F} kvm_exec.cu -o {d}/kvm_exec", shell=True, capture_output=True, text=True)
    vol.commit()
    return d, f"rc {r.returncode} {time.time() - t0:.0f}s {r.stderr[-300:]}"


AB = r'''export PATH=$PATH:/usr/local/cuda/bin; O=/logs/TAG; mkdir -p $O /w
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader | tee $O/env.txt
cd /pre && python3 bench/vs_pytorch/gen.py /w/m_f32 --model 'MODEL' --steps 1 --mode tf32 --tiling a100 > $O/gen.log 2>&1; ls /w/m_f32/bundle | head -3
for r in $(seq 1 REPS); do
  echo "rep $r pre $(timeout 900 DPRE/kvm_exec --quiet /w/m_f32/bundle 2>&1 | grep -E '^(OK|FAIL)' | tail -1)" | tee -a $O/ab.txt
  for v in VARIANTS; do
    case $v in now) E="A=1";; now72) E="KVM_DYN_SMEM=73728";; now80) E="KVM_DYN_SMEM=81920";; esac
    echo "rep $r $v $(env $E timeout 900 DNOW/kvm_exec --quiet /w/m_f32/bundle 2>&1 | grep -E '^(OK|FAIL)|rror' | tail -1)" | tee -a $O/ab.txt
  done
done
python3 - <<PY | tee $O/summary.txt
import re, statistics as st, collections
d = collections.defaultdict(list)
for l in open("$O/ab.txt"):
    m = re.match(r"rep \d+ (\S+) OK: .* ([0-9.]+) ms", l)
    if m: d[m.group(1)].append(float(m.group(2)))
base = st.median(d["pre"]) if d["pre"] else float("nan")
for k, v in d.items(): print("%-8s median %8.1f ms  min %8.1f  max %8.1f  vs pre %6.3f  n=%d" % (k, st.median(v), min(v), max(v), st.median(v) / base, len(v)))
PY
'''


@app.function(gpu="A100-40GB", cpu=8, memory=65536, timeout=3 * 3600, volumes={"/logs": vol})
def ab(tag: str, dpre: str, dnow: str, variants: str, reps: int):
    vol.reload()
    M = '{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4}'
    sh = AB.replace("TAG", tag).replace("MODEL", M).replace("REPS", str(reps)).replace("DPRE", dpre).replace("DNOW", dnow).replace("VARIANTS", variants.replace(",", " "))
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True)
    vol.commit()
    return r.stdout[-3000:] + r.stderr[-1000:]


@app.local_entrypoint()
def main(tag: str = "regress", variants: str = "now,now72,now80", reps: int = 7):
    pre, now = list(build.map(["pre", "now"]))
    print(pre, now)
    for b in (pre, now):
        if b[1] != "cached" and "rc 0" not in b[1]:
            raise SystemExit("build failed")
    print(ab.remote(tag, pre[0], now[0], variants, reps))
