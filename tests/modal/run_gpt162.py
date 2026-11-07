"""gpt_162m executor-vs-PyTorch correctness check on an A100 via Modal (nothing in the file names a host or key).
Usage (repo root):  modal run --detach tests/modal/run_gpt162.py --stage build|check --tag NAME [--n1 40 --n2 50]
  build: compile demo/kvm_run (no game pack) for sm_80 on a CPU container into the volume.
  check: step-1 loss/gradients vs torch (reuse none, 1 step), then loss curves vs torch for the chain stream and for a repeated batch.
Logs and JSON land in volume 'kvm-gpt162' under /<tag>/ ; fetch with: modal volume get kvm-gpt162 /<tag> results/<dir>/raw
"""
import modal, os, pathlib

_p = pathlib.Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) > 2 else pathlib.Path('/')
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("rsync", "git")
         .pip_install("torch", "numpy")
         .add_local_dir(str(ROOT), "/src", copy=True,
                        ignore=[".tmp", "results", "logs", ".git", "docs", "**/__pycache__", "doom/third_party", "doom/build"]))
app = modal.App("kvm-gpt162", image=image)
vol = modal.Volume.from_name("kvm-gpt162", create_if_missing=True)

BUILD = r'''
mkdir -p /logs/BIN
nvcc --version | tail -1
time nvcc -O3 -arch=sm_80 -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 demo/kvm_run.cu -o /logs/BIN/kvm_run 2>&1 | tail -20
ls -la /logs/BIN
'''

# A: a 1-step window (reuse none keeps every gradient buffer alive) dumped for the torch comparison.
# B: window-1 runs of N steps, losses observed after every step, against torch's own N-step curves.
CHECK = r'''
set -u
OUT=/logs/$TAG; mkdir -p $OUT
cp /logs/BIN/kvm_run demo/kvm_run
echo "== environment"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 - <<PY
import torch; p = torch.cuda.get_device_properties(0)
print('torch', torch.__version__, 'cuda', torch.version.cuda, p.name, 'SMs', p.multi_processor_count)
PY
M162='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4}'
M162R='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"repeat_batch":true}'
gen() { # dir model steps lr warmup reuse
  ( cd frontend && python3 -m kvmfe.demo --out $1 --model "$2" --train-steps $3 --window 1 --serve-every 0 --game-frames 0 --tiling a100 \
      --lr $4 --warmup $5 --reuse $6 --max-bundles 8 ) 2>&1 | tail -3
}
run() { # demo out [extra kvm_run args]
  d=$1; o=$2; shift 2
  timeout 3000 ./demo/kvm_run $d --out $o --observe '^loss\.' --compile-cmd "python3 runtime/compile_workload.py" "$@" > $o.log 2>&1; echo "kvm_run exit $?"; tail -5 $o.log
}
echo "== A: step-1 gradients (1 step, reuse none, lr 3e-4 warmup 2)"
gen /work/g1 "$M162" 1 3e-4 2 none
DUMP=$(python3 - <<PY
import json, re
s = json.load(open('/work/g1/workloads/train.json'))
g = sorted({t['args']['g'] for gr in s['graphs'] for t in gr['tasks'] if t['op'] in ('adamw', 'adamw_lr')})
print('^(' + '|'.join(re.escape(x) for x in g) + r'|loss\.0|clip_coef|m\..*|w\..*)\$')
PY
)
mkdir -p /work/g1out; run /work/g1 /work/g1out --dump "$DUMP"
cp /work/g1out.log $OUT/A_kvm_run.log; cp /work/g1out/observations.jsonl $OUT/A_observations.jsonl; cp /work/g1out/run.json $OUT/A_run.json 2>/dev/null
python3 tests/gpt162/torch_ref.py /work/g1 --model "$M162" --steps 1 --lr 3e-4 --warmup 2 --grads /work/g1out/buffers --out $OUT/A_grads.json 2>&1 | tee $OUT/A_grads.log
rm -rf /work/g1out/buffers
echo "== B1: chain stream (preset: lr 3e-4, warmup 2), $N1 steps"
gen /work/c1 "$M162" $N1 3e-4 2 shape
mkdir -p /work/c1out; run /work/c1 /work/c1out
cp /work/c1out.log $OUT/B1_kvm_run.log; cp /work/c1out/observations.jsonl $OUT/B1_observations.jsonl
python3 tests/gpt162/torch_ref.py /work/c1 --model "$M162" --steps $N1 --lr 3e-4 --warmup 2 --curve --obs /work/c1out/observations.jsonl --out $OUT/B1_curve.json 2>&1 | tee $OUT/B1_curve.log
echo "== B2: one repeated batch, lr 3e-4 warmup 2, $N2 steps"
gen /work/o1 "$M162R" $N2 3e-4 2 shape
mkdir -p /work/o1out; run /work/o1 /work/o1out
cp /work/o1out.log $OUT/B2_kvm_run.log; cp /work/o1out/observations.jsonl $OUT/B2_observations.jsonl
python3 tests/gpt162/torch_ref.py /work/o1 --model "$M162R" --steps $N2 --lr 3e-4 --warmup 2 --curve --obs /work/o1out/observations.jsonl --out $OUT/B2_curve.json 2>&1 | tee $OUT/B2_curve.log
echo "== B3: one repeated batch, lr 1e-3 warmup 5, $N2 steps"
gen /work/o2 "$M162R" $N2 1e-3 5 shape
mkdir -p /work/o2out; run /work/o2 /work/o2out
cp /work/o2out.log $OUT/B3_kvm_run.log; cp /work/o2out/observations.jsonl $OUT/B3_observations.jsonl
python3 tests/gpt162/torch_ref.py /work/o2 --model "$M162R" --steps $N2 --lr 1e-3 --warmup 5 --curve --obs /work/o2out/observations.jsonl --out $OUT/B3_curve.json 2>&1 | tee $OUT/B3_curve.log
echo "== done"
'''


# learn: the executor and torch on a low-entropy noisy-bigram stream (tests/gpt162/learning.sh; no shared-box lock on a private GPU), then the plot.
LEARN = r'''
cp /logs/BIN/kvm_run demo/kvm_run
OUT=/logs/$TAG; mkdir -p $OUT
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"data_vocab":'$K',"data_noise":'$NOISE'}'
echo "$M" > $OUT/model.json
KVM_NO_LOCK=1 KVM_SCRATCH=/work/scr bash tests/gpt162/learning.sh $OUT/run "$M" $STEPS $LR $WARMUP 8
pip list 2>/dev/null | grep -i matplotlib || pip install -q matplotlib
python3 tests/gpt162/learning_plot.py $OUT/run --model-json "$M" --png $OUT/run/loss.png --csv $OUT/run/loss.csv
echo "== done"
'''


def _drive(script, tag, extra_env, logname):
    import subprocess, time
    out = f"/logs/{tag}"; os.makedirs(out, exist_ok=True)
    f = open(f"{out}/{logname}", "w")
    env = dict(os.environ, TAG=tag, **extra_env)
    p = subprocess.Popen(["bash", "-c", "cd /src && export PATH=$PATH:/usr/local/cuda/bin && mkdir -p /work && " + script],
                         stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    while p.poll() is None:   # live log: commit the volume every 30 s
        time.sleep(30); f.flush(); vol.commit()
    f.close(); vol.commit()


@app.function(cpu=4, memory=16384, timeout=7200, volumes={"/logs": vol})
def build(tag: str):
    _drive(BUILD, tag, {}, "build.log")


@app.function(gpu="A100-80GB", cpu=8, memory=65536, timeout=10800, volumes={"/logs": vol})
def check(tag: str, n1: int, n2: int):
    _drive(CHECK, tag, {"N1": str(n1), "N2": str(n2)}, "check.log")


@app.function(gpu="A100-80GB", cpu=8, memory=65536, timeout=10800, volumes={"/logs": vol})
def learn(tag: str, k: int, noise: float, steps: int, lr: str, warmup: int):
    _drive(LEARN, tag, {"K": str(k), "NOISE": str(noise), "STEPS": str(steps), "LR": lr, "WARMUP": str(warmup)}, "learn.log")


@app.local_entrypoint()
def main(stage: str = "check", tag: str = "run", n1: int = 40, n2: int = 50, k: int = 256, noise: float = 0.1, steps: int = 300,
         lr: str = "3e-4", warmup: int = 5):
    if stage == "build":
        build.remote(tag)
    elif stage == "learn":
        learn.remote(tag, k, noise, steps, lr, warmup)
    else:
        check.remote(tag, n1, n2)
