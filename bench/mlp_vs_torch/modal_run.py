"""Run the kvm.cu measurement session (bench/mlp_vs_torch/session.sh) on a Modal A100-40GB: kvm.cu, both VM programs and every torch variant
in ONE container, so ratios share one GPU state. No ssh box, no GPU lock.

  modal run --detach bench/mlp_vs_torch/modal_run.py --tag NAME [--runs "1 8 32"] [--nvcc "-DKSTAGE"] [--torch-args "--variants X,Y"] [--rounds 3 --reps 5]
  modal volume get kvm-suite /NAME results/<dir>          # fetch (the log is also live in the volume at /NAME/session.log)
"""
import os
import pathlib

import modal

_p = pathlib.Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) > 2 else pathlib.Path('/')
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("rsync", "git")
         .pip_install("numpy", "torch==2.7.0")
         .add_local_dir(str(ROOT), "/src", copy=True,
                        ignore=[".tmp", "results", "logs", ".git", ".claude", "docs", "assets", "**/__pycache__", "doom", "dist", "demo", "out", "frontend", "tests", "integration"]))
app = modal.App("kvm-mlp-bench", image=image)
vol = modal.Volume.from_name("kvm-suite", create_if_missing=True)

SCRIPT = r'''
set -u
export PATH=$PATH:/usr/local/cuda/bin
cd /src
FLAGS="-DKTILE=32 -DKKC=64 -DKSTG=4 $NVCC"
echo "# tag=$TAG $(date -u +%FT%TZ)"
nvcc -O3 -arch=sm_80 -std=c++17 $FLAGS kvm.cu -o kvm 2>&1 | grep -i "error"
nvcc -O3 -arch=sm_80 -std=c++17 $FLAGS -shared -Xcompiler -fPIC -DKVM_LIB kvm.cu -o kvm_lib.so 2>&1 | grep -i "error"
echo "commit: $COMMIT" > COMMIT
KVM_FLAGS="-O3 -arch=sm_80 -std=c++17 $FLAGS" bash bench/mlp_vs_torch/session.sh /logs/$TAG/out $RUNS
'''


@app.function(gpu="A100-40GB", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/logs": vol})
def run(tag: str, runs: str, nvcc: str, torch_args: str, rounds: int, reps: int, commit: str):
    import subprocess
    import time
    os.makedirs(f"/logs/{tag}", exist_ok=True)
    f = open(f"/logs/{tag}/session.log", "w")
    env = dict(os.environ, TAG=tag, RUNS=runs, NVCC=nvcc, TORCH_ARGS=torch_args, ROUNDS=str(rounds), REPS=str(reps), COMMIT=commit)
    p = subprocess.Popen(["bash", "-c", SCRIPT], stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    while p.poll() is None:
        time.sleep(30)
        f.flush()
        vol.commit()
    f.close()
    vol.commit()
    return p.returncode


@app.local_entrypoint()
def main(tag: str, runs: str = "1 8 32", nvcc: str = "", torch_args: str = "", rounds: int = 3, reps: int = 5):
    import subprocess
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("container exit code", run.remote(tag, runs, nvcc, torch_args, rounds, reps, commit))
