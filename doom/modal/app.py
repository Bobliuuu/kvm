"""Modal runner for the Doom split-tick work (own A100-40GB, per-second billing; the shared Lambda box is not used).

  KVM_NINST=64 modal run --detach doom/modal/app.py::main --job verify        (jobs: verify speed instances study prof sanitize all)
Build (CPU only, cached in the image): engine objects for KVM_NINST instances, CPU oracle, doom_exec binaries, profiler build.
Every result is printed as one JSON line prefixed RESULT and written to the Modal Volume kvm-game-results under /<job>-<UTC>/.
"""
import json
import os
import subprocess
import time

import modal

NINST = int(os.environ.get("KVM_NINST", "64"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("kvm-game-parallel")
vol = modal.Volume.from_name("kvm-game-results", create_if_missing=True)


def build_all():
    env = dict(os.environ, NVCC="/usr/local/cuda/bin/nvcc", JOBS=str(os.cpu_count() or 8), LOG_LEVEL="info")
    def sh(cmd, **kw):
        print("+", cmd, flush=True)
        subprocess.run(cmd, shell=True, check=True, cwd="/work", env={**env, **kw.get("env", {})})
    sh("bash doom/setup.sh")
    sh(f"DOOM_INSTANCES={NINST} bash doom/build.sh")
    sh("bash doom/cpu_build.sh")
    objs = " ".join(["doom/build/doom_dev.o"] + [f"doom/build/doom_dev_i{k}.o" for k in range(1, NINST)])
    flags = "-O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off"
    sh("mkdir -p bin")
    sh(f"$NVCC {flags} -DKVM_DOOM_INSTANCES=1 doom/exec_main.cu doom/build/doom_dev.o -o bin/doom_exec_n1 2>&1 | grep -v 'cannot be statically' | grep -i error; true")
    sh(f"$NVCC {flags} -DKVM_DOOM_INSTANCES={NINST} doom/exec_main.cu {objs} -o bin/doom_exec_nN 2>&1 | grep -v 'cannot be statically' | grep -i error; true")
    sh("DOOM_PROF=1 BUILD=/work/doom/build_prof DOOM_INSTANCES=1 bash doom/build.sh")
    sh("$NVCC -O2 -arch=sm_80 -rdc=true -DDOOM_PROF_READ -Iinclude doom/tests/doomtest.cu doom/build_prof/doom_dev.o -o bin/doomprof")
    sh("$NVCC -O2 -arch=sm_80 -rdc=true -Iinclude doom/tests/doomtest.cu doom/build/doom_dev.o -o bin/doomtest")
    sh("ls -l bin; ls doom/build | head -3")


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("clang", "llvm", "git", "curl", "rsync", "build-essential", "procps")
    .add_local_dir(ROOT, "/work", copy=True, ignore=[".git", "results", "logs", "doom/build", "doom/build_*", "doom/third_party", "dist", "demo"])
    .run_function(build_all, cpu=16, memory=32768, timeout=7200)
)


@app.function(image=image, gpu="A100-40GB", timeout=3 * 3600, volumes={"/vol": vol})
def run_job(job: str, args: str = ""):
    out = f"/vol/{job}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    os.makedirs(out, exist_ok=True)
    smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
    print("RESULT", json.dumps({"job": job, "gpu": smi, "ninst": NINST, "out": out}), flush=True)
    r = subprocess.run(["python3", "doom/modal/harness.py", job, out, str(NINST), args], cwd="/work", env=dict(os.environ, NVCC="/usr/local/cuda/bin/nvcc"))
    vol.commit()
    return r.returncode


@app.local_entrypoint()
def main(job: str = "verify", args: str = ""):
    print("exit", run_job.remote(job, args))
