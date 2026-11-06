"""Build and run the kvm GPU tests on an H100 (sm_90a) via Modal.
Usage (repo root):  modal run --detach tests/modal/run_h100.py [--stage core|ops|full|all] [--tag NAME]
Logs land in the Modal volume 'kvm-h100-logs' under /<tag>/ ; fetch with: modal volume get kvm-h100-logs /<tag> results/<dir>/logs
"""
import modal, os, pathlib

_p = pathlib.Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) > 2 else pathlib.Path('/')
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("rsync", "git")
         .pip_install("torch", "numpy")
         .add_local_dir(str(ROOT), "/src", copy=True,
                        ignore=[".tmp", "results", "logs", ".git", "docs", "**/__pycache__"]))
app = modal.App("kvm-h100", image=image)
vol = modal.Volume.from_name("kvm-h100-logs", create_if_missing=True)

SCRIPT = {
 "core": r'''
KVM_ARCH=sm_90a bash tests/gpu/build_core.sh test_exec_core 2>&1 | tail -50
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
timeout 1500 ./test_exec_core 2>&1 | tail -60; echo "exit ${PIPESTATUS[0]}"
''',
 "ops": r'''
for t in test_ops test_extra; do
 (cd tests/gpu_ops && nvcc -O2 -arch=sm_90a -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. $t.cu -o $t 2>&1 | grep -v -i 'remark\|^\s*\^\|^$' | tail -40) &
done; wait
cd tests/gpu_ops
for t in test_ops test_extra; do echo "== $t"; timeout 900 ./$t 2>&1 | tail -40; echo "exit ${PIPESTATUS[0]}"; done
''',
 "full": r'''
KVM_ARCH=sm_90a bash tests/gpu/build.sh 2>&1 | tail -50
timeout 3000 ./test_exec 2>&1 | tail -80; echo "exit ${PIPESTATUS[0]}"
''',
}

@app.function(gpu="H100", cpu=8, memory=32768, timeout=5400, volumes={"/logs": vol})
def run(stage: str, tag: str):
    import subprocess, threading, time
    out = f"/logs/{tag}"; os.makedirs(out, exist_ok=True)
    stages = ["core", "ops", "full"] if stage == "all" else [stage]
    for s in stages:
        f = open(f"{out}/{s}.log", "w")
        p = subprocess.Popen(["bash", "-c", "cd /src && export PATH=$PATH:/usr/local/cuda/bin && set -x && " + SCRIPT[s]],
                             stdout=f, stderr=subprocess.STDOUT, text=True)
        while p.poll() is None:   # live log: commit the volume every 30 s
            time.sleep(30); f.flush(); vol.commit()
        f.close(); vol.commit()

BUILD = r'''
mkdir -p /logs/BIN
for t in test_ops test_extra; do
 (cd tests/gpu_ops && nvcc -O2 -arch=sm_90a -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. $t.cu -o /logs/BIN/$t 2>&1 | grep -v -i 'remark\|^\s*\^\|^$' | tail -30) &
done
(KVM_ARCH=sm_90a bash tests/gpu/build.sh 2>&1 | tail -30; cp test_exec /logs/BIN/) &
wait
ls -la /logs/BIN
'''

@app.function(cpu=16, memory=32768, timeout=7200, volumes={"/logs": vol})
def build(tag: str):
    import subprocess, time
    os.makedirs(f"/logs/{tag}", exist_ok=True)
    f = open(f"/logs/{tag}/build.log", "w")
    p = subprocess.Popen(["bash", "-c", "cd /src && export PATH=$PATH:/usr/local/cuda/bin && " + BUILD], stdout=f, stderr=subprocess.STDOUT, text=True)
    while p.poll() is None:
        time.sleep(30); f.flush(); vol.commit()
    f.close(); vol.commit()

@app.function(gpu="H100", cpu=4, timeout=3600, volumes={"/logs": vol})
def runbin(tag: str, which: str = "test_ops,test_extra,test_exec"):
    import subprocess
    os.makedirs(f"/logs/{tag}", exist_ok=True)
    f = open(f"/logs/{tag}/runbin.log", "w")
    cmd = ("cd /src/tests/gpu_ops && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; "
           "for t in $(echo " + which.replace("test_exec","") + " | tr ',' ' '); do echo == $t; timeout 900 /logs/BIN/$t 2>&1 | tail -30; echo exit ${PIPESTATUS[0]}; done; "
           + ("" if "test_exec" not in which else "cd /src && echo == test_exec_full; timeout 2400 /logs/BIN/test_exec 2>&1 | grep -v '^\\[kvm_exec\\]' | tail -60; echo exit ${PIPESTATUS[0]}"))
    subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, text=True)
    f.close(); vol.commit()

@app.local_entrypoint()
def main(stage: str = "core", tag: str = "run"):
    if stage == "build": build.remote(tag)
    elif stage.startswith("runbin"): runbin.remote(tag, stage.split(":")[1] if ":" in stage else "test_ops,test_extra,test_exec")
    else: run.remote(stage, tag)
