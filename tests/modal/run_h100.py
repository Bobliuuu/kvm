"""sm_90a (H100) builds and tests of the Hopper TMA + wgmma tiles on Modal. Nothing here names a host or a key.

  modal run --detach tests/modal/run_h100.py --stage build --tag NAME     # CPU-only container (no GPU held for the long ptxas links)
  modal run --detach tests/modal/run_h100.py --stage test  --tag NAME     # H100: test_hopper, test_ops, test_bf16, test_extra, test_exec, sanitizers
  modal run --detach tests/modal/run_h100.py --stage bench --tag NAME     # H100: test_hopper --bench
  modal run --detach tests/modal/run_h100.py --stage custom --tag NAME --cmd 'bash ...'   # H100, binaries in bin_h100/, logs to OUT
  modal run --detach tests/modal/run_h100.py --stage cpu --tag NAME --cmd 'bash ...'      # the same on the CPU-only build container
  modal volume get kvm-h100-logs /NAME results/<dir>/raw

Wraps tests/gpu_ops/run_hopper_box.sh (the same script runs on any sm_90 box); binaries live in the volume under /NAME/bin_h100.
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
                        ignore=[".tmp", "results", "logs", ".git", ".claude", "docs", "assets", "**/__pycache__", "doom/third_party", "doom/build",
                                "out", "bin_h100"]))
app = modal.App("kvm-h100", image=image)
vol = modal.Volume.from_name("kvm-h100-logs", create_if_missing=True)


def _run(tag: str, body: str, logname: str):
    import subprocess
    import time
    out = f"/logs/{tag}"
    os.makedirs(out, exist_ok=True)
    f = open(f"{out}/{logname}.log", "w")
    cmd = (f"cd /src && export PATH=$PATH:/usr/local/cuda/bin && mkdir -p {out}/bin_h100 && rm -rf bin_h100 && ln -s {out}/bin_h100 bin_h100 && "
           + body)
    p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, OUT=f"{out}/{logname}"))
    while p.poll() is None:   # live log: commit the volume every 30 s
        time.sleep(30)
        f.flush()
        vol.commit()
    f.close()
    vol.commit()
    return p.returncode


@app.function(cpu=32, memory=131072, timeout=4 * 3600, volumes={"/logs": vol})
def build(tag: str, cmd: str = ""):
    return _run(tag, cmd or "bash tests/gpu_ops/run_hopper_box.sh build $OUT", "cpu" if cmd else "build")


@app.function(gpu="H100!", cpu=8, memory=65536, timeout=3 * 3600, volumes={"/logs": vol})
def gpu(tag: str, stage: str, cmd: str):
    body = cmd if stage == "custom" else f"bash tests/gpu_ops/run_hopper_box.sh {stage} $OUT"
    return _run(tag, "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; " + body, stage)


@app.local_entrypoint()
def main(stage: str = "test", tag: str = "run", cmd: str = ""):
    if stage in ("build", "cpu"):   # cpu: a custom command on the CPU-only build container (e.g. rebuild one binary)
        print("exit", build.remote(tag, cmd if stage == "cpu" else ""))
    else:
        print("exit", gpu.remote(tag, stage, cmd))
