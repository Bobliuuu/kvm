"""Build and run GEMM tile benchmark variants on a Modal A100 (kvm-gemm app). No hosts/keys.

  modal run --detach tests/modal/run_gemm_bench.py --tag NAME --variants 'v1=-DKVM_MMA_BK=64 -DKVM_MMA_NST=3 -DKVM_GPT_MMA_SMEM=147456;v2=...' [--prog bench_mma_bf16|test_bf16] [--env 'MMA_SET=gpt']
  modal volume get kvm-gemm /NAME results/<dir>/raw
"""
import os
import pathlib

import modal

_p = pathlib.Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) > 2 else pathlib.Path('/')
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .add_local_dir(str(ROOT), "/src", copy=True,
                        ignore=[".tmp", "results", "logs", ".git", ".claude", "docs", "assets", "**/__pycache__", "doom", "out", "dist", "demo"]))
app = modal.App("kvm-gemm", image=image)
vol = modal.Volume.from_name("kvm-gemm", create_if_missing=True)


@app.function(gpu="A100-40GB", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/logs": vol})
def run(tag: str, variants: str, prog: str, env: str, extra: str):
    import subprocess
    out = f"/logs/{tag}"
    os.makedirs(out, exist_ok=True)
    d = "/src/tests/gpu_ops"
    base = "nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. -diag-suppress 177"
    nv = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1"], capture_output=True, text=True).stdout
    open(f"{out}/env.txt", "w").write(nv)
    vs = [v.split("=", 1) for v in variants.split(";") if v.strip()]
    procs = []
    for name, flags in vs:   # compile in parallel
        procs.append((name, subprocess.Popen(["bash", "-c", f"cd {d} && {base} {flags} {prog}.cu -lcublas -o /tmp/{prog}_{name} 2> /tmp/{name}.err"])))
    for name, p in procs:
        rc = p.wait()
        if rc:
            open(f"{out}/{prog}_{name}.log", "w").write("BUILD FAIL\n" + open(f"/tmp/{name}.err").read()[-4000:])
    for name, _ in vs:
        if not os.path.exists(f"/tmp/{prog}_{name}"):
            continue
        r = subprocess.run(["bash", "-c", f"cd {d} && {env} /tmp/{prog}_{name} {extra}"], capture_output=True, text=True)
        open(f"{out}/{prog}_{name}.log", "w").write(r.stdout + r.stderr + f"\nexit {r.returncode}\n")
        vol.commit()
    vol.commit()


@app.local_entrypoint()
def main(tag: str, variants: str, prog: str = "bench_mma_bf16", env: str = "MMA_SET=gpt", extra: str = ""):
    run.spawn(tag, variants, prog, env, extra)
