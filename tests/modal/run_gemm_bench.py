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
    vs = []   # variant syntax: label=FLAGS|ENV  (binaries are shared between labels with the same FLAGS)
    for v in variants.split(";"):
        if not v.strip():
            continue
        label, rest = v.split("=", 1)
        flags, _, venv = rest.partition("|")
        vs.append((label.strip(), flags.strip(), venv.strip()))
    bins = {}
    for _, flags, _ in vs:
        if flags not in bins:
            bins[flags] = f"{prog}_{len(bins)}"
    procs = []
    for flags, b in bins.items():   # compile in parallel
        procs.append((flags, b, subprocess.Popen(["bash", "-c", f"cd {d} && {base} {flags} {prog}.cu -lcublas -o /tmp/{b} 2> /tmp/{b}.err"])))
    for flags, b, p in procs:
        if p.wait():
            open(f"{out}/{prog}_{b}.builderr", "w").write(flags + "\n" + open(f"/tmp/{b}.err").read()[-4000:])
    for label, flags, venv in vs:
        b = bins[flags]
        if not os.path.exists(f"/tmp/{b}"):
            continue
        r = subprocess.run(["bash", "-c", f"cd {d} && {env} {venv} /tmp/{b} {extra}"], capture_output=True, text=True)
        open(f"{out}/{prog}_{label}.log", "w").write(f"# flags: {flags}  env: {env} {venv}\n" + r.stdout + r.stderr + f"\nexit {r.returncode}\n")
        vol.commit()
    vol.commit()


@app.local_entrypoint()
def main(tag: str, variants: str, prog: str = "bench_mma_bf16", env: str = "MMA_SET=gpt BENCH_REPS=11", extra: str = ""):
    run.spawn(tag, variants, prog, env, extra)
