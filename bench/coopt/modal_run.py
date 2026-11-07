"""Modal A100 driver for the KVM successive-halving co-optimization bench. Own app name; detached run.
  modal run --detach bench/coopt/modal_run.py --tag NAME [--args '--runs 32 --steps 300 --E 75 --K 3']
Results land in volume kvm-coopt:/<tag>/ (build log, run log, torch_result.json, dumps)."""
import os, pathlib
import modal

HERE = pathlib.Path(__file__).resolve().parent
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("numpy", "torch==2.5.1")
         .add_local_dir(str(HERE), "/src", copy=True, ignore=["**/__pycache__", "out"]))
app = modal.App("kvm-coopt", image=image)
vol = modal.Volume.from_name("kvm-coopt", create_if_missing=True)

@app.function(gpu="A100-40GB", cpu=8, memory=32768, timeout=3600, volumes={"/logs": vol})
def run(tag: str, args: str, flags: str):
    import subprocess
    out = f"/logs/{tag}"; os.makedirs(out, exist_ok=True)
    cmd = f"""set -x; export PATH=$PATH:/usr/local/cuda/bin; cd /src
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc -O3 -arch=sm_80 -std=c++17 {flags} coopt_sh.cu -o coopt_sh 2>&1 | grep -v 'warning\\|Remark\\|^$\\|\\^' 
./coopt_sh {args} --out {out}/dump 2>&1 | tee {out}/kvm.log
python3 -I torch_sh.py {out}/dump 2>&1 | tee {out}/torch.log"""
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    open(f"{out}/driver.log", "w").write(p.stdout + "\n" + p.stderr)
    vol.commit()
    return p.stdout[-6000:] + p.stderr[-3000:]

@app.local_entrypoint()
def main(tag: str = "t0", args: str = "--runs 32 --steps 300 --E 75 --K 3", flags: str = "-DKTILE=32 -DKKC=64 -DKSTG=4"):
    print(run.remote(tag, args, flags))
