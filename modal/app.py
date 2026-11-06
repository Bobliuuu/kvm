"""Modal runner for the admission study. Detached only: modal run --detach modal/app.py::main --job JOBNAME
Job scripts live in results/2026-11-06-admission/jobs/<job>.sh and run on an A100-40GB with the repo copied to /src."""
import json, subprocess, sys, modal
app = modal.App("kvm-admit")
vol = modal.Volume.from_name("kvm-admit-results", create_if_missing=True)
img = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
       .add_local_dir(".", "/src", ignore=["**/.git", "**/results/20*/**/*.csv", "**/dist", "**/doom", "**/demo"], copy=True))

@app.function(gpu="A100-40GB", image=img, volumes={"/out": vol}, timeout=3 * 3600, cpu=16, memory=65536)
def run(job: str):
    import os, shutil, time
    t0 = time.time()
    env = dict(os.environ, OUT=f"/out/{job}")
    os.makedirs(env["OUT"], exist_ok=True)
    print(json.dumps({"event": "start", "job": job, "gpu": subprocess.run("nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()}), flush=True)
    p = subprocess.run(["bash", f"/src/results/2026-11-06-admission/jobs/{job}.sh"], cwd="/src", env=env, capture_output=True, text=True)
    open(f"{env['OUT']}/stdout.txt", "w").write(p.stdout + "\n--- stderr\n" + p.stderr)
    print(p.stdout[-6000:], flush=True); print(p.stderr[-2000:], flush=True)
    print(json.dumps({"event": "done", "job": job, "rc": p.returncode, "secs": round(time.time() - t0)}), flush=True)
    vol.commit()

@app.local_entrypoint()
def main(job: str):
    run.spawn(job)
    print("spawned", job)
