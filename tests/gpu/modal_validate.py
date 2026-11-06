"""Validate the integrated tree on a private Modal A100-40GB (no shared GPU, so bandwidth/latency numbers are clean).
Run:  modal run --detach tests/gpu/modal_validate.py            (never plain `modal run`: it dies with the laptop)
Each step's output is printed as one JSON line to the app logs and written to the Modal Volume `kvm-validate`."""
import json, os, subprocess, time, pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
img = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
       .add_local_dir(str(ROOT / "include"), "/w/include")
       .add_local_dir(str(ROOT / "tests/gpu"), "/w/tests/gpu")
       .add_local_dir(str(ROOT / "runtime"), "/w/runtime", ignore=["__pycache__"])
       .add_local_file(str(ROOT / "kvm_exec.cu"), "/w/kvm_exec.cu")
       .add_local_file(str(ROOT / "kvm.cu"), "/w/kvm.cu"))
app = modal.App("kvm-validate")
vol = modal.Volume.from_name("kvm-validate", create_if_missing=True)
NV = "nvcc -O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -Iinclude -Itests/gpu"
EXTRA = "-DKVM_EXEC_NO_MAIN -DKVM_EXTRA_OPS_DEF='\"kvm_ops_test.def\"' -DKVM_EXTRA_OPS_CUH='\"kvm_ops_test.cuh\"'"
BUILDS = {
    "test_sched": f"{NV} {EXTRA} tests/gpu/test_sched.cu -o test_sched",
    "test_exec": f"{NV} {EXTRA} tests/gpu/test_exec.cu -o test_exec",
    "kvm": "nvcc -O3 -arch=sm_80 -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4 kvm.cu -o kvm",
}
STEPS = [  # (name, command, timeout s)
    ("sched_functional", "./test_sched --quiet cost latfirst share reserve edf inherit rand", 1500),
    ("sched_bench", "./test_sched --quiet calib dispatch arbiter rebalance baselines overload affinity", 2400),
    ("sched_mix", "./test_sched --quiet mix", 1500),
    ("exec_suite", "./test_exec --quiet", 1800),
    ("kvm_selftest", "./kvm --selftest", 120),
    ("kvm_r8", "./kvm --runs 8 --steps 300 --reps 7", 600),
    ("kvm_fault_claim", "./kvm --runs 1 --steps 10 --reps 1 --modes claim --watchdog-ms 300 --inject-fault", 120),
    ("kvm_fault_queue", "./kvm --runs 1 --steps 10 --reps 1 --modes queue --watchdog-ms 300 --inject-fault", 120),
]

def sh(cmd, t):
    t0 = time.time()
    p = subprocess.run(["bash", "-c", cmd], cwd="/w", capture_output=True, text=True, timeout=t)
    return {"cmd": cmd, "rc": p.returncode, "sec": round(time.time() - t0, 1), "out": p.stdout[-60000:], "err": p.stderr[-4000:]}

@app.function(gpu="A100-40GB", image=img, timeout=3 * 3600, volumes={"/vol": vol})
def run():
    os.makedirs("/vol/out", exist_ok=True)
    def emit(name, r):
        r["step"] = name
        print("RESULT " + json.dumps(r), flush=True)
        pathlib.Path(f"/vol/out/{name}.json").write_text(json.dumps(r, indent=1)); vol.commit()
    emit("env", sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -2", 60))
    for n, c in BUILDS.items():
        emit("build_" + n, sh(c + " 2>&1 | grep -v 'warning\\|^ *\\^\\|Remark\\|^$' ; exit ${PIPESTATUS[0]}", 1800))
    for n, c, t in STEPS:
        try: emit(n, sh(c, t))
        except subprocess.TimeoutExpired: emit(n, {"cmd": c, "rc": -9, "out": "TIMEOUT", "err": ""})

@app.local_entrypoint()
def main():
    run.remote()
