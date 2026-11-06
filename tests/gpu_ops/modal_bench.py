"""Run the op benchmarks/tests on a Modal A100 (use when the shared GPU box is busy).
  modal run --detach tests/gpu_ops/modal_bench.py --what mm [--nvflags "-DMM_MINB=2"] [--out logs/x.txt]
what: mm (bench_mm.cu), bench (bench_ops.cu with cuBLAS), test_ops, test_extra, exec (test_exec), gpt (gpt_tiny bundle).
"""
import os, subprocess, modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
image = (modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
         .add_local_dir(os.path.join(ROOT, "include"), "/w/include")
         .add_local_dir(os.path.join(ROOT, "tests"), "/w/tests", ignore=["**/__pycache__/**", "**/*.pyc"])
         .add_local_dir(os.path.join(ROOT, "runtime"), "/w/runtime", ignore=["**/__pycache__/**"])
         .add_local_dir(os.path.join(ROOT, "frontend"), "/w/frontend", ignore=["**/__pycache__/**"])
         .add_local_file(os.path.join(ROOT, "kvm_exec.cu"), "/w/kvm_exec.cu"))
app = modal.App("kvm-perf2-matmul", image=image)

def sh(cmd):
    p = subprocess.run(cmd, shell=True, cwd="/w/tests/gpu_ops", capture_output=True, text=True)
    return p.stdout + p.stderr

@app.function(gpu="A100-40GB", timeout=3600, cpu=8)
def run(what: str, nvflags: str, wrap: str, env: str):
    base = "nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I."
    out = sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -2")
    if what == "mm":
        out += sh(f"{base} -Xptxas -v {nvflags} bench_mm.cu -lcublas -o t 2>&1 | grep -E 'error|spill|registers' | sort | uniq -c; {env} ./t")
    elif what == "bench":
        out += sh(f"{base} -DBENCH_CUBLAS {nvflags} bench_ops.cu -lcublas -o t 2>&1 | grep -E 'error|spill'; {env} ./t")
    elif what == "test_ops":
        out += sh(f"{base} {nvflags} test_ops.cu -o t 2>&1 | grep error; {wrap} ./t | tail -8")
    elif what == "test_extra":
        out += sh(f"nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo --fmad=false -Xcompiler -ffp-contract=off -I../../include -I. {nvflags} test_extra.cu -o t 2>&1 | grep error; {wrap} ./t | tail -8")
    elif what == "exec":   # executor suite (tests/gpu/test_exec.cu) with the kernel's launch bounds from nvflags (e.g. -DKVM_MIN_BLOCKS=2)
        out += sh(f"cd /w && bash tests/gpu/build.sh {nvflags} 2>&1 | grep -E 'error|warning: .*spill' | head; {env} timeout 3000 ./test_exec 2>&1 | tail -25")
    elif what == "gpt":    # frontend gpt_tiny bundle through the CLI: 544 tasks, every scheduler mode, buffers must be byte-identical
        out += sh(f"cd /w && pip install -q numpy >/dev/null 2>&1; KVM_NO_LOCK=1 bash tests/gpu/run_gpt.sh 2>&1 | tail -25")
    return out

@app.local_entrypoint()
def main(what: str = "mm", nvflags: str = "", wrap: str = "", env: str = "", out: str = ""):
    """nvflags may hold several variants separated by ';' (one container each, in parallel); results are concatenated."""
    variants = nvflags.split(";") if nvflags else [""]
    res = ""
    for v, r in zip(variants, run.starmap([(what, v, wrap, env) for v in variants])):
        res += f"=== VARIANT [{v}]\n{r}\n"
    print(res)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        open(out, "w").write(res)
