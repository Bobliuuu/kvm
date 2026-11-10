"""Run a KVM GPU suite on a Modal A100 (no ssh box, no GPU lock, no shared disk). Nothing here names a host or a key.

  modal run --detach tests/modal/run_suite.py --suite exec --tag NAME            # run from the checkout under test (its tree is uploaded)
  suites: exec | litmus | negative | negative-park | sanitize | gpt | generic | custom (--cmd 'bash ...')
  modal volume get kvm-suite /NAME results/<dir>/raw                             # fetch logs

Mirrors run_exec_tests.sh: same build.sh / test_exec / litmus scripts, on a Modal container with a 96 GB host (the full-ops ptxas link
needs ~35 GB and is OOM-killed on the shared box). The GPU is A100-40GB (sm_80, 108 SMs), the same part as the lab box.
"""
import os
import pathlib

import modal

_p = pathlib.Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) > 2 else pathlib.Path('/')
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("rsync", "git")
         .pip_install("numpy")
         .add_local_dir(str(ROOT), "/src", copy=True,
                        ignore=[".tmp", "results", "logs", ".git", ".claude", "docs", "assets", "**/__pycache__", "doom/third_party", "doom/build", "out"]))
app = modal.App("kvm-suite", image=image)
vol = modal.Volume.from_name("kvm-suite", create_if_missing=True)

HEAD = r'''
set -u
export PATH=$PATH:/usr/local/cuda/bin KVM_ARCH=${KVM_ARCH:-sm_80} KVM_NO_LOCK=1
echo "# suite=$SUITE tag=$TAG $(date -u +%FT%TZ)"; nvcc --version | tail -2
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
echo "host RAM: $(free -g | awk '/Mem:/{print $2}') GB, cores: $(nproc)"
'''
BUILD = 'echo "BUILD: tests/gpu/build.sh $FLAGS"; time bash tests/gpu/build.sh $FLAGS 2>&1 | grep -v "warning #177\\|TILE\\|\\^\\|^$\\|Remark"; ls -la test_exec\n'
SUITES = {
    "exec": 'python3 tests/gpu/make_initgen_vectors.py --check && echo "initgen vectors identical"\nFLAGS=""\n' + BUILD + 'timeout 2400 ./test_exec; echo "exit: $?"',
    "negative": 'FLAGS="-DKVM_BREAK_FENCE"\n' + BUILD + 'timeout 2400 ./test_exec --expect-fail versions; echo "exit: $?"',
    "negative-park": 'FLAGS="-DKVM_BREAK_PARK_RECHECK"\n' + BUILD + 'timeout 2400 ./test_exec --expect-fail litmus; echo "exit: $?"',
    "sanitize": 'FLAGS=""\n' + BUILD + 'for t in memcheck racecheck synccheck; do echo "== compute-sanitizer --tool $t smoke"; timeout 1800 compute-sanitizer --tool $t --error-exitcode 9 ./test_exec smoke 2>&1 | tail -40; echo "sanitizer exit: ${PIPESTATUS[0]}"; done',
    "litmus": 'bash tests/litmus/run_litmus.sh; echo "exit: $?"',
    "generic": 'bash tests/generic/run_box.sh /logs/$TAG/generic; echo "exit: $?"',
    "gpt": 'FLAGS=""\n' + BUILD + 'bash tests/gpu/run_gpt.sh; echo "exit: $?"',
}


@app.function(gpu="A100-40GB", cpu=16, memory=98304, timeout=4 * 3600, volumes={"/logs": vol})
def run_suite(tag: str, name: str, cmd: str):
    import subprocess
    import time
    body = cmd if name == "custom" else SUITES[name]
    out = f"/logs/{tag}"
    os.makedirs(out, exist_ok=True)
    f = open(f"{out}/{name}.log", "w")
    env = dict(os.environ, SUITE=name, TAG=tag)
    p = subprocess.Popen(["bash", "-c", "cd /src && " + HEAD + body], stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    while p.poll() is None:   # live log: commit the volume every 30 s so a crash keeps the output
        time.sleep(30)
        f.flush()
        vol.commit()
    f.close()
    vol.commit()
    return p.returncode


@app.local_entrypoint()
def main(suite: str = "exec", tag: str = "run", cmd: str = ""):
    if suite not in SUITES and suite != "custom":
        raise SystemExit(f"unknown suite {suite}; one of {sorted(SUITES)} or custom")
    print(f"suite {suite} -> volume kvm-suite:/{tag}/{suite}.log")
    print("container exit code", run_suite.remote(tag, suite, cmd))
