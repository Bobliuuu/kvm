"""Class-aware admission on a Modal A100 (use when the shared GPU box is busy). Results go to the Modal volume kvm-admit2.
  modal run --detach tests/modal/run_latlane.py::exec_job
  modal run --detach tests/modal/run_latlane.py::adaptive_job --reps 3
Needs, beside the repo (git-ignored, copied from a box that built them): logs/doomobj/{doom_dev.o,libdoom_cpu.so,doom1.wad}, scn/ and scn_big/
(adaptive/make_scenario.py output, see adaptive/batch.sh).
"""
import os, subprocess, modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCR = os.environ.get("KVM_SCN_DIR", os.path.join(ROOT, "logs", "scn"))
D = os.path.join(ROOT, "logs", "doomobj")
ign = ["**/__pycache__/**", "**/*.pyc"]
image = (modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11").pip_install("numpy")
         .add_local_dir(os.path.join(ROOT, "include"), "/w/include")
         .add_local_dir(os.path.join(ROOT, "tests"), "/w/tests", ignore=ign)
         .add_local_dir(os.path.join(ROOT, "runtime"), "/w/runtime", ignore=ign)
         .add_local_dir(os.path.join(ROOT, "frontend"), "/w/frontend", ignore=ign)
         .add_local_dir(os.path.join(ROOT, "adaptive"), "/w/adaptive", ignore=ign)
         .add_local_dir(os.path.join(ROOT, "doom"), "/w/doom", ignore=ign)
         .add_local_file(os.path.join(D, "doom_dev.o"), "/w/doom/build/doom_dev.o")
         .add_local_file(os.path.join(D, "libdoom_cpu.so"), "/w/doom/build/libdoom_cpu.so")
         .add_local_file(os.path.join(D, "doom1.wad"), "/w/doom/third_party/doom1.wad")
         .add_local_dir(os.path.join(SCR, "scn"), "/w/scn").add_local_dir(os.path.join(SCR, "scn_big"), "/w/scn_big")
         .add_local_file(os.path.join(ROOT, "kvm_exec.cu"), "/w/kvm_exec.cu"))
app = modal.App("kvm-admit2", image=image)
vol = modal.Volume.from_name("kvm-admit2", create_if_missing=True)

def sh(cmd, log):
    with open(log, "a") as f:
        f.write(f"$ {cmd}\n"); f.flush()
        p = subprocess.run(cmd, shell=True, cwd="/w", stdout=f, stderr=subprocess.STDOUT, executable="/bin/bash")
        f.write(f"[exit {p.returncode}]\n")
    return p.returncode

HDR = "nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -2; git --version >/dev/null; echo commit ${KVM_COMMIT}"

@app.function(gpu="A100-40GB", timeout=4 * 3600, cpu=16, volumes={"/out": vol})
def exec_job(commit: str = ""):
    os.makedirs("/out/exec", exist_ok=True); L = "/out/exec/exec.log"
    os.environ["KVM_COMMIT"] = commit
    sh(HDR, L)
    sh("bash tests/gpu/build_core.sh test_exec_core && ls -la test_exec_core", L)
    sh("bash tests/gpu/build_core.sh test_exec_mut -DKVM_BREAK_LAT_SLICE", L)
    sh("./test_exec_core latlane 2>&1 | grep -vE 'ring wrap|ring full'", L)
    sh("./test_exec_core 2>&1 | grep -vE 'ring wrap|ring full|^\\[kvm_exec\\]' | tail -80", L)
    sh("./test_exec_mut --expect-fail latlane 2>&1 | grep -vE 'ring wrap|ring full|^\\[kvm_exec\\]' | tail -30", L)
    for t in ["memcheck", "racecheck", "synccheck"]:
        sh(f"compute-sanitizer --tool {t} --error-exitcode 9 ./test_exec_core smoke latlane 2>&1 | grep -vE 'ring wrap|ring full|^\\[kvm_exec\\]' | tail -30", L)
    vol.commit()

@app.function(gpu="A100-40GB", timeout=8 * 3600, cpu=16, volumes={"/out": vol})
def adaptive_job(reps: int = 3, dur: int = 24, commit: str = "", groups: str = "lat lat_big"):
    os.makedirs("/out/adaptive", exist_ok=True); L = "/out/adaptive/adaptive.log"
    os.environ["KVM_COMMIT"] = commit
    sh(HDR, L)
    # the two builds are independent
    sh("(bash adaptive/build.sh > build_adaptive.log 2>&1 &); bash tests/gpu/build_adaptive.sh > build_test.log 2>&1; while [ ! -x adaptive_run ]; do sleep 10; done; sleep 20; ls -la adaptive_run test_adaptive; grep -c error build_adaptive.log build_test.log", L)
    sh("./test_adaptive admit_gran 2>&1 | grep -vE '^\\[kvm_exec\\]' | tee /out/adaptive/admit_gran.txt", L)
    vol.commit()
    for g in groups.split():
        sh(f"mkdir -p /out/adaptive/runs-{g}; env OUT=/out/adaptive/runs-{g} REPS={reps} DUR={dur} BATCH_COMMIT={commit} bash adaptive/batch.sh {g} > /out/adaptive/batch-{g}.log 2>&1", L)
        vol.commit()
