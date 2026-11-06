"""KVM_MIN_BLOCKS=2 validation on a Modal A100-40GB: every executor suite, ops packs, GPT bundle, gptdiff, sanitizers, litmus, dispatch bench, step times.
Usage (repo root):  modal run --detach tests/modal/run_minblocks.py [--tag NAME] [--blocks 2]
An orchestrator function (server side, survives the client) compiles on CPU containers into the volume 'kvm-minblocks' and then runs every
suite on its own A100-40GB container; logs land in /<tag>/*.txt on the volume:  modal volume get kvm-minblocks /<tag> results/<dir>/logs
Binaries are built for sm_80 at -O3; kvm_exec.cu takes 40+ min, so all builds run in parallel (one TU each, nvcc is single threaded per TU).
"""
import modal, os, pathlib, subprocess, time, concurrent.futures as cf

ROOT = pathlib.Path(__file__).resolve().parents[2]
image = (modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
         .apt_install("rsync")
         .pip_install("numpy")
         .add_local_dir(str(ROOT), "/src", copy=True,
                        ignore=[".git", "results", "logs", "docs", "out", "**/__pycache__", "**/*.pyc", "kvm_exec_bin", "test_exec"]))
app = modal.App("kvm-minblocks", image=image)
vol = modal.Volume.from_name("kvm-minblocks", create_if_missing=True)
V = "/vol"
NVB = "nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo"


def sh(cmd, log=None, timeout=None):
    p = subprocess.run(["bash", "-c", "set -o pipefail; cd /src && export PATH=$PATH:/usr/local/cuda/bin && " + cmd],
                       capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr + f"\n[exit {p.returncode}]\n"
    return out


@app.function(cpu=8, memory=32768, timeout=6 * 3600, volumes={V: vol})
def build(job: str, n: int) -> str:
    """Compile one binary set. n = KVM_MIN_BLOCKS."""
    F = f"-DKVM_MIN_BLOCKS={n}"
    d = f"{V}/bin/{job}"
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    if job.startswith("test_exec"):
        out = sh(f"KVM_ARCH=sm_80 bash tests/gpu/build.sh {F} -Xptxas -v 2>&1 | grep -E 'error|kvm_exec_kernel|Used .* registers|spill' | sort | uniq -c | sort -rn | head -20; cp test_exec {d}/")
    elif job.startswith("kexec"):
        out = sh(f"{NVB} {F} -diag-suppress 177 -Xptxas -v kvm_exec.cu -o {d}/kvm_exec 2>&1 | grep -E 'error|Used .* registers|spill' | sort | uniq -c | sort -rn | head -20")
    elif job == "litmus":
        out = sh(f"mkdir -p out && KVM_NVFLAGS='{F}' bash tests/litmus/build.sh /src/out 2>&1 | tail -3; "
                 f"B='-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -lineinfo {F}'; "
                 f"(nvcc $B -DKVM_EXEC_NO_MAIN -DKVM_NEW tests/litmus/e2e.cu -o out/e2e 2>&1 | grep -i ' error') & "
                 f"(nvcc $B -DKVM_EXEC_NO_MAIN -DKVM_NEW -DKVM_BREAK_FENCE tests/litmus/e2e.cu -o out/e2e_breakfence 2>&1 | grep -i ' error') & wait; "
                 f"cp -r out/. {d}/; ls {d}")
    elif job == "ops2":
        out = sh(f"cd tests/gpu_ops && (nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. {F} -DBENCH_MINB={n} test_ops.cu -o {d}/test_ops 2>&1 | grep -E 'error') & "
                 f"(nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo --fmad=false -Xcompiler -ffp-contract=off -I../../include -I. {F} -DBENCH_MINB={n} test_extra.cu -o {d}/test_extra 2>&1 | grep error) & wait; ls {d}")
    elif job == "bundles":
        out = sh(f"set -e; mkdir -p {d}; python3 examples/models/gen_bundle.py {d}/tiny --model gpt_tiny --steps 2 | tail -1; "
                 f"python3 examples/models/gen_bundle.py {d}/mid --model '{{\"name\":\"gpt_mid\",\"vocab\":512,\"d_model\":256,\"layers\":4,\"heads\":8,\"seq\":128,\"batch\":4}}' --steps 2 --tiling a100 | tail -1; "
                 f"head -c 1500 {d}/mid/meta.json | grep -E 'tasks|edges|layers|d_model'; grep -E '\"tasks\"' {d}/tiny/meta.json")
    out += f"\n[build {job} MIN_BLOCKS={n}: {time.time() - t0:.0f} s]\n"
    vol.commit()
    return out


HDR = ("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvidia-smi -q | grep -m1 'Product Name'; "
       "nvcc --version | tail -2; python3 - <<'E'\nimport subprocess\nprint('SMs:', subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())\nE\n")


def prep(*jobs):
    vol.reload()
    return "; ".join(f"cp -r {V}/bin/{j} /src/bin_{j}" for j in jobs)


@app.function(gpu="A100-40GB", cpu=4, memory=16384, timeout=5 * 3600, volumes={V: vol})
def run(name: str, n: int) -> str:
    """One suite group on one GPU container. n = MIN_BLOCKS of the binaries under test."""
    vol.reload()
    out = sh(HDR)
    B = f"{V}/bin"
    if name == "exec_full":      # all default suites (everything but bench)
        out += sh(f"cp {B}/test_exec{n}/test_exec . && timeout 11000 ./test_exec > full.log 2>&1; echo test_exec exit $?; grep -E '^== |checks,|^PASS|^FAIL|FAIL' full.log | grep -v '^\\[kvm_exec\\]' | tail -60; tail -5 full.log")
    elif name == "exec_sanitize":
        for tool in ("memcheck", "racecheck", "synccheck"):
            out += f"== compute-sanitizer --tool {tool} test_exec smoke\n"
            out += sh(f"cp {B}/test_exec{n}/test_exec . && timeout 3000 compute-sanitizer --tool {tool} --error-exitcode 9 ./test_exec smoke 2>&1 | grep -v '^\\[kvm_exec\\]' | tail -40; echo sanitizer exit ${{PIPESTATUS[0]}}")
    elif name == "exec_bench":   # executor dispatch at 1 vs 2 blocks/SM, alternating, 3 rounds
        for rnd in range(3):
            for k in (1, 2):
                out += f"== round {rnd} MIN_BLOCKS={k}\n" + sh(f"cp {B}/test_exec{k}/test_exec ./te{k} && ./te{k} bench 2>&1 | grep -v '^\\[kvm_exec\\]' | tail -14")
    elif name == "ops":
        out += sh(f"cd tests/gpu_ops && {B}/ops2/test_ops 2>&1 | tail -8; {B}/ops2/test_extra 2>&1 | tail -8")
        for t in ("test_ops", "test_extra"):
            for tool in ("memcheck", "racecheck", "synccheck"):
                out += f"== {t} {tool}\n" + sh(f"cd tests/gpu_ops && compute-sanitizer --tool {tool} --error-exitcode 9 {B}/ops2/{t} 2>&1 | tail -4; echo exit ${{PIPESTATUS[0]}}")
    elif name == "gpt":          # run_gpt.sh (544-task bundle, every scheduler mode, sanitizers) then the gptdiff golden checks
        out += sh(f"mkdir -p out && cp {B}/kexec{n}/kvm_exec out/kvm_exec_gpt && KVM_NO_LOCK=1 KVM_NVFLAGS=-DKVM_MIN_BLOCKS={n} bash tests/gpu/run_gpt.sh 2>&1 | tail -30")
        out += "== gptdiff\n" + sh(f"cp {B}/kexec{n}/kvm_exec ./kvm_exec; mkdir -p gd; fail=0; "
              "for spec in 'cpu 1' 'cpu 2' 'cpu 5' 'a100 2'; do set -- $spec; for mode in tf32 precise; do c=$1_s$2_$mode; "
              "python3 tests/gptdiff/gen_bundle.py gd/$c --steps $2 --tiling $1 $([ $mode = precise ] && echo --precise) >/dev/null; "
              "./kvm_exec --blocks 4 --quiet --dump-buffers gd/$c/bundle > gd/$c/dump.txt 2> gd/$c/err.txt; echo \"$c kvm_exec exit $?\"; "
              "python3 tests/gptdiff/golden_check.py gd/$c/workload.json gd/$c/bundle gd/$c/dump.txt --mode $mode | tail -4 || fail=1; done; done; "
              "for c in cpu_s2_tf32 cpu_s5_precise; do for tool in memcheck racecheck synccheck; do echo \"== sanitizer $tool $c\"; "
              "compute-sanitizer --tool $tool --error-exitcode 9 ./kvm_exec --blocks 4 --quiet gd/$c/bundle 2>&1 | tail -3; echo \"exit ${PIPESTATUS[0]}\"; done; done; echo gptdiff fail=$fail")
    elif name == "litmus":
        out += sh(f"mkdir -p out && cp -r {B}/litmus/. out/ && KVM_NO_LOCK=1 KVM_PREBUILT=1 bash tests/litmus/run_litmus.sh 2>&1 | tail -60")
    elif name == "steps":        # kernel_ms of two bundles at 1 vs 2 blocks/SM, alternating, 7 runs each; grid size is occupancy * SMs
        for model in ("tiny", "mid"):
            for rnd in range(2):
                for k in (1, 2):
                    out += f"== {model} MIN_BLOCKS={k} round {rnd}\n" + sh(f"cp {B}/kexec{k}/kvm_exec ./ke{k}; for i in 1 2 3 4 5 6 7; do ./ke{k} --quiet {B}/bundles/{model}/bundle 2>&1 | tail -1; done; ./ke{k} {B}/bundles/{model}/bundle 2>&1 | grep -i -E 'grid|blocks|resident' | head -3")
    return out


@app.function(cpu=2, timeout=24 * 3600, volumes={V: vol})
def orchestrate(tag: str):
    os.makedirs(f"{V}/{tag}", exist_ok=True)

    def save(name, text):
        open(f"{V}/{tag}/{name}.txt", "w").write(text)
        vol.commit()
        print(f"##### {name}\n{text[-3000:]}", flush=True)

    def pipe(name, builds, runs):
        try:
            txt = ""
            for j, n in builds:
                txt += build.remote(j, n)
            save(name + "_build", txt)
            for r, n in runs:
                save(r + f"_n{n}", run.remote(r, n))
        except Exception as e:  # noqa
            save(name + "_ERROR", repr(e))

    # every pipeline is independent; builds are the long part
    with cf.ThreadPoolExecutor(12) as ex:
        fs = [
            ex.submit(pipe, "exec2", [("test_exec2", 2)], [("exec_full", 2), ("exec_sanitize", 2)]),
            ex.submit(pipe, "exec1", [("test_exec1", 1)], []),
            ex.submit(pipe, "kexec2", [("kexec2", 2)], [("gpt", 2)]),
            ex.submit(pipe, "kexec1", [("kexec1", 1)], []),
            ex.submit(pipe, "litmus", [("litmus", 2)], [("litmus", 2)]),
            ex.submit(pipe, "ops", [("ops2", 2)], [("ops", 2)]),
            ex.submit(pipe, "bundles", [("bundles", 0)], []),
        ]
        # bench and steps need both builds: run after they are done
        for f in fs:
            f.result()
    try:
        save("exec_bench", run.remote("exec_bench", 0))
        save("steps", run.remote("steps", 0))
    except Exception as e:  # noqa
        save("late_ERROR", repr(e))


@app.local_entrypoint()
def main(tag: str = "run"):
    orchestrate.spawn(tag)
    print("spawned; logs in volume kvm-minblocks under /" + tag)
