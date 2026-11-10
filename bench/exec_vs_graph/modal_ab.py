"""A/B executor variants against PyTorch on ONE Modal A100-40GB container (same GPU state for every number). Nothing here names a host or key.

  python3 bench/exec_vs_graph/stage_variants.py STAGE name=gitref[:kvm_exec args] ...   # git-archive each ref into STAGE/<tree>/
  modal run --detach bench/exec_vs_graph/modal_ab.py --stage STAGE --tag NAME [--configs "tiny small"] [--rounds 3] [--reps 5]
         [--nvcc "-DKVM_LEAN"] [--steps 50] [--modes graph,compile_graph] [--post 'bash cmd run in /work after the bench']
  modal volume get kvm-suite /NAME results/<dir>/raw

Builds (kvm_exec.cu, -O3 sm_80, --fmad=false -ffp-contract=off as in bench/exec_vs_graph) run first on a CPU-only container with enough
host RAM for the full-ops ptxas (~35 GB each), into the volume (/NAME/bin/<tree>/kvm_exec); the GPU container only measures. Configs:
tiny / small (gpt_tiny / gpt_small, TF32, --steps), b16 / f32 (gpt_162m bf16s / f32, a 1-step tape; torch runs autocast-bf16 / TF32).
Workloads come from bench/vs_pytorch/gen.py of the FIRST variant's tree. Every round times torch (MODES) once, then every variant REPS
launches in a rotated order, so clock drift hits all variants alike. Per-round JSON, summary.json and table.txt land in the volume.
"""
import json
import os
import pathlib

import modal

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("numpy", "torch==2.7.0"))
app = modal.App("kvm-exec-ab", image=image)
vol = modal.Volume.from_name("kvm-suite", create_if_missing=True)

M162 = '{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"precision":"%s"}'
CONFIGS = {  # name: (gen --model, steps override or None, torch precision)
    "tiny": ("gpt_tiny", None, "tf32"), "small": ("gpt_small", None, "tf32"),
    "b16": (M162 % "bf16s", 1, "bf16"), "f32": (M162 % "f32", 1, "tf32"),
}
FLAGS = "-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"


def _untar(tarball, dst):
    import io, tarfile
    os.makedirs(dst, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(tarball)).extractall(dst)


@app.function(cpu=16, memory=131072, timeout=2 * 3600, volumes={"/logs": vol})
def build(tag: str, tarball: bytes, trees: list, nvcc: str):
    import subprocess, time
    _untar(tarball, "/work")
    bdir = f"/logs/{tag}/bin"; os.makedirs(bdir, exist_ok=True)
    t0 = time.time(); procs = []
    for t in trees:
        os.makedirs(f"{bdir}/{t}", exist_ok=True)
        procs.append((t, subprocess.Popen(["bash", "-c", f"export PATH=$PATH:/usr/local/cuda/bin; nvcc {FLAGS} {nvcc} kvm_exec.cu -o {bdir}/{t}/kvm_exec > {bdir}/{t}/build.log 2>&1; echo rc=$?"],
                                          cwd=f"/work/{t}", stdout=subprocess.PIPE, text=True)))
    out = []
    for t, p in procs:
        out.append(f"build {t}: {p.communicate()[0].strip()}")
    out.append(f"builds done in {time.time() - t0:.0f} s (flags {FLAGS} {nvcc})")
    open(f"{bdir}/builds.txt", "w").write("\n".join(out) + "\n"); vol.commit()
    return "\n".join(out)


@app.function(gpu="A100-40GB", cpu=8, memory=49152, timeout=4 * 3600, volumes={"/logs": vol})
def run(tag: str, tarball: bytes, variants: list, configs: str, rounds: int, reps: int, steps: int, modes: str, post: str):
    import subprocess, statistics
    out = f"/logs/{tag}"; os.makedirs(out, exist_ok=True)
    log = open(f"{out}/ab.log", "a")

    def sh(cmd, cwd="/work", check=False, timeout=None):
        log.write(f"$ {cmd}\n"); log.flush()
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        log.flush(); vol.commit()
        if check and p.returncode: raise SystemExit(f"failed: {cmd}")
        return p.returncode

    _untar(tarball, "/work")
    sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1; cat " + f"{out}/bin/builds.txt")
    for v in variants:
        if not os.path.exists(f"{out}/bin/{v['tree']}/kvm_exec"): raise SystemExit(f"no binary for {v['tree']} (build failed?)")
        sh(f"cp {out}/bin/{v['tree']}/kvm_exec {v['tree']}/kvm_exec; chmod +x {v['tree']}/kvm_exec")
    first = variants[0]["tree"]
    st = {}
    for c in configs.split():
        model, so, _ = CONFIGS.get(c, (c, None, "tf32"))
        st[c] = so or steps
        sh(f"python3 -I {first}/bench/vs_pytorch/gen.py w_{c} --model '{model}' --steps {st[c]} --mode tf32 --tiling a100 > {out}/gen_{c}.json 2>&1", check=True)
    res = {"variants": variants, "rounds": []}
    for r in range(rounds):
        rr = {}
        for c in configs.split():
            prec = CONFIGS.get(c, (c, None, "tf32"))[2]
            if modes:
                sh(f"python3 -I {first}/bench/vs_pytorch/torch_bench.py w_{c} --reps {reps} --precision {prec} --modes {modes} --out {out}/r{r}_{c}_torch.json 2>&1 | grep -v Warning")
            order = variants[r % len(variants):] + variants[:r % len(variants)]
            for v in order:
                extra = v.get("args", "")
                sh(f"python3 -I {first}/bench/vs_pytorch/exec_bench.py w_{c} {v['tree']}/kvm_exec --reps {reps} --out {out}/r{r}_{c}_{v['name']}.json" + (f" -- {extra}" if extra else ""))
                try:
                    rr.setdefault(c, {})[v["name"]] = json.load(open(f"{out}/r{r}_{c}_{v['name']}.json")).get("ms_total", [])
                except Exception as e:
                    rr.setdefault(c, {})[v["name"]] = str(e)
        res["rounds"].append(rr)
        json.dump(res, open(f"{out}/summary.json", "w"), indent=1); vol.commit()
    lines = []
    for c in configs.split():
        for r in range(rounds):
            try:
                tj = json.load(open(f"{out}/r{r}_{c}_torch.json"))
                lines.append(f"{c:6s} torch r{r} ({tj.get('gpu', '?')}): " + "  ".join(
                    f"{m} {v['ms_per_step_median'] * 1e3:.0f}" if isinstance(v, dict) and 'ms_per_step_median' in v else f"{m} FAILED"
                    for m, v in tj.get('modes', {}).items()) + " us/step (median)")
            except Exception:
                pass
        for v in variants:
            ms = [x for rr in res["rounds"] for x in (rr.get(c, {}).get(v["name"]) or []) if isinstance(x, float)]
            n = st[c]
            lines.append(f"{c:6s} {v['name']:14s} " + (f"best {min(ms) / n * 1e3:9.0f} median {statistics.median(ms) / n * 1e3:9.0f} us/step (n={len(ms)})" if ms else "NO DATA"))
    open(f"{out}/table.txt", "w").write("\n".join(lines) + "\n")
    if post:
        sh(post, timeout=3 * 3600)
    log.close(); vol.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def main(stage: str, tag: str, configs: str = "tiny small", rounds: int = 3, reps: int = 5, nvcc: str = "-DKVM_LEAN", steps: int = 50,
         modes: str = "graph,compile_graph", post: str = "", skip_build: bool = False, build_only: bool = False):
    import io, tarfile
    st = pathlib.Path(stage)
    variants = json.loads((st / "variants.json").read_text())
    trees = sorted({v["tree"] for v in variants})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for t in trees:
            tf.add(st / t, arcname=t)
    tb = buf.getvalue()
    print(f"{len(variants)} variants over {len(trees)} trees, tarball {len(tb) / 1e6:.1f} MB -> volume kvm-suite:/{tag}")
    if not skip_build:
        print(build.remote(tag, tb, trees, nvcc))
    if build_only:
        return
    print(run.remote(tag, tb, variants, configs, rounds, reps, steps, modes, post))
