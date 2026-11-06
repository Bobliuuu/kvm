"""Modal runner for the GPT-op kernel benches: builds test_ops + bench_ops for each variant and runs them on one GPU.
Launch (detached): BASE_DIR=<dir with include/ and tests/ of the baseline> GPU=A100-40GB modal run --detach tests/gpu_ops/modal_bench.py
Each result is printed as a JSON line and also written to the Modal volume 'kvm-kern-results'."""
import json, os, subprocess, time
import modal

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASE = os.environ.get('BASE_DIR', '')
GPU = os.environ.get('GPU', 'A100-40GB')
ARCH = os.environ.get('ARCH', 'sm_80')
TAG = os.environ.get('TAG', 'run')
VARIANTS = json.loads(os.environ.get('VARIANTS', '{"head": "", "base": ""}'))   # name -> extra nvcc flags ("base" uses BASE_DIR)
img = modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.11').add_local_dir(
    os.path.join(REPO, 'include'), '/w/head/include', copy=True).add_local_dir(os.path.join(REPO, 'tests', 'gpu_ops'), '/w/head/tests/gpu_ops', copy=True)
if BASE:
    img = img.add_local_dir(os.path.join(BASE, 'include'), '/w/base/include', copy=True).add_local_dir(os.path.join(BASE, 'tests', 'gpu_ops'), '/w/base/tests/gpu_ops', copy=True)
vol = modal.Volume.from_name('kvm-kern-results', create_if_missing=True)
app = modal.App('kvm-kern-bench')

def sh(cmd, cwd=None, env=None):
    t = time.time(); p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return p.returncode, p.stdout + p.stderr, time.time() - t

@app.function(image=img, gpu=GPU, cpu=16, memory=65536, timeout=3 * 3600, volumes={'/out': vol})
def run(variants: dict, arch: str, tag: str):
    out = {}
    procs = {}
    for name, flags in variants.items():
        d = f'/w/{name if name != "head" and name != "base" else name}/tests/gpu_ops'
        if not os.path.isdir(d): d = '/w/head/tests/gpu_ops'; src = '/w/head'
        for prog, src_f in (('bench', 'bench_ops.cu'), ('test', 'test_ops.cu')):
            cmd = f'nvcc -O2 -arch={arch} -std=c++17 -lineinfo -I../../include -I. {flags} {src_f} -o {name}_{prog}.bin 2>&1 | grep -E "error|spill"'
            procs[(name, prog)] = (subprocess.Popen(cmd, shell=True, cwd=d), d)
    for k, (p, d) in procs.items(): p.wait()
    for name in variants:
        d = f'/w/{name}/tests/gpu_ops'
        if not os.path.isdir(d): d = '/w/head/tests/gpu_ops'
        for label, exe, env in (('test', f'{name}_test.bin', {'X': '1'}), ('mm', f'{name}_bench.bin', {'BENCH_ONLY': 'mm', 'BENCH_REPS': '5'}), ('attn', f'{name}_bench.bin', {'BENCH_ONLY': 'attn', 'BENCH_REPS': '5'})):
            rc, o, dt = sh(f'./{exe}', cwd=d, env=env)
            rec = {'tag': tag, 'variant': name, 'part': label, 'rc': rc, 'secs': round(dt, 1), 'out': o}
            print(json.dumps(rec), flush=True)
            open(f'/out/{tag}.{name}.{label}.txt', 'w').write(o)
    vol.commit()

@app.local_entrypoint()
def main():
    run.remote(VARIANTS, ARCH, TAG)
