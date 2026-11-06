"""Real multi-GPU runs of the sharded KVM task graphs and comm ops on Modal (single containers with 2/4/8 GPUs: NVLink inside the
container; NO multi-node network, Modal cannot give that in one container).

    modal run --detach dist/modal_scale.py::main --gpus a100x2      # also a100x4, h100x8 (only after the A100 runs pass)

Each remote call builds test_dist for the GPU's arch, runs the suites, prints every result as one JSON line (app logs) and writes
files to the Modal Volume `kvm-scale-results` under /results/<tag>/. Spend per run = wall hours x GPUs x list price (printed).
"""
import json
import os
import subprocess
import time

import modal

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = modal.App('kvm-scale-evidence')
vol = modal.Volume.from_name('kvm-scale-results', create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.11')
         .pip_install('numpy')
         .add_local_dir(SRC, '/src', copy=True, ignore=['.git', 'results', 'logs', 'doom', 'demo', 'frontend', 'examples', '**/__pycache__']))

CONF = {  # name: (gpu spec, arch, $/GPU/h list price as given by the coordinator)
    'a100x2': ('A100-40GB:2', 'sm_80', 2.10),
    'a100x4': ('A100-40GB:4', 'sm_80', 2.10),
    'h100x8': ('H100:8', 'sm_90', 3.95),
}


def sh(cmd, log, timeout):
    t0 = time.time()
    p = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=timeout, cwd='/src')
    out = p.stdout + p.stderr
    with open(log, 'w') as f:
        f.write(out)
    return p.returncode, out, time.time() - t0


def body(tag, ngpu, arch, price, big):
    t_start = time.time()
    R = f'/results/{tag}'
    os.makedirs(R, exist_ok=True)

    def emit(**k):
        print('RESULT ' + json.dumps(dict(tag=tag, **k)), flush=True)
    rc, out, _ = sh('nvidia-smi -L; nvidia-smi topo -m; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvcc --version | tail -1',
                    f'{R}/hw.txt', 120)
    emit(stage='hw', rc=rc, text=out[-3000:])
    rc, out, dt = sh(f'nvcc -O3 -arch={arch} -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN '
                     f'-diag-suppress 177 tests/dist/test_dist.cu -o /tmp/test_dist -lpthread 2>&1 | grep -v warning | tail -20', f'{R}/build.txt', 1500)
    emit(stage='build', rc=rc, s=round(dt, 1), tail=out[-500:])
    if not os.path.exists('/tmp/test_dist'):
        return
    env = 'export KVM_TEST_DIST=/tmp/test_dist; '
    # -- real peer-to-peer: bandwidth and cross-GPU hop latency
    for suite in ('bw', 'pingpong', 'ops'):
        rc, out, dt = sh(f'{env} /tmp/test_dist {suite}', f'{R}/{suite}.txt', 900)
        emit(stage=suite, rc=rc, s=round(dt, 1), text=out[-6000:])
    # -- partitioner plans, one rank per GPU
    sh(f'python tests/dist/make_plans.py /tmp/plans --devices {ngpu}', f'{R}/make_plans.txt', 600)
    for p in ('tp_gpus', 'zero_gpus', 'pipe_p2p', 'tp_ring', 'zero_mixed', 'tp_direct', 'tp_tree'):
        rc, out, dt = sh(f'{env} /tmp/test_dist plan /tmp/plans/{p}', f'{R}/plan_{p}.txt', 600)
        emit(stage=f'plan_{p}', rc=rc, s=round(dt, 1), text=out[-1500:])
    # -- sharded AdamW stack: CPU golden compared on the devices, ranks spread over the GPUs round-robin
    rc, out, dt = sh(f'{env} python dist/scale_gpu.py golden {R}/golden', f'{R}/golden.txt', 1500)
    emit(stage='golden_stack', rc=rc, s=round(dt, 1), text=out[-6000:])
    # -- unsharded (GPU 0) vs sharded over all GPUs, same seeded data
    for (d, L, m) in big:
        rc, out, dt = sh(f'{env} python dist/scale_gpu.py pair {R}/pair_d{d}_L{L}_r{ngpu} --d {d} --layers {L} --ranks {ngpu} --m {m} --steps 3 --timeout 1200',
                         f'{R}/pair_d{d}_L{L}.txt', 2400)
        emit(stage=f'pair_d{d}_L{L}', rc=rc, s=round(dt, 1), text=out[-5000:])
    hours = (time.time() - t_start) / 3600
    emit(stage='cost', wall_s=round(hours * 3600), gpus=ngpu, usd_estimate=round(hours * ngpu * price, 2))
    vol.commit()


def make(name):
    gpu, arch, price = CONF[name]

    @app.function(gpu=gpu, image=image, volumes={'/results': vol}, timeout=3 * 3600, name=f'run_{name}')
    def run(tag):
        big = [(1024, 8, 64), (2048, 8, 64)] if name != 'h100x8' else [(2048, 8, 64), (4096, 8, 64)]
        body(tag, int(gpu.split(':')[1]), arch, price, big)
    return run


FUNCS = {n: make(n) for n in CONF}


@app.local_entrypoint()
def main(gpus: str = 'a100x2'):
    tag = f'{gpus}-{time.strftime("%Y%m%dT%H%M%S")}'
    call = FUNCS[gpus].spawn(tag)
    print('spawned', tag, call.object_id)
