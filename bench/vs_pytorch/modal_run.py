"""Run the executor-vs-PyTorch campaign on a Modal GPU (an alternative to a shared box).

  modal run --detach bench/vs_pytorch/modal_run.py [--gpu A100-40GB] [--out modal_out]

Builds kvm_exec with the repository's flags, generates the workloads, runs bench/vs_pytorch/campaign.sh (no lock: the GPU is private)
and returns the result directory, which is unpacked under --out. Raw output stays in the Modal volume `kvm-vs-pytorch` too.
"""
import io
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
app = modal.App('kvm-vs-pytorch')
vol = modal.Volume.from_name('kvm-vs-pytorch', create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.10')
         .pip_install('torch==2.7.0', 'numpy')
         .add_local_dir(str(ROOT), '/repo', copy=True, ignore=['.git', 'logs', 'docs', 'results', 'doom/third_party', 'doom/build', 'dist']))
MID = '{"name":"gpt_mid","vocab":512,"d_model":256,"layers":4,"heads":4,"seq":128,"batch":8}'


@app.function(image=image, gpu='A100-40GB', timeout=3 * 3600, volumes={'/vol': vol})
def run(commit: str = 'unknown'):
    import os
    import subprocess
    sh = lambda c: subprocess.run(['bash', '-c', c], cwd='/repo', check=True, env={**os.environ, 'KVM_COMMIT': commit})
    sh('nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o kvm_exec')
    g = 'python3 bench/vs_pytorch/gen.py'
    sh(f'{g} w_tiny --model gpt_tiny --steps 50 >/dev/null && {g} w_tiny_p --model gpt_tiny --steps 50 --mode precise >/dev/null && '
       f'{g} w_small --model gpt_small --steps 50 >/dev/null && {g} w_mid --model \'{MID}\' --steps 20 >/dev/null')
    sh('bash bench/vs_pytorch/campaign.sh runs/modal 2>&1 | tee runs/modal.log')
    sh('mkdir -p /vol/runs && rm -rf /vol/runs/modal && cp -r runs/modal /vol/runs/modal && cp runs/modal.log /vol/runs/')
    vol.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        t.add('/repo/runs/modal', arcname='modal')
        t.add('/repo/runs/modal.log', arcname='modal.log')
    return buf.getvalue()


@app.local_entrypoint()
def main(out: str = 'modal_out'):
    import subprocess
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    data = run.remote(commit)
    Path(out).mkdir(parents=True, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(data), mode='r:gz').extractall(out)
    print('unpacked to', out)
