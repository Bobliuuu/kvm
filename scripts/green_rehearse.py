#!/usr/bin/env python3
"""CPU rehearsal of the green table (scripts/green_all.py): no GPU, no Modal, minutes.

  python3 scripts/green_rehearse.py [--only job1,job2] [--real-unit]

1. Wiring: every job names builds that exist, every `$B/x` or `$BZ/x` a job reads is a build's output, every repository path a build or a
   job names exists in this tree, and every job script (the orchestrator's HEAD + PASSFN + the job) and build command passes `bash -n`.
2. Verdicts: each job runs its own script, verbatim, in a copy of the tree, against stand-ins for what needs a GPU or hours (the binaries
   in $B, the GPU tools on PATH, the tree scripts that drive them). Every stand-in prints the real tool's own result lines (its source
   names them), first a passing transcript, then a failing one; the job must end in `RESULT: PASS`, then in `RESULT: FAIL`. So a pass
   pattern that cannot match, a $B path that is wrong, or a failure the job misses shows up here, before any GPU hour.
3. `--real-unit` also runs the `unit` job for real (it needs no GPU): the CPU suites of this tree, as the table runs them.
The orchestrator's volume (/logs) is a temporary folder here; nothing else in the scripts changes.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS, FAIL = 'pass', 'fail'
# What a stand-in prints and returns: {mode: (output, exit code)}. Lines come from the real tools' sources.
OK_EXEC = ('8613 checks, 0 failed, 0 suites failed', 0)
BAD_EXEC = ('8613 checks, 2 failed, 1 suites failed', 1)
OK_OPS = ('5403 checks passed, 0 failed', 0)
BAD_OPS = ('5401 checks passed, 2 failed', 1)
BINARIES = {   # $B/name: argument-aware stand-ins (bash case on "$*")
    'test_exec': {PASS: '*) echo "%s"; exit %d;;' % OK_EXEC, FAIL: '*) echo "%s"; exit %d;;' % BAD_EXEC},
    'test_exec_bf': {PASS: '*--expect-fail*) echo "violation detected, as expected"; exit 0;;', FAIL: '*) echo "no violation detected"; exit 1;;'},
    'test_exec_bp': {PASS: '*--expect-fail*) echo "violation detected, as expected"; exit 0;;', FAIL: '*) echo "no violation detected"; exit 1;;'},
    'test_exec_bw': {PASS: '*recycle*) echo "FAIL window_done false for 3 of 4096 finished windows after the 64-entry table wrapped 64 times"; '
                           'echo "%s"; exit 1;;' % BAD_EXEC[0],
                     FAIL: '*recycle*) echo "%s"; exit 0;;' % OK_EXEC[0]},
    'test_ops': {PASS: '*) echo "%s"; exit %d;;' % OK_OPS, FAIL: '*) echo "%s"; exit %d;;' % BAD_OPS},
    'test_e2e': {PASS: '*) echo "7 of 7 end-to-end cases passed"; exit 0;;', FAIL: '*) echo "6 of 7 end-to-end cases passed"; exit 1;;'},
    'test_extra': {PASS: '*) echo "%s"; exit %d;;' % OK_OPS, FAIL: '*) echo "%s"; exit %d;;' % BAD_OPS},
    'test_bf16': {PASS: '*) echo "%s"; exit %d;;' % OK_OPS, FAIL: '*) echo "%s"; exit %d;;' % BAD_OPS},
    'test_sched': {PASS: '*) echo "%s"; exit %d;;' % OK_EXEC, FAIL: '*) echo "%s"; exit %d;;' % BAD_EXEC},
    'kvm_exec': {PASS: '*--dump-buffers*) echo "buffer 0 w.fc1 ok"; exit 0;; *) echo "OK: 544/544 tasks"; exit 0;;',
                 FAIL: '*) echo "DEVICE ERROR KVM_ERR_TIMEOUT (code 3)"; exit 2;;'},
    'kvm_run_nogame': {PASS: '*) exit 0;;', FAIL: '*) exit 1;;'},
    'kvm_run_game': {PASS: '*) exit 0;;', FAIL: '*) exit 1;;'},
}
TOOLS = {   # stand-ins on PATH for the GPU's own tools
    'nvidia-smi': {PASS: '*) echo "NVIDIA A100-SXM4-40GB (stand-in), 570.00, 1410 MHz"; exit 0;;', FAIL: '*) echo "NVIDIA A100-SXM4-40GB (stand-in)"; exit 0;;'},
    'compute-sanitizer': {PASS: '*) while [ "${1#--}" != "$1" ]; do shift 2; done; "$@"; rc=$?; echo "========= ERROR SUMMARY: 0 errors"; exit $rc;;',
                          FAIL: '*) echo "========= Invalid __global__ read of size 4 (stand-in)"; echo "========= ERROR SUMMARY: 1 error"; exit 9;;'},
}
SCRIPTS = {   # tree scripts the jobs call, replaced in the copy: {path: {mode: body}}; sh for .sh, python for .py
    'tests/litmus/run_litmus.sh': {PASS: 'echo "SUMMARY: 31 properties/scenarios/probes passed, 9 mutants/controls detected, 0 failures"; echo "LITMUS PASS"',
                                   FAIL: 'echo "SUMMARY: 30 properties/scenarios/probes passed, 9 mutants/controls detected, 1 failures"; echo "LITMUS FAIL"; exit 1'},
    'tests/gpu/run_gpt.sh': {PASS: 'echo "GPT PASS"', FAIL: 'echo "GPT FAIL (1)"; exit 1'},
    'tests/static/run_check.sh': {PASS: 'echo "PASS: static schedule bitwise equal to the dynamic path; fused MLP step within its gates"',
                                  FAIL: 'echo "FAIL: run gpt_static_w107 (--static)"; echo "FAIL: 1 checks failed"; exit 1'},
    'examples/models/run_box.sh': {PASS: 'echo "PASS mlp_tiny: check.py ok"; echo "exit 0"', FAIL: 'echo "FAIL vit_bf16s: weights mismatch 3.1e-02"; exit 1'},
    'tests/gptdiff/gen_bundle.py': {PASS: 'import os, sys; os.makedirs(sys.argv[1] + "/bundle", exist_ok=True)', FAIL: 'import os, sys; os.makedirs(sys.argv[1] + "/bundle", exist_ok=True)'},
    'tests/gptdiff/golden_check.py': {PASS: 'print("golden: every buffer within tolerance")', FAIL: 'print("golden: w.fc1 off by 4.2e-02"); raise SystemExit(1)'},
    'tests/bf16/gen.py': {PASS: 'import os, sys\nfor c in ("f32_s50", "bf16s_s50", "bf16s_s1", "bf16s_s5"): os.makedirs(f"{sys.argv[1]}/{c}/bundle", exist_ok=True)',
                          FAIL: 'import os, sys\nfor c in ("f32_s50", "bf16s_s50", "bf16s_s1", "bf16s_s5"): os.makedirs(f"{sys.argv[1]}/{c}/bundle", exist_ok=True)'},
    'tests/bf16/check.py': {PASS: 'print("bf16 check ok")', FAIL: 'print("bf16 check: loss off by 3e-02"); raise SystemExit(1)'},
    'bench/vs_pytorch/grads162.sh': {PASS: 'echo "  worst tensor: blocks.0.attn exec 1.20e-03 (torch-tf32 on it 1.50e-03, torch-bf16 2.1e-02); all 148 tensors: max exec 1.20e-03, max torch-tf32 1.50e-03, max torch-bf16 2.10e-02"',
                                     FAIL: 'echo "  worst tensor: blocks.0.attn exec 4.00e-01 (torch-tf32 on it 1.50e-03, torch-bf16 2.1e-02); all 148 tensors: max exec 4.00e-01, max torch-tf32 1.50e-03, max torch-bf16 2.10e-02"'},
    'tests/gpt162/learning.sh': {PASS: 'mkdir -p "$1"; cat > "$1/summary.json" <<EOF\n{"max_rel_gap_vs_tf32": {"value": 4e-4}, "max_rel_gap_tf32_vs_fp32 (noise floor)": {"value": 3e-4}, '
                                       '"last_mean": {"executor": 0.5123, "torch_tf32": 0.5121}}\nEOF\necho "300 steps done"',
                                 FAIL: 'mkdir -p "$1"; cat > "$1/summary.json" <<EOF\n{"max_rel_gap_vs_tf32": {"value": 2e-2}, "max_rel_gap_tf32_vs_fp32 (noise floor)": {"value": 3e-4}, '
                                       '"last_mean": {"executor": 0.6123, "torch_tf32": 0.5121}}\nEOF\necho "300 steps done"'},
    'tests/gpt162/learning_plot.py': {PASS: 'print("plot written")', FAIL: 'print("plot written")'},
    'demo/remote_demo.sh': {PASS: 'mkdir -p runs/big; printf "PASS game: compared 700 frames, raw-identical 700/700\\nPASS cadence 35.0 Hz\\nALL CHECKS PASSED\\n" > runs/big/check.log; '
                                  'echo "window sizing: 8 windows" > runs/big/kvm_run.log; echo "{}" > runs/big/report.json',
                            FAIL: 'mkdir -p runs/big; printf "FAIL game: compared 700 frames, raw-identical 698/700\\n" > runs/big/check.log; : > runs/big/kvm_run.log; exit 1'},
    'dist/run_multigpu_tests.sh': {PASS: 'echo "PASS multigpu-p2p-pingpong (3s)"; echo "SKIP rdma-device-present: no RDMA device node (/dev/infiniband/uverbs*)"; '
                                         'echo "SKIP multinode-tcp-plans: needs KVM_NODES with 2+ nodes"; echo "passed 14, failed 0, skipped 2   (GPUs: 2; logs: logs/x-dist)"; echo "RESULT: PASS"',
                                   FAIL: 'echo "SKIP multigpu-p2p-pingpong: needs 2 GPUs"; echo "passed 9, failed 0, skipped 1"; echo "RESULT: PASS"'},
}


def load_green(path: Path) -> types.SimpleNamespace:
    """BUILDS, JOBS, NEEDS, HEAD and PASSFN from green_all.py, with a stand-in `modal` module so it runs without Modal."""
    class Chain:
        def __getattr__(self, _name):
            return lambda *a, **k: self

        def __call__(self, *a, **k):
            return self
    stub = types.ModuleType('modal')
    stub.Image, stub.Volume = Chain(), Chain()
    stub.App = lambda *a, **k: types.SimpleNamespace(function=lambda *a, **k: (lambda fn: fn), local_entrypoint=lambda *a, **k: (lambda fn: fn))
    saved = sys.modules.get('modal')
    sys.modules['modal'] = stub
    try:
        scope = {'__file__': str(path), '__name__': 'green_all'}
        exec(compile(path.read_text(), str(path), 'exec'), scope)
    finally:
        sys.modules.pop('modal', None) if saved is None else sys.modules.__setitem__('modal', saved)
    return types.SimpleNamespace(**{k: scope[k] for k in ('BUILDS', 'JOBS', 'NEEDS', 'HEAD', 'PASSFN', 'IGN')})


PATHS = re.compile(r'(?<![\w/$.-])((?:tests|examples|bench|demo|doom|dist|frontend|runtime|scripts|include)/[\w./-]+|kvm_exec\.cu)')


def wiring(g) -> list[str]:
    errs = []
    for job in g.JOBS:
        if job not in g.NEEDS:
            errs.append(f'job {job} has no NEEDS entry')
    for job, builds in g.NEEDS.items():
        errs += [f'NEEDS[{job}] names unknown build {b}' for b in builds if b not in g.BUILDS]
    outputs = {o for _, outs in g.BUILDS.values() for o in outs}
    for job, (script, _) in g.JOBS.items():
        for used in re.findall(r'\$BZ?/([\w.-]+)', script):
            if used not in outputs:
                errs.append(f'job {job} reads $B/{used}, which no build outputs')
    texts = [('build ' + n, c) for n, (c, _) in g.BUILDS.items()] + [('job ' + n, s) for n, (s, _) in g.JOBS.items()]
    made = tuple(i for i in g.IGN if '*' not in i)   # paths a build makes (the orchestrator's upload skips them): never in a clean tree
    for what, text in texts:
        for p in sorted(set(PATHS.findall(text))):
            if not p.startswith(made) and not (ROOT / p.rstrip('.')).exists():
                errs.append(f'{what} names {p}, which this tree lacks')
    for what, text in texts:
        script = (g.HEAD + g.PASSFN + text) if what.startswith('job') else text
        r = subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True)
        if r.returncode:
            errs.append(f'{what}: bash -n: {r.stderr.strip()[:200]}')
    return errs


def stub(path: Path, body: str, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text({'sh': '#!/usr/bin/env bash\n', 'py': '#!/usr/bin/env python3\n', 'case': '#!/usr/bin/env bash\ncase "$*" in\n'}[kind] + body +
                    ('\nesac\n' if kind == 'case' else '\n'))
    path.chmod(0o755)


def sandbox(base: Path, mode: str, g) -> tuple[Path, Path, dict]:
    """A copy of the tree with the stand-ins for MODE, its $B, its PATH and its /logs."""
    tree, logs, tools = base / 'tree', base / 'logs', base / 'tools'
    shutil.copytree(ROOT, tree, symlinks=True, ignore=shutil.ignore_patterns('.git', '__pycache__', 'results', 'out', 'third_party', 'build'))
    for rel, modes in SCRIPTS.items():
        stub(tree / rel, modes[mode], 'py' if rel.endswith('.py') else 'sh')
    key, keyz = 'rehearsal', 'rehearsalz'
    bins, binz = logs / 'bin' / key, logs / 'bin' / keyz
    for name, modes in BINARIES.items():
        stub(bins / name, modes[mode], 'case')
    for d in ('litmus_bins', ):
        (bins / d).mkdir(parents=True, exist_ok=True)
        stub(bins / d / 'litmus_prod', 'exit 0', 'sh')
    for d in ('zoo_bundles', 'zoo_bundles_bf16s'):
        (binz / d / 'case0').mkdir(parents=True, exist_ok=True)
    (tree / 'doom' / 'build').mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', 'czf', str(bins / 'doom_assets.tgz'), '-C', str(tree), 'doom/build'], check=True)
    for name, modes in TOOLS.items():
        stub(tools / name, modes[mode], 'case')
    env = dict(os.environ, KEY=key, KEYZ=keyz, PATH=f'{tools}:{os.environ["PATH"]}', GREEN_STANDIN=mode)
    return tree, logs, env


def run_job(g, name: str, tree: Path, logs: Path, env: dict, tag: str) -> str:
    script = (g.HEAD + g.PASSFN + g.JOBS[name][0]).replace('/logs/', f'{logs}/').replace('~/.gpu.lock', f'{logs}/.gpu.lock')
    (logs / tag).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['bash', '-c', script], cwd=tree, capture_output=True, text=True, env=dict(env, TAG=tag, JOB=name), timeout=3600)
    (logs / tag / f'{name}.log').write_text(r.stdout + r.stderr)
    last = [line for line in (r.stdout + r.stderr).splitlines() if line.startswith('RESULT:')]
    return last[-1] if last else 'RESULT: FAIL no result line'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', default='', help='comma-separated jobs (default: every job)')
    ap.add_argument('--real-unit', action='store_true', help='also run the unit job for real (the CPU suites)')
    ap.add_argument('--keep', action='store_true', help='keep the sandboxes and their logs')
    a = ap.parse_args()
    g = load_green(ROOT / 'scripts' / 'green_all.py')
    errs = wiring(g)
    print(f'wiring: {len(g.BUILDS)} builds, {len(g.JOBS)} jobs: ' + ('ok' if not errs else f'{len(errs)} problem(s)'))
    for e in errs:
        print(f'  WIRING {e}')
    names = [n for n in g.JOBS if (not a.only or n in a.only.split(',')) and n != 'unit']
    base = Path(tempfile.mkdtemp(prefix='green-rehearsal-'))
    bad = len(errs)
    try:
        verdicts = {}
        for mode in (PASS, FAIL) if names else ():
            tree, logs, env = sandbox(base / mode, mode, g)
            for n in names:
                verdicts[(n, mode)] = run_job(g, n, tree, logs, env, 'rehearsal')
        w = max([len(n) for n in names] + [len('unit')])
        if names:
            print(f'verdicts: each job against a passing, then a failing stand-in transcript (logs {base if a.keep else "removed"}):')
        for n in names:
            vp, vf = verdicts[(n, PASS)], verdicts[(n, FAIL)]
            ok = vp.startswith('RESULT: PASS') and vf.startswith('RESULT: FAIL')
            bad += not ok
            print(f'  {n:<{w}}  {"ok      " if ok else "MISMATCH"}  pass case: {vp[8:90]}' + ('' if ok else f'  | fail case: {vf[8:90]}'))
        if a.real_unit and (not a.only or 'unit' in a.only.split(',')):
            v = run_job(g, 'unit', ROOT, base / 'real-logs', dict(os.environ, KEY='real', KEYZ='realz'), 'real')
            bad += not v.startswith('RESULT: PASS')
            print(f'  {"unit":<{w}}  {"REAL ok " if v.startswith("RESULT: PASS") else "REAL FAIL"}  {v[8:200]}')
    finally:
        if not a.keep:
            shutil.rmtree(base, ignore_errors=True)
    print('REHEARSAL ' + ('PASS: wiring clean, every job passes a sound run and fails a broken one' if not bad else f'FAIL: {bad} problem(s)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
