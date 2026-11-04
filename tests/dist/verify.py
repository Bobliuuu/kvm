"""partition + simulate + compare against the unpartitioned CPU VM."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'dist'), str(Path(__file__).parent)]
import partition as pt   # noqa: E402
import sim               # noqa: E402
import workload_vm       # noqa: E402
import compile_workload as cw  # noqa: E402


def golden(spec):
    return workload_vm.run_golden(cw.compile_spec(copy.deepcopy(spec)))['buffers']


def check(spec, cluster, strategy, seeds=range(8), exact=True, rel=1e-5, outputs=None):
    """Returns (plan, report dict). Raises AssertionError describing the first mismatch."""
    plan = pt.partition(spec, cluster, strategy)
    gold = golden(plan.spec)
    worst = 0.0
    for seed in seeds:
        results, devs = sim.run(plan, seed)
        got = plan.reconstruct(results)
        for name in (outputs or got):
            if name not in gold:
                continue
            for i, (g, w) in enumerate(zip(got[name], gold[name])):
                if g == w:
                    continue
                err = abs(g - w) / max(1.0, abs(w))
                worst = max(worst, err)
                if exact or err > rel:
                    raise AssertionError(f'seed {seed}: {name}[{i}] = {g!r}, unpartitioned {w!r} (strategy {strategy})')
    return plan, dict(worst_rel=worst, seeds=len(list(seeds)))
