"""Reference host runner: executes a demo directory on the CPU VM using ONLY MANIFEST.json (no model or game knowledge).

    python3 -m kvmfe.run_manifest DIR [--allow-opaque]

Admissions run one at a time, in the first order the manifest's conditions allow (time conditions are ignored: this is a serial
simulation, not a clock). It is the executable specification of the manifest semantics for the real host runner.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import _runtime, log
from .ir import FrontendError


GUARD_CMP = {'eq': lambda a, b: a == b, 'ne': lambda a, b: a != b, 'lt': lambda a, b: a < b, 'le': lambda a, b: a <= b,
             'gt': lambda a, b: a > b, 'ge': lambda a, b: a >= b}


def validate_guard(adm):
    """A guarded admission (manifest_version 2) runs only if a scalar of a buffer, read when the admission becomes eligible, passes the
    comparison: {"buffer": name, "index": i, "cmp": eq|ne|lt|le|gt|ge, "value": x}. It must not publish a variable: whether it runs is
    decided by data, so the version arithmetic every other admission was compiled against (base_versions, `when`) has to stay static."""
    g = adm['guard']
    if set(g) != {'buffer', 'index', 'cmp', 'value'} or g['cmp'] not in GUARD_CMP:
        raise FrontendError(f"admission {adm['name']}: guard must be exactly buffer, index, cmp ({'|'.join(GUARD_CMP)}), value; got {g!r}")
    if adm['publishes']:
        raise FrontendError(f"admission {adm['name']}: a guarded admission cannot publish {sorted(adm['publishes'])} (its version flow would depend on data)")


def guard_holds(adm, buffers):
    g = adm['guard']
    return GUARD_CMP[g['cmp']](buffers[g['buffer']][g['index']], g['value'])


def eligible(adm, done, versions):
    if any(b not in done for b in adm['after_bundles']):
        return False
    for w in adm['when']:
        v = versions.get(w['var'], 0)
        if v < w['min'] or ('max' in w and v > w['max']):
            return False
    return True


def run_manifest(root, allow_opaque=False, timeout=3600, after_each=None):
    import compile_workload as cw
    import workload_vm as wvm
    root = Path(root)
    m = json.loads((root / 'MANIFEST.json').read_text())
    vm = wvm.Runtime(1, deterministic=True, allow_opaque=allow_opaque)
    for wl in sorted({b['owner'] for b in m['buffers'].values()}):          # initialize each buffer once, from its owner workload
        adm = next(a for a in m['admissions'] if a['workload'] == wl)
        c = adm['compile']
        comp = cw.compile_spec(json.loads((root / c['workload']).read_text()), base_versions=c['base_versions'], data_dir=root / c['data_dir'])
        for b in comp.buffers:
            if m['buffers'][b.name]['owner'] == wl:
                vm.buffers[b.name] = wvm.Buf(b.materialize(comp.data_dir), b.rows, b.cols, b.dtype)
                vm.shapes[b.name] = (b.rows, b.cols, b.dtype)
    vm.start()
    for a in m['admissions']:
        if 'guard' in a:
            validate_guard(a)
    if any('guard' in a for a in m['admissions']) and m['manifest_version'] < 2:
        raise FrontendError('guarded admissions need manifest_version 2')
    done, pending, order, skipped = set(), list(m['admissions']), [], []
    try:
        while pending:
            adm = next((a for a in pending if eligible(a, done, vm.versions)), None)
            if adm is None:
                raise FrontendError(f"no admission can start: pending {[a['name'] for a in pending]}, versions {vm.versions}")
            pending.remove(adm)
            # data-dependent control flow: a guard that fails, or a skipped admission in after_bundles, skips the admission without running it.
            # `done` still receives its name so after_bundles chains see it as finished; nothing is published, so a skipped admission that
            # would have published (only reachable through the cascade: guards forbid it) is an error, never a silent version gap.
            cascade = any(b in skipped for b in adm['after_bundles'])
            if cascade or ('guard' in adm and not guard_holds(adm, vm.buffers)):
                if adm['publishes']:
                    raise FrontendError(f"admission {adm['name']} publishes {sorted(adm['publishes'])} but depends on a skipped admission")
                skipped.append(adm['name'])
                done.add(adm['name'])
                log.info('skipped %s (%s)', adm['name'], 'depends on a skipped admission' if cascade else 'guard false')
                continue
            c = adm['compile']
            spec = json.loads((root / c['workload']).read_text())
            base = {v: (n if v in adm['publishes'] else max(n, vm.versions.get(v, 0))) for v, n in c['base_versions'].items()}
            vm.submit_compiled(cw.compile_spec(spec, base_versions=base, data_dir=root / c['data_dir']))
            vm.wait(timeout)
            done.add(adm['name'])
            order.append(adm['name'])
            if after_each:
                after_each(adm, vm)
            log.info('ran %s -> versions %s', adm['name'], vm.versions)
    finally:
        vm.shutdown()
    vm.skipped = skipped
    return vm, order


def main(argv=None):
    p = argparse.ArgumentParser(prog='kvmfe.run_manifest', description=__doc__)
    p.add_argument('dir')
    p.add_argument('--allow-opaque', action='store_true')
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    vm, order = run_manifest(a.dir, a.allow_opaque)
    print(json.dumps({'order': order, 'versions': vm.versions}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
