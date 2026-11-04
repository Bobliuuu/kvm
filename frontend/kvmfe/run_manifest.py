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
    done, pending, order = set(), list(m['admissions']), []
    try:
        while pending:
            adm = next((a for a in pending if eligible(a, done, vm.versions)), None)
            if adm is None:
                raise FrontendError(f"no admission can start: pending {[a['name'] for a in pending]}, versions {vm.versions}")
            c = adm['compile']
            spec = json.loads((root / c['workload']).read_text())
            base = {v: (n if v in adm['publishes'] else max(n, vm.versions.get(v, 0))) for v, n in c['base_versions'].items()}
            vm.submit_compiled(cw.compile_spec(spec, base_versions=base, data_dir=root / c['data_dir']))
            vm.wait(timeout)
            done.add(adm['name'])
            pending.remove(adm)
            order.append(adm['name'])
            if after_each:
                after_each(adm, vm)
            log.info('ran %s -> versions %s', adm['name'], vm.versions)
    finally:
        vm.shutdown()
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
