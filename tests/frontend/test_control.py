"""Device-side control (predicated tasks, pred_cmp, bounded while-loops, branches) on the CPU golden paths: the frontend float64 oracle in
program and random dataflow order, and the compiler + CPU runtime VM (the executor-semantics reference, with the real ABI encoding)."""
import json
import struct
import sys
import unittest
from pathlib import Path

import _path  # noqa: F401
from kvmfe import refvm
from kvmfe.autodiff import backward
from kvmfe.emit import GraphPolicy, lower_program, to_workload
from kvmfe.ir import Builder, FrontendError
from kvmfe.lower import Tiling

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime'))
import compile_workload as cw  # noqa: E402
import workload_vm  # noqa: E402

CPU = Tiling.preset('cpu')


def lowered(b):
    return to_workload(lower_program(b.ops, CPU, 'none', 0), {'main': GraphPolicy()})


def halving(v, max_iters=6, thr=1.0):
    """while x > thr: x *= 0.5 (at most max_iters iterations); x lives in a pinned 1x1 state tensor initialised to v."""
    b = Builder()
    x = b.tensor('x', 1, 1, 'state', buf='x', init=('values', [v]))
    fl = b.while_loop(max_iters, lambda bb: (x, 0, '>', thr), lambda bb, i: bb.scale_scalar(x, 0.5))
    return b, fl


def expected_halving(v, max_iters=6, thr=1.0):
    n = 0
    while n < max_iters and v > thr:
        v, n = v * 0.5, n + 1
    return v, n


def branch(vals):
    """y = 2y if sum(u) > 0 else -y, y starts at 3; the sum is computed on the device."""
    b = Builder()
    u = b.tensor('u', len(vals), 1, 'state', buf='u', init=('values', vals))
    y = b.tensor('y', 1, 1, 'state', buf='y', init=('values', [3.0]))
    s = b.colsum(u)
    p = b.pred_cmp(s, '>', 0.0)
    with b.when(p):
        b.scale_scalar(y, 2.0)
    with b.when(p, negate=True):
        b.scale_scalar(y, -1.0)
    return b


class Loop(unittest.TestCase):
    def test_trip_count_follows_the_data_with_one_program(self):
        specs = {}
        for v in (10.0, 0.5, 1.0, 1e6, 3.0):
            b, fl = halving(v)
            spec = lowered(b)
            specs[v] = spec
            want, trips = expected_halving(v)
            for order, seed in (('program', 0), ('random', 1), ('random', 2)):
                mem = refvm.execute(spec, order=order, seed=seed)
                self.assertEqual(mem.bufs['x'].a[0], want, (v, order))
                flags = [int(f) for f in mem.bufs[fl.buf].a]
                self.assertEqual(sum(flags[:-1]), trips, (v, flags))
                self.assertEqual(flags[-1], int(trips == 6 and want > 1.0))
        # the task graph is identical for every input: only buffer contents differ
        shapes = {json.dumps([g['tasks'] for g in s['graphs']], sort_keys=True) for s in specs.values()}
        self.assertEqual(len(shapes), 1)
        self.assertEqual({e['op'] for g in specs[10.0]['graphs'] for e in g['tasks'] if 'when' in e}, {'scale_flat'})

    def test_compiler_abi_and_cpu_runtime_vm(self):
        words, argwords = {}, set()
        for v in (10.0, 0.5, 40.0):
            b, fl = halving(v, max_iters=4)
            comp = cw.compile_spec(lowered(b))
            words[v] = comp.tasks
            argwords.add(comp.args)
            golden = workload_vm.run_golden(comp)
            want, trips = expected_halving(v, 4)
            self.assertEqual(golden['buffers']['x'][0], want)
            self.assertEqual(sum(golden['buffers'][fl.buf][:-1]), trips)
            if v == 10.0:
                abi = cw.abi.load_task_abi()
                recs = [dict(zip(abi.fields, abi.layout.unpack_from(comp.tasks, i * abi.size))) for i in range(len(comp.program))]
                pred = [r for r in recs if r['flags']]
                self.assertEqual(len(pred), 4)
                for r in pred:
                    self.assertEqual(r['flags'] & 3, 1)
                    self.assertEqual(comp.buffers[r['reserved1']].name, fl.buf)
                self.assertEqual(sorted(r['flags'] >> 8 for r in pred), [0, 1, 2, 3])
        self.assertEqual(len(set(words.values())), 1, 'tasks.bin must not depend on the data')
        self.assertEqual(len(argwords), 1, 'args.bin must not depend on the data')

    def test_cap_reached_flag_and_zero_iterations(self):
        b, fl = halving(1e9, max_iters=3)
        mem = refvm.execute(lowered(b))
        self.assertEqual([int(f) for f in mem.bufs[fl.buf].a], [1, 1, 1, 1])
        b, fl = halving(0.0, max_iters=3)
        mem = refvm.execute(lowered(b))
        self.assertEqual([int(f) for f in mem.bufs[fl.buf].a], [0, 0, 0, 0])


class Branch(unittest.TestCase):
    def test_exactly_one_arm_runs(self):
        for vals, want in (([1.0, 2.0, -0.5, 0.25], 6.0), ([-1.0, -2.0, 0.5, 0.25], -3.0), ([0.0, 0.0], -3.0)):
            spec = lowered(branch(vals))
            for order, seed in (('program', 0), ('random', 3), ('random', 4)):
                self.assertEqual(refvm.execute(spec, order=order, seed=seed).bufs['y'].a[0], want, (vals, order))
            comp = cw.compile_spec(spec)
            self.assertEqual(workload_vm.run_golden(comp)['buffers']['y'][0], want)

    def test_two_arms_share_one_task_graph(self):
        a, b = (json.dumps(lowered(branch(v))['graphs'], sort_keys=True) for v in ([1.0, 2.0], [-1.0, -2.0]))
        self.assertEqual(a, b)

    def test_or_and_combination(self):
        b = Builder()
        x = b.tensor('x', 1, 2, 'state', buf='x', init=('values', [5.0, -5.0]))
        y = b.tensor('y', 1, 1, 'state', buf='y', init=('values', [1.0]))
        fl = b.flags('p', 3)
        b.pred_cmp(x, '>', 0.0, 0, out=fl, out_index=0)
        b.pred_cmp(x, '>', 0.0, 1, out=fl, out_index=1)
        b.pred_cmp(x, '>=', 5.0, 0, out=fl, out_index=2, prev=fl, prev_index=1, comb='or')     # (x0 >= 5) OR flag1 -> true
        with b.when(fl, 2):
            b.scale_scalar(y, 7.0)
        with b.when(fl, 1):
            b.scale_scalar(y, 100.0)                                                          # flag1 is false: skipped
        mem = refvm.execute(lowered(b), order='random', seed=9)
        self.assertEqual((mem.bufs['y'].a[0], [int(v) for v in mem.bufs['p'].a]), (7.0, [1, 0, 1]))


class Refusals(unittest.TestCase):
    def test_frontend(self):
        b = Builder()
        p = b.flags('p', 2)
        x = b.tensor('x', 1, 1, 'state', buf='x')
        with b.when(p, 0):
            with self.assertRaisesRegex(FrontendError, 'cannot nest'):
                with b.when(p, 1):
                    pass
        with self.assertRaisesRegex(FrontendError, 'i32 tensor'):
            b.when(x)
        with self.assertRaisesRegex(FrontendError, 'index inside'):
            b.when(p, 2)
        with self.assertRaisesRegex(FrontendError, 'kind in'):
            b.pred_cmp(x, '~', 0.0)
        with self.assertRaisesRegex(FrontendError, 'prev exactly'):
            b.pred_cmp(x, '<', 0.0, comb='and')

    def test_no_autodiff_through_predicated_ops(self):
        b = Builder()
        w = b.tensor('w', 2, 2, 'param', buf='w', rg=True, init=('fill', 1.0))
        x = b.tensor('x', 2, 2, 'input', buf='xin', init=('fill', 1.0))
        p = b.flags('p', 1, 1)
        with b.when(p):
            out = b.linear(x, w)
        with self.assertRaisesRegex(FrontendError, 'predicated'):
            backward(b, [out], list(b.ops), [w])

    def test_compiler_validation(self):
        spec = lowered(halving(4.0)[0])
        def with_when(w):
            s = json.loads(json.dumps(spec))
            next(e for g in s['graphs'] for e in g['tasks'] if 'when' in e)['when'] = w
            return s
        for bad, msg in (({'buf': 'x', 'index': 0}, 'must be i32'), ({'buf': 'nope', 'index': 0}, 'unknown buffer'),
                         ({'buf': 'loop', 'index': 99}, 'outside buffer'), ({'buf': 'loop', 'index': 0, 'negate': 1}, 'boolean'),
                         ({'buf': 'loop'}, 'missing')):
            with self.assertRaisesRegex(cw.CompileError, msg):
                cw.compile_spec(with_when(bad))


if __name__ == '__main__':
    unittest.main()
