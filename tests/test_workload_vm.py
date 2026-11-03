import contextlib
import io
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import abi
import compile_workload as cw
import ref_ops
import workload_vm as vm
from test_compiler import EXAMPLE, pub, read, task, wl


def run_one(op, args, buffers):
    """Run a single op on the VM and return all buffers."""
    spec = {'buffers': buffers, 'graphs': [{'name': 'g', 'tasks': [task('t', op, args)]}]}
    return vm.run_spec(spec, deterministic=True)[0].buffers


def vals(name, *v):
    return {'name': name, 'rows': 1, 'cols': len(v), 'init': {'kind': 'values', 'values': list(v)}}


class OpSemanticsTests(unittest.TestCase):
    def test_every_registered_op_runs(self):
        covered = {'noop', 'copy', 'add', 'scale', 'relu', 'sgd', 'predict', 'mse_grad', 'matmul', 'fill', 'spin'}
        self.assertEqual(covered, set(abi.load_ops()))

    def test_copy_add_scale_relu(self):
        b = [vals('x', -1, 2, -3, 4), vals('y', 1, 1, 1, 1), vals('o', 0, 0, 0, 0)]
        self.assertEqual(run_one('copy', {'src': 'x', 'dst': 'o', 'n': 4}, b)['o'], [-1, 2, -3, 4])
        self.assertEqual(run_one('add', {'a': 'x', 'b': 'y', 'out': 'o', 'n': 4}, b)['o'], [0, 3, -2, 5])
        self.assertEqual(run_one('scale', {'x': 'x', 'out': 'o', 'alpha': 0.5, 'n': 3}, b)['o'], [-0.5, 1, -1.5, 0])
        self.assertEqual(run_one('relu', {'x': 'x', 'out': 'o', 'n': 4}, b)['o'], [0, 2, 0, 4])

    def test_in_place_ops(self):
        out = run_one('relu', {'x': 'x', 'out': 'x', 'n': 3}, [vals('x', -1, 2, -3)])
        self.assertEqual(out['x'], [0, 2, 0])

    def test_sgd_predict_fill(self):
        self.assertEqual(run_one('sgd', {'w': 'w', 'g': 'g', 'lr': 0.5, 'n': 2}, [vals('w', 1, 2), vals('g', 2, 4)])['w'], [0, 0])
        self.assertEqual(run_one('predict', {'x': 'x', 'w': 'w', 'out': 'o', 'n': 3},
                                 [vals('x', 1, 2, 3), vals('w', 2), vals('o', 0, 0, 0)])['o'], [2, 4, 6])
        self.assertEqual(run_one('fill', {'x': 'x', 'value': 7.5, 'n': 2}, [vals('x', 0, 0, 9)])['x'], [7.5, 7.5, 9])

    def test_mse_grad(self):
        b = [vals('p', 1, 3), vals('t', 2, 2), vals('x', 1, 2), vals('l', 0), vals('g', 0)]
        out = run_one('mse_grad', {'pred': 'p', 'target': 't', 'x': 'x', 'loss': 'l', 'grad': 'g', 'n': 2}, b)
        self.assertEqual((out['l'], out['g']), ([1.0], [1.0]))     # mean(1,1); mean(2*-1*1, 2*1*2) = 1

    def test_matmul_row_major_and_aliased_output(self):
        a = {'name': 'a', 'rows': 2, 'cols': 3, 'init': {'kind': 'values', 'values': [1, 2, 3, 4, 5, 6]}}
        b = {'name': 'b', 'rows': 3, 'cols': 2, 'init': {'kind': 'values', 'values': [1, 0, 0, 1, 1, 1]}}
        o = {'name': 'o', 'rows': 2, 'cols': 2}
        out = run_one('matmul', {'a': 'a', 'b': 'b', 'out': 'o', 'm': 2, 'n': 2, 'k': 3}, [a, b, o])
        self.assertEqual(out['o'], [4, 5, 10, 11])

    def test_float32_rounding_each_op(self):
        out = run_one('scale', {'x': 'x', 'out': 'x', 'alpha': 3.0, 'n': 1}, [vals('x', 0.1)])
        self.assertEqual(out['x'], [ref_ops.f32(ref_ops.f32(0.1) * 3.0)])
        self.assertNotEqual(out['x'][0], 0.1 * 3.0)
        huge = run_one('scale', {'x': 'x', 'out': 'x', 'alpha': 2.0, 'n': 1}, [vals('x', 3e38)])
        self.assertEqual(huge['x'], [float('inf')])


class VersionSemanticsTests(unittest.TestCase):
    def test_live_example_golden_values(self):
        compiled = cw.compile_spec(EXAMPLE())
        g = vm.run_golden(compiled)
        self.assertEqual((g['buffers']['w'], g['buffers']['snap_w'], g['buffers']['snap_out'], g['buffers']['fresh_out']),
                         ([2.375], [1.75], [5.25], [7.125]))
        self.assertEqual(g['versions'], {'weight': 2})

    def test_threaded_runs_match_golden_and_observe_exact_versions(self):
        golden = vm.run_golden(cw.compile_spec(EXAMPLE()))['buffers']
        for i in range(25):
            runtime, trace = vm.run_spec(EXAMPLE(), workers=4)
            self.assertEqual(runtime.buffers, golden, f'iteration {i}')
            seen = {e['task']: e['observed'] for e in trace}
            self.assertEqual(seen['serve/snapshot'], {'weight': 1})
            self.assertEqual(seen['serve/answer_fresh'], {'weight': 2})

    def test_exact_reader_blocks_next_publisher(self):
        # slow reader of version 1 must finish before version 2 is published
        order = []
        def slow(b, a):
            order.append('reader-start'); time.sleep(0.05); order.append('reader-end')
        spec = wl([pub('p1', 'w'), pub('p2', 'w', ['p1']),
                   read('r', 'w', 'exact', 1, op='spin', args={'us': 1})], ('w',))
        with mock.patch.dict(ref_ops.OPS, {'spin': ref_ops.RefOp(slow, lambda a: {}, {})}):
            runtime = vm.Runtime(workers=3)
            runtime.submit(spec)
            runtime.start()
            trace = runtime.wait(5)
            runtime.shutdown()
        by = {e['task']: e for e in trace}
        self.assertLessEqual(by['g/r']['finished'], by['g/p2']['started'])
        self.assertEqual(runtime.versions, {'w': 2})

    def test_at_least_reader_waits_for_publisher(self):
        gate = threading.Event()
        def slow(b, a):
            gate.wait(2)
        spec = wl([task('slowpre', 'spin', {'us': 1}), pub('p', 'w', ['slowpre']),
                   read('r', 'w', 'at_least', 1)], ('w',))
        with mock.patch.dict(ref_ops.OPS, {'spin': ref_ops.RefOp(slow, lambda a: {}, {})}):
            runtime = vm.Runtime(workers=2)
            runtime.submit(spec)
            runtime.start()
            time.sleep(0.1)
            self.assertNotIn('g/r', runtime.completed)       # blocked on the version, not on a DAG edge
            gate.set()
            trace = runtime.wait(5)
            runtime.shutdown()
        self.assertEqual({e['task']: e['observed'] for e in trace}['g/r'], {'w': 1})

    def test_gate_task_carries_extra_wait(self):
        reads = [{'var': 'a', 'policy': 'at_least', 'version': 1}, {'var': 'b', 'policy': 'at_least', 'version': 1}]
        spec = wl([pub('pa', 'a'), pub('pb', 'b'), task('r', reads=reads)], ('a', 'b'))
        runtime, trace = vm.run_spec(spec, workers=3)
        order = [e['task'] for e in trace]
        self.assertLess(order.index('__gate/g/r/1'), order.index('g/r'))
        self.assertEqual(len(order), 4)

    def test_trace_cli_prints_observed(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            vm.main([str(ROOT / 'examples' / 'live_training_inference.json'), '--deterministic', '--trace', '--log-level', 'CRITICAL'])
        self.assertIn('serve/snapshot\tobserved={"weight": 1}', out.getvalue())


class DynamicSubmissionTests(unittest.TestCase):
    def test_second_workload_admitted_while_running_continues_versions(self):
        release = threading.Event()
        def blocker(b, a):
            release.wait(5)
        first = wl([task('hold', 'spin', {'us': 1}), pub('p1', 'w', ['hold'])], ('w',),
                   buffers=[{'name': 'a'}])
        second = wl([read('r', 'w', 'at_least', 1, op='fill', args={'x': 'a', 'value': 3.0, 'n': 1}),
                     pub('p2', 'w', ['r'])], ('w',), buffers=[{'name': 'a'}])
        with mock.patch.dict(ref_ops.OPS, {'spin': ref_ops.RefOp(blocker, lambda a: {}, {})}):
            runtime = vm.Runtime(workers=2)
            runtime.start()
            runtime.submit(first)
            c2 = runtime.submit(second)                       # compiled against projected version 1
            self.assertEqual(c2.variables[0].base_version, 1)
            self.assertEqual(c2.variables[0].final_version, 2)
            time.sleep(0.1)
            self.assertEqual(runtime.buffers['a'], [0.0])      # second batch waits for p1 (version order across batches)
            release.set()
            runtime.wait(5)
            runtime.shutdown()
        self.assertEqual((runtime.versions, runtime.buffers['a']), ({'w': 2}, [3.0]))

    def test_exact_read_stale_against_admitted_work_is_rejected(self):
        runtime = vm.Runtime(workers=1)
        runtime.submit(wl([pub('p', 'w')], ('w',)))
        with self.assertRaises(cw.CompileError):
            runtime.submit(wl([read('r', 'w', 'exact', 0)], ('w',)))
        runtime.shutdown()

    def test_buffer_shape_conflict_between_submissions(self):
        runtime = vm.Runtime(workers=1)
        runtime.submit(wl([task('t')], buffers=[{'name': 'a', 'rows': 2}]))
        with self.assertRaises(vm.GraphError):
            runtime.submit(wl([task('t')], buffers=[{'name': 'a', 'rows': 3}]))
        runtime.shutdown()


class FailureTests(unittest.TestCase):
    def test_failure_cancels_descendants_and_version_waiters(self):
        def boom(b, a):
            raise ZeroDivisionError('boom')
        spec = wl([task('bad', 'spin', {'us': 1}), pub('p', 'w', ['bad']), read('r', 'w', 'at_least', 1),
                   task('indep')], ('w',))
        with mock.patch.dict(ref_ops.OPS, {'spin': ref_ops.RefOp(boom, lambda a: {}, {})}):
            runtime = vm.Runtime(workers=2)
            runtime.submit(spec)
            runtime.start()
            with self.assertRaises(RuntimeError) as cm:
                runtime.wait(3)
            runtime.shutdown()
        self.assertIn('boom', str(cm.exception))
        self.assertEqual(len(runtime.completed), 4)
        self.assertEqual(len(set(runtime.completed)), 4)
        self.assertEqual(runtime.versions, {'w': 0})

    def test_exact_violation_is_detected_not_silent(self):
        spec = wl([read('r', 'w', 'exact', 0), pub('p', 'w')], ('w',))
        compiled = cw.compile_spec(spec)
        runtime = vm.Runtime(workers=1, deterministic=True)
        runtime.submit_compiled(compiled)
        runtime.versions['w'] = 1                               # simulate an external publish racing the reader
        runtime.start()
        with self.assertRaises(RuntimeError) as cm:
            runtime.wait(3)
        runtime.shutdown()
        self.assertIn('snapshot violated', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
