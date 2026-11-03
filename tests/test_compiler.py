import contextlib
import copy
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import abi
import compile_workload as cw
import ref_ops
from compile_workload import CompileError, compile_spec

TASK_FIELDS = ('op_id', 'succ_off', 'succ_cnt', 'indegree', 'priority', 'deadline_us', 'args_off', 'args_words',
               'wait_var', 'wait_ver', 'pub_var', 'pub_ver', 'workload_id', 'flags', 'reserved0', 'reserved1')


FIXTURES = ROOT / 'tests' / 'fixtures'


def all_ops():
    """Every registry (core from include/, gpt/game/comm from tests/fixtures), loaded like the executor does."""
    import shutil
    import tempfile
    global _ALL_OPS
    if _ALL_OPS is None:
        d = Path(tempfile.mkdtemp(prefix='kvm-ops-'))
        shutil.copy(abi.INCLUDE_DIR / 'kvm_ops.def', d)
        for f in FIXTURES.glob('kvm_ops_*.def'):
            shutil.copy(f, d)
        _ALL_OPS = abi.load_ops(d)
        shutil.rmtree(d)
    return _ALL_OPS


_ALL_OPS = None


def EXAMPLE():
    return json.loads((ROOT / 'examples' / 'live_training_inference.json').read_text())


def task(tid, op='noop', args=None, **kw):
    return dict(id=tid, op=op, **({'args': args} if args is not None else {}), **kw)


def wl(tasks, variables=(), buffers=None, graphs=None):
    return {'variables': list(variables),
            'buffers': buffers if buffers is not None else [{'name': 'a', 'rows': 2, 'cols': 2}, {'name': 'b'}],
            'graphs': graphs or [{'name': 'g', 'tasks': tasks}]}


def pub(tid, var, after=(), **kw):
    return task(tid, after=list(after), publishes=var, **kw)


def read(tid, var, policy, version, **kw):
    return task(tid, reads=[{'var': var, 'policy': policy, 'version': version}], **kw)


def descriptors(c):
    size = abi.load_task_abi().size
    return [dict(zip(TASK_FIELDS, abi.load_task_abi().layout.unpack_from(c.tasks, i * size)))
            for i in range(c.metadata['task_count'])]


class AbiTests(unittest.TestCase):
    def test_header_layout_is_64_bytes_and_in_declared_order(self):
        a = abi.load_task_abi()
        self.assertEqual(a.size, 64)
        self.assertEqual(a.fields, TASK_FIELDS)
        self.assertEqual((a.version, a.no_var), (2, -1))

    def test_ops_parsed_from_registry(self):
        ops = abi.load_ops()
        self.assertEqual(ops['matmul'].id, 8)
        self.assertEqual([(x.name, x.type) for x in ops['scale'].args],
                         [('x', 'buf'), ('out', 'buf'), ('alpha', 'f32'), ('n', 'i32')])

    def test_every_registered_op_has_reference_semantics(self):
        self.assertLessEqual(set(abi.load_ops()), set(ref_ops.OPS))
        self.assertLessEqual(set(all_ops()), set(ref_ops.OPS))


class LayoutTests(unittest.TestCase):
    def diamond(self):
        return wl([task('s'), task('l', after=['s'], priority=10), task('r', after=['s'], deadline_us=50),
                   task('e', after=['l', 'r'])])

    def test_diamond_descriptors_and_csr(self):
        c = compile_spec(self.diamond())
        d = descriptors(c)
        self.assertEqual(len(c.tasks), 4 * 64)
        self.assertEqual([x['indegree'] for x in d], [0, 1, 1, 2])
        self.assertEqual([x['succ_cnt'] for x in d], [2, 1, 1, 0])
        self.assertEqual(struct.unpack('<4i', c.successors), (1, 2, 3, 3))
        self.assertEqual((d[1]['priority'], d[1]['deadline_us'], d[2]['deadline_us']), (10, -1, 50))
        self.assertEqual([x['wait_var'] for x in d], [-1] * 4)
        self.assertTrue(all(x['flags'] == x['reserved0'] == x['reserved1'] == 0 for x in d))
        self.assertEqual(c.metadata['initial_ready'], [0])
        self.assertEqual((c.metadata['abi_version'], c.metadata['edge_count']), (2, 4))

    def test_typed_args_are_bitcast_words(self):
        spec = wl([task('t', 'scale', {'x': 'a', 'out': 'b', 'alpha': 0.5, 'n': 1}),
                   task('u', 'copy', {'src': 'a', 'dst': 'a', 'n': 4})])
        c = compile_spec(spec)
        words = struct.unpack(f'<{c.metadata["args_words"]}i', c.args)
        self.assertEqual(words[:4], (0, 1, struct.unpack('<i', struct.pack('<f', 0.5))[0], 1))
        self.assertEqual(words[4:], (0, 0, 4))
        d = descriptors(c)
        self.assertEqual([(x['args_off'], x['args_words']) for x in d], [(0, 4), (4, 3)])
        self.assertEqual(d[0]['op_id'], abi.load_ops()['scale'].id)

    def test_buffer_table_and_init_metadata(self):
        spec = wl([task('t')], buffers=[{'name': 'w', 'rows': 1, 'cols': 3, 'init': {'kind': 'values', 'values': [1, 2, 0.1]}},
                                        {'name': 's'}])
        m = compile_spec(spec).metadata
        self.assertEqual([(b['name'], b['index'], b['rows'], b['cols']) for b in m['buffers']], [('w', 0, 1, 3), ('s', 1, 1, 1)])
        self.assertEqual(m['buffers'][0]['init']['values'], [1.0, 2.0, ref_ops.f32(0.1)])
        self.assertEqual(m['buffers'][1]['init'], {'kind': 'fill', 'value': 0})

    def test_determinism_byte_identical_bundles(self):
        spec = EXAMPLE()
        outs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                cw.write_bundle(compile_spec(copy.deepcopy(spec)), d)
                outs.append({p.name: p.read_bytes() for p in Path(d).iterdir()})
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(sorted(outs[0]), ['args.bin', 'metadata.json', 'successors.bin', 'tasks.bin'])

    def test_metadata_matches_binary_sizes(self):
        c = compile_spec(EXAMPLE())
        m = c.metadata
        self.assertEqual((len(c.tasks), len(c.successors), len(c.args)), (64 * m['task_count'], 4 * m['edge_count'], 4 * m['args_words']))
        self.assertEqual(len(m['task_names']), m['task_count'])


class SchemaErrorTests(unittest.TestCase):
    def fails(self, spec, *needles):
        with self.assertRaises(CompileError) as cm:
            compile_spec(spec)
        for n in needles:
            self.assertIn(n, str(cm.exception))

    def one(self, op, args, **kw):
        return wl([task('t', op, args)], **kw)

    def test_unknown_op(self):
        self.fails(self.one('doom', {}), 'g/t', 'unknown op', 'doom')

    def test_missing_arg(self):
        self.fails(self.one('copy', {'src': 'a', 'dst': 'b'}), 'g/t', "missing arg(s) ['n']")

    def test_extra_arg(self):
        self.fails(self.one('relu', {'x': 'a', 'out': 'b', 'n': 1, 'bogus': 1}), "no arg(s) ['bogus']")

    def test_mistyped_args(self):
        base = {'src': 'a', 'dst': 'b', 'n': 1}
        for key, bad, needle in [('n', '4', 'expected an integer'), ('n', 1.5, 'expected an integer'), ('n', True, 'expected an integer'),
                                 ('src', 3, 'expected a buffer name'), ('src', 'nope', 'unknown buffer')]:
            with self.subTest(key=key, bad=bad):
                self.fails(self.one('copy', {**base, key: bad}), "arg '%s'" % key, needle)
        self.fails(self.one('scale', {'x': 'a', 'out': 'b', 'alpha': 'x', 'n': 1}), "arg 'alpha'", 'expected a number')

    def test_int32_and_range_checks(self):
        self.fails(self.one('copy', {'src': 'a', 'dst': 'b', 'n': 2**31}), 'overflows int32')
        self.fails(self.one('copy', {'src': 'a', 'dst': 'b', 'n': 0}), 'below the minimum 1')
        self.fails(self.one('spin', {'us': -1}), 'below the minimum 0')
        compile_spec(self.one('spin', {'us': 2**31 - 1}))

    def test_f32_overflow_nan_inf(self):
        a = {'x': 'a', 'out': 'b', 'n': 1}
        self.fails(self.one('scale', {**a, 'alpha': 1e39}), 'overflows float32')
        self.fails(self.one('scale', {**a, 'alpha': float('nan')}), 'non-finite')
        self.fails(self.one('scale', {**a, 'alpha': float('inf')}), 'non-finite')
        c = compile_spec(self.one('scale', {**a, 'alpha': float('inf')}), allow_nonfinite=True)
        self.assertEqual(struct.unpack('<f', c.args[8:12])[0], float('inf'))

    def test_buffer_extent_checked_at_compile_time(self):
        self.fails(self.one('copy', {'src': 'a', 'dst': 'b', 'n': 4}), "buffer 'b'", 'needs 4 element(s)')
        self.fails(self.one('matmul', {'a': 'a', 'b': 'a', 'out': 'a', 'm': 2, 'n': 2, 'k': 3}), "arg 'a'", 'needs 6 element(s)')

    def test_buffer_declaration_errors(self):
        for buffers, needle in [([{'name': 'a'}, {'name': 'a'}], 'duplicate buffer'),
                                ([{'name': 'a', 'rows': 0}], 'rows must be'),
                                ([{'name': 'a', 'rows': 2, 'init': {'kind': 'values', 'values': [1]}}], 'list of 2'),
                                ([{'name': 'a', 'init': {'kind': 'zeros'}}], "'fill', 'values', 'seeded', 'file'"),
                                ([{'name': 'a', 'dtype': 'bf16'}], 'dtype must be one of'),
                                ([{'name': 'a', 'layout': 'x'}], 'unknown key')]:
            with self.subTest(needle=needle):
                self.fails(wl([task('t')], buffers=buffers), needle)

    def test_unknown_keys_reject_old_schema(self):
        self.fails({'state': {}, 'graphs': wl([task('t')])['graphs']}, "unknown key(s) ['state']")
        self.fails(wl([task('t', deadline=3)]), "unknown key(s) ['deadline']")

    def test_priority_and_deadline(self):
        self.fails(wl([task('t', deadline_us=1.5)]), 'deadline_us')
        self.fails(wl([task('t', deadline_us=-2)]), 'deadline_us')
        self.fails(wl([task('t', priority=2**31)]), 'priority')

    def test_empty_workload(self):
        self.fails({'graphs': []}, 'nonempty')


class GraphErrorTests(unittest.TestCase):
    def fails(self, tasks, *needles, **kw):
        with self.assertRaises(CompileError) as cm:
            compile_spec(wl(tasks, **kw))
        for n in needles:
            self.assertIn(n, str(cm.exception))

    def test_cycle_names_tasks(self):
        self.fails([task('a', after=['c']), task('b', after=['a']), task('c', after=['b'])], 'dependency cycle', 'g/a')

    def test_self_dependency(self):
        self.fails([task('a', after=['a'])], 'depends on itself')

    def test_duplicate_edge(self):
        self.fails([task('a'), task('b', after=['a', 'a'])], 'g/b', 'duplicate dependency')

    def test_duplicate_task_and_graph(self):
        self.fails([task('a'), task('a')], 'duplicate task id')
        with self.assertRaises(CompileError):
            compile_spec(wl(None, graphs=[{'name': 'g', 'tasks': [task('a')]}, {'name': 'g', 'tasks': [task('a')]}]))

    def test_unknown_dependency(self):
        self.fails([task('a', after=['zzz'])], 'unknown task', 'zzz')

    def test_cross_graph_task_ids_independent(self):
        c = compile_spec(wl(None, graphs=[{'name': 'g', 'tasks': [task('a')]}, {'name': 'h', 'tasks': [task('a', after=[])]}]))
        self.assertEqual(c.metadata['task_names'], ['g/a', 'h/a'])
        self.assertEqual([d['workload_id'] for d in descriptors(c)], [0, 1])


class VersioningTests(unittest.TestCase):
    def comp(self, tasks, base=None, variables=('w',)):
        return compile_spec(wl(tasks, variables), base)

    def fails(self, tasks, *needles, base=None, variables=('w',)):
        with self.assertRaises(CompileError) as cm:
            self.comp(tasks, base, variables)
        for n in needles:
            self.assertIn(n, str(cm.exception))

    def test_publish_versions_follow_program_order(self):
        c = self.comp([pub('p1', 'w'), pub('p2', 'w', ['p1']), pub('p3', 'w', ['p2'])])
        d = descriptors(c)
        self.assertEqual([(x['pub_var'], x['pub_ver']) for x in d], [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(c.metadata['final_versions'], {'w': 3})
        self.assertEqual([p['version'] for p in c.metadata['publishers']], [1, 2, 3])

    def test_base_versions_shift_and_first_publisher_orders_after_live_head(self):
        c = self.comp([pub('p1', 'w'), pub('p2', 'w', ['p1'])], base={'w': 5})
        d = descriptors(c)
        self.assertEqual([x['pub_ver'] for x in d], [6, 7])
        self.assertEqual((d[0]['wait_var'], d[0]['wait_ver']), (0, 5))     # not before the earlier batch's v5
        self.assertEqual(d[1]['wait_var'], -1)
        self.assertEqual(c.metadata['variables'], [{'name': 'w', 'id': 0, 'base_version': 5}])
        self.assertEqual(c.metadata['reads'], [])

    def test_unordered_publishers(self):
        self.fails([pub('p1', 'w'), pub('p2', 'w')], 'unordered publishers', 'g/p1', 'g/p2')

    def test_publishers_ordered_through_a_longer_path_and_other_vars_independent(self):
        self.comp([pub('p1', 'w'), task('m', after=['p1']), pub('p2', 'w', ['m'])])
        self.comp([pub('p1', 'w'), pub('q1', 'v')], variables=('w', 'v'))

    def test_undeclared_variable(self):
        self.fails([pub('p', 'nope')], 'undeclared variable')
        self.fails([read('r', 'nope', 'exact', 0)], 'undeclared variable')

    def test_bad_read_specs(self):
        self.fails([read('r', 'w', 'sometimes', 0)], 'policy must be')
        self.fails([read('r', 'w', 'exact', -1)], 'version must be')
        self.fails([task('r', reads=[{'var': 'w', 'policy': 'exact', 'version': 0}] * 2)], "reads variable 'w' twice")
        self.fails([task('r', reads=[{'var': 'w', 'policy': 'exact'}])], 'missing key')

    def test_at_least_sets_wait_slot(self):
        c = self.comp([pub('p1', 'w'), pub('p2', 'w', ['p1']), read('r', 'w', 'at_least', 2)])
        r = descriptors(c)[2]
        self.assertEqual((r['wait_var'], r['wait_ver']), (0, 2))
        self.assertEqual(c.metadata['edge_count'], 1)                       # at_least adds no edge
        self.assertEqual(c.metadata['reads'][0]['observed'], None)

    def test_exact_adds_write_after_read_edge_and_reports_observed(self):
        c = self.comp([pub('p1', 'w'), pub('p2', 'w', ['p1']), read('r', 'w', 'exact', 1)])
        names = c.metadata['task_names']
        r, p2 = names.index('g/r'), names.index('g/p2')
        self.assertIn(p2, struct.unpack_from(f'<{c.metadata["edge_count"]}i', c.successors)[
            descriptors(c)[r]['succ_off']:][:descriptors(c)[r]['succ_cnt']])
        self.assertEqual(descriptors(c)[p2]['indegree'], 2)
        self.assertEqual(c.metadata['reads'][0]['observed'], 1)
        self.assertEqual((descriptors(c)[r]['wait_var'], descriptors(c)[r]['wait_ver']), (0, 1))

    def test_exact_read_of_last_version_needs_no_edge(self):
        c = self.comp([pub('p1', 'w'), read('r', 'w', 'exact', 1)])
        self.assertEqual(c.metadata['edge_count'], 0)

    def test_exact_on_base_version(self):
        c = self.comp([pub('p1', 'w'), read('r', 'w', 'exact', 4)], base={'w': 4})
        self.assertEqual(c.metadata['edge_count'], 1)
        self.assertEqual(c.metadata['reads'][0]['observed'], 4)

    def test_stale_exact_read(self):
        self.fails([read('r', 'w', 'exact', 1)], 'stale', 'g/r', base={'w': 3})
        self.fails([read('r', 'w', 'exact', 1), pub('p', 'w')], 'stale', base={'w': 2})

    def test_unsatisfiable_versions(self):
        self.fails([pub('p1', 'w'), read('r', 'w', 'exact', 2)], 'never be satisfied')
        self.fails([read('r', 'w', 'at_least', 1)], 'never be satisfied')

    def test_exact_after_next_publisher_is_impossible(self):
        self.fails([pub('p1', 'w'), pub('p2', 'w', ['p1']),
                    task('r', after=['p2'], reads=[{'var': 'w', 'policy': 'exact', 'version': 1}])],
                   'can never hold', 'g/r', 'g/p2')

    def test_read_that_precedes_its_publisher_deadlocks(self):
        self.fails([read('r', 'w', 'at_least', 1), pub('p', 'w', ['r'])], 'deadlock')
        self.fails([task('p', publishes='w', reads=[{'var': 'w', 'policy': 'at_least', 'version': 1}])], 'deadlock')

    def test_read_and_publish_in_one_task_needs_no_gate(self):
        c = self.comp([pub('p1', 'w'), task('p2', after=['p1'], publishes='w',
                                            reads=[{'var': 'w', 'policy': 'exact', 'version': 1}])])
        self.assertEqual(c.metadata['task_count'], 2)
        d = descriptors(c)[1]
        self.assertEqual((d['wait_var'], d['wait_ver'], d['pub_var'], d['pub_ver']), (0, 1, 0, 2))

    def test_two_reads_insert_one_gate_task(self):
        spec = wl([pub('pw', 'w'), pub('pv', 'v'),
                   task('r', op='noop', reads=[{'var': 'w', 'policy': 'at_least', 'version': 1},
                                               {'var': 'v', 'policy': 'exact', 'version': 1}], priority=7, deadline_us=9)],
                  variables=('w', 'v'))
        c = compile_spec(spec)
        m = c.metadata
        self.assertEqual(m['task_names'], ['g/pw', 'g/pv', '__gate/g/r/1', 'g/r'])
        d = descriptors(c)
        gate, reader = d[2], d[3]
        self.assertEqual((gate['op_id'], gate['wait_var'], gate['wait_ver'], gate['pub_var']), (0, 1, 1, -1))
        self.assertEqual((reader['wait_var'], reader['wait_ver'], reader['indegree']), (0, 1, 1))
        self.assertEqual((gate['priority'], gate['deadline_us']), (7, 9))
        self.assertEqual(struct.unpack_from('<i', c.successors, gate['succ_off'] * 4)[0], 3)   # gate -> reader
        self.assertEqual([x['gate'] for x in m['reads']], [None, 2])
        self.assertIn(2, m['initial_ready'])

    def test_three_reads_two_gates(self):
        reads = [{'var': v, 'policy': 'at_least', 'version': 1} for v in 'abc']
        spec = wl([pub('pa', 'a'), pub('pb', 'b'), pub('pc', 'c'), task('r', reads=reads)], variables='abc')
        self.assertEqual(sum(n.startswith('__gate/') for n in compile_spec(spec).metadata['task_names']), 2)

    def test_exact_edge_goes_from_the_real_reader_not_the_gate(self):
        reads = [{'var': 'w', 'policy': 'at_least', 'version': 1}, {'var': 'v', 'policy': 'exact', 'version': 1}]
        spec = wl([pub('pw', 'w'), pub('pv', 'v'), pub('pv2', 'v', ['pv']), task('r', reads=reads)], variables=('w', 'v'))
        c = compile_spec(spec)
        m = c.metadata
        names = m['task_names']
        d = descriptors(c)
        out = struct.unpack(f'<{m["edge_count"]}i', c.successors)
        reader = names.index('g/r')
        self.assertIn(names.index('g/pv2'), out[d[reader]['succ_off']:][:d[reader]['succ_cnt']])

    def test_base_versions_must_name_declared_integers(self):
        with self.assertRaises(CompileError):
            compile_spec(wl([task('t')], ('w',)), {'zzz': 1})
        with self.assertRaises(CompileError):
            compile_spec(wl([task('t')], ('w',)), {'w': -1})

    def test_parse_base_versions(self):
        self.assertEqual(cw.parse_base_versions('a=1,b=22'), {'a': 1, 'b': 22})
        for bad in ('a', 'a=x', 'a=1,a=2'):
            with self.assertRaises(CompileError):
                cw.parse_base_versions(bad)


class ExampleAndCliTests(unittest.TestCase):
    def run_cli(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            import logging
            h = logging.StreamHandler(err)
            logging.getLogger('kvm').addHandler(h)
            try:
                code = cw.main(list(map(str, argv)))
            finally:
                logging.getLogger('kvm').removeHandler(h)
        return code, err.getvalue()

    def test_live_example_structure(self):
        c = compile_spec(EXAMPLE())
        reads = {r['name']: r for r in c.metadata['reads']}
        self.assertEqual((reads['serve/snapshot']['policy'], reads['serve/snapshot']['observed']), ('exact', 1))
        self.assertEqual((reads['serve/answer_fresh']['policy'], reads['serve/answer_fresh']['observed']), ('at_least', None))
        names = c.metadata['task_names']
        snap = descriptors(c)[names.index('serve/snapshot')]
        succ = struct.unpack(f'<{c.metadata["edge_count"]}i', c.successors)[snap['succ_off']:][:snap['succ_cnt']]
        self.assertIn(names.index('train/upd2'), succ)                      # update 2 may not overwrite the snapshot

    def test_cli_writes_bundle_and_golden(self):
        with tempfile.TemporaryDirectory() as d:
            code, _ = self.run_cli(ROOT / 'examples' / 'live_training_inference.json', d, '--emit-golden')
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()),
                             ['args.bin', 'golden.json', 'metadata.json', 'successors.bin', 'tasks.bin'])
            golden = json.loads((Path(d) / 'golden.json').read_text())
            self.assertEqual(golden['versions'], {'weight': 2})
            self.assertEqual(golden['buffers']['snap_out'], [5.25])
            self.assertEqual({r['task']: r['observed'] for r in golden['reads']}, {'serve/snapshot': 1, 'serve/answer_fresh': 2})

    def test_cli_base_versions_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / 'w.json'
            spec.write_text(json.dumps(wl([pub('p', 'w')], ('w',))))
            code, _ = self.run_cli(spec, Path(d) / 'out', '--base-versions', 'w=7')
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((Path(d) / 'out' / 'metadata.json').read_text())['publishers'][0]['version'], 8)
            spec.write_text(json.dumps(wl([task('t', 'doom')])))
            code, log = self.run_cli(spec, Path(d) / 'out2')
            self.assertEqual(code, 1)
            self.assertIn('unknown op', log)
            self.assertFalse((Path(d) / 'out2').exists())


if __name__ == '__main__':
    unittest.main()
