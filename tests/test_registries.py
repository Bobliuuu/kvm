"""Multi-registry loading, i32 buffers, optional buffer args, seeded / file initial values."""
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import abi
import compile_workload as cw
import initgen
import ref_ops
import workload_vm as vm
from compile_workload import CompileError, compile_spec
from test_compiler import FIXTURES, all_ops, descriptors, task, wl


def words(c):
    return struct.unpack(f'<{c.metadata["args_words"]}i', c.args)


class RegistryTests(unittest.TestCase):
    def test_sparse_ids_across_registries(self):
        ops = all_ops()
        self.assertEqual((ops['noop'].id, ops['embedding_fwd'].id, ops['doom_tick'].id, ops['put'].id), (0, 32, 128, 64))
        self.assertEqual(len({o.id for o in ops.values()}), len(ops))
        self.assertEqual(list(ops)[:3], ['noop', 'copy', 'add'])               # registry order: core first

    def test_optional_marker_parsed(self):
        e = all_ops()['embedding_fwd']
        self.assertEqual([(a.name, a.optional) for a in e.args if a.type == 'buf'],
                         [('ids', False), ('wte', False), ('wpe', True), ('out', False)])
        self.assertFalse(any(a.optional for a in abi.load_ops()['copy'].args))

    def _dir(self, files):
        """A temporary include directory with `files`, and the executor's real registry list unless `files` brings its own."""
        d = Path(tempfile.mkdtemp(prefix='kvm-reg-'))
        self.addCleanup(shutil.rmtree, d)
        files = {abi.OPS_LIST: (abi.INCLUDE_DIR / abi.OPS_LIST).read_text(), **files}
        for name, text in files.items():
            (d / name).write_text(text)
        return d

    def test_registry_list_is_the_executors(self):
        listed = [line.split('"')[1] for line in (abi.INCLUDE_DIR / abi.OPS_LIST).read_text().splitlines()
                  if line.lstrip().startswith('#include "') and line.rstrip().endswith('.def"')]
        self.assertEqual(list(abi.registry_files()), listed)
        self.assertEqual(listed[0], 'kvm_ops.def')
        self.assertIn('kvm_ops_fmlp.def', listed)            # the fused MLP step's members, which a fixed list once left out

    def test_every_registry_the_executor_includes_is_loaded(self):
        ops = abi.load_ops()
        by_id = {o.id: n for n, o in ops.items()}
        for name in abi.registry_files():
            path = abi.INCLUDE_DIR / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if line.startswith('X('):
                    op, op_id = (x.strip() for x in line[2:].split(',')[:2])
                    with self.subTest(registry=name, op=op):
                        self.assertEqual(by_id.get(int(op_id)), op)
        self.assertEqual((ops['fmlp_slice'].id, ops['fmlp_sgd'].id), (124, 125))

    def test_registry_list_required_and_well_formed(self):
        core = (abi.INCLUDE_DIR / 'kvm_ops.def').read_text()
        for inc, needle in [(None, 'registry list not found'), ('// nothing\n', 'includes no registry file'),
                            ('#include "kvm_ops.def"\n#include "kvm_ops.def"\n', 'included twice')]:
            with self.subTest(inc=inc):
                d = self._dir({'kvm_ops.def': core} if inc is None else {'kvm_ops.def': core, abi.OPS_LIST: inc})
                if inc is None:
                    (d / abi.OPS_LIST).unlink()
                with self.assertRaises(abi.AbiError) as cm:
                    abi.load_ops(d)
                self.assertIn(needle, str(cm.exception))

    def test_cross_registry_duplicates_and_bad_lines_rejected(self):
        core = (abi.INCLUDE_DIR / 'kvm_ops.def').read_text()
        for gpt, needle in [('X(clash, 3, "")\n', 'duplicate operator name or id'),
                            ('X(copy, 99, "")\n', 'duplicate operator name or id'),
                            ('X(bad, 99, "x:f64")\n', 'bad argument'),
                            ('X(bad, 99, "x:buf?, x:buf")\n', 'duplicate argument'),
                            ('X(bad 99)\n', 'unparsable')]:
            with self.subTest(gpt=gpt), self.assertRaises(abi.AbiError) as cm:
                abi.load_ops(self._dir({'kvm_ops.def': core, 'kvm_ops_gpt.def': gpt}))
            self.assertIn(needle, str(cm.exception))

    def test_core_registry_required_others_optional(self):
        with self.assertRaises(abi.AbiError):
            abi.load_ops(self._dir({'kvm_ops_gpt.def': 'X(a, 40, "")\n'}))
        self.assertEqual(set(abi.load_ops(self._dir({'kvm_ops.def': 'X(noop, 0, "")\n'}))), {'noop'})

    def test_real_registries_all_have_reference_entries(self):
        missing = set(abi.load_ops()) - set(ref_ops.OPS)
        self.assertFalse(missing, f'operators without reference semantics: {sorted(missing)}')


class OptionalArgTests(unittest.TestCase):
    EMB = {'ids': 'ids', 'wte': 'wte', 'wpe': None, 'out': 'out', 'r0': 0, 'r1': 2, 'T': 2, 'd': 2}

    def spec(self, args=None, **kw):
        bufs = [{'name': 'ids', 'rows': 2, 'dtype': 'i32', 'init': {'kind': 'values', 'values': [1, 0]}},
                {'name': 'wte', 'rows': 3, 'cols': 2, 'init': {'kind': 'values', 'values': [1, 2, 3, 4, 5, 6]}},
                {'name': 'out', 'rows': 2, 'cols': 2}]
        return wl([task('e', 'embedding_fwd', args or self.EMB)], buffers=bufs, **kw)

    def test_null_encodes_minus_one_word_and_runs(self):
        ops = all_ops()
        c = compile_spec(self.spec(), ops=ops)
        self.assertEqual(words(c)[:4], (0, 1, -1, 2))
        self.assertIsNone(c.program[0].args['wpe'])
        runtime, _ = vm.run_spec(self.spec(), deterministic=True, ops=ops)
        self.assertEqual(runtime.buffers['out'], [3, 4, 1, 2])

    def test_null_for_required_arg_rejected(self):
        with self.assertRaises(CompileError) as cm:
            compile_spec(self.spec({**self.EMB, 'wte': None}), ops=all_ops())
        self.assertIn("'wte'", str(cm.exception))
        self.assertIn('required', str(cm.exception))

    def test_missing_optional_key_is_still_an_error(self):
        args = {k: v for k, v in self.EMB.items() if k != 'wpe'}
        with self.assertRaises(CompileError) as cm:
            compile_spec(self.spec(args), ops=all_ops())
        self.assertIn("missing arg(s) ['wpe']", str(cm.exception))
        self.assertIn('wpe:buf?', str(cm.exception))


class DtypeTests(unittest.TestCase):
    def test_i32_buffer_metadata_and_values(self):
        spec = wl([task('t')], buffers=[{'name': 'i', 'rows': 1, 'cols': 3, 'dtype': 'i32',
                                          'init': {'kind': 'values', 'values': [1, 2.0, -3]}}, {'name': 'f'}])
        c = compile_spec(spec)
        self.assertEqual([b['dtype'] for b in c.metadata['buffers']], [1, 0])
        self.assertEqual(c.metadata['buffers'][0]['init']['values'], [1, 2, -3])
        self.assertTrue(all(type(v) is int for v in c.metadata['buffers'][0]['init']['values']))

    def test_i32_rejects_fraction_and_overflow(self):
        for v in (1.5, 2**31, 'x'):
            with self.subTest(v=v), self.assertRaises(CompileError):
                compile_spec(wl([task('t')], buffers=[{'name': 'i', 'dtype': 'i32', 'init': {'kind': 'fill', 'value': v}}]))

    def test_op_dtype_contract(self):
        ops = all_ops()
        base = [{'name': 'src', 'rows': 2, 'cols': 2}, {'name': 'dst', 'rows': 2, 'cols': 2}]
        args = {'src': 'src', 'idx': 'idx', 'dst': 'dst', 'r0': 0, 'r1': 2, 'd': 2}
        good = base + [{'name': 'idx', 'rows': 2, 'dtype': 'i32'}]
        compile_spec(wl([task('g', 'gather_rows', args)], buffers=good), ops=ops)
        bad = base + [{'name': 'idx', 'rows': 2}]
        with self.assertRaises(CompileError) as cm:
            compile_spec(wl([task('g', 'gather_rows', args)], buffers=bad), ops=ops)
        self.assertIn("needs a i32 buffer, 'idx' is f32", str(cm.exception))
        with self.assertRaises(CompileError):                                  # core op on an i32 buffer
            compile_spec(wl([task('c', 'copy', {'src': 'i', 'dst': 'f', 'n': 1})],
                            buffers=[{'name': 'i', 'dtype': 'i32'}, {'name': 'f'}]))

    def test_any_dtype_ops(self):
        ops = all_ops()
        for dt in ('f32', 'i32'):
            runtime, _ = vm.run_spec(wl([task('z', 'zero_buf', {'x': 'x', 'e0': 0, 'e1': 2})], buffers=[
                {'name': 'x', 'rows': 2, 'dtype': dt, 'init': {'kind': 'fill', 'value': 7}}]), deterministic=True, ops=ops)
            self.assertEqual(list(runtime.buffers['x']), [0, 0])
            self.assertEqual(runtime.buffers['x'].dtype, dt)

    def test_buffer_redeclared_with_other_dtype_between_submissions(self):
        runtime = vm.Runtime(workers=1)
        runtime.submit(wl([task('t')], buffers=[{'name': 'a'}]))
        with self.assertRaises(vm.GraphError):
            runtime.submit(wl([task('t')], buffers=[{'name': 'a', 'dtype': 'i32'}]))
        runtime.shutdown()


class SeededInitTests(unittest.TestCase):
    def test_splitmix64_known_answers(self):
        # reference outputs of the published SplitMix64 for state 0 (Vigna)
        self.assertEqual([initgen.draw(0, j) for j in range(3)],
                         [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F])

    def seeded(self, **kw):
        init = {'kind': 'seeded', 'seed': 7, 'scale': 0.5, **kw}
        return wl([task('t')], buffers=[{'name': 'w', 'rows': 4, 'cols': 64, 'init': init}])

    def test_metadata_carries_everything_the_executor_needs(self):
        i = compile_spec(self.seeded(dist='normal')).metadata['buffers'][0]['init']
        self.assertEqual((i['kind'], i['seed'], i['dist'], i['draws'], i['bias']), ('seeded', 7, 'normal', 4, 2.0))
        self.assertEqual(i['coef'], ref_ops.f32(0.5 * 3 ** 0.5))
        u = compile_spec(self.seeded(dist='uniform')).metadata['buffers'][0]['init']
        self.assertEqual((u['draws'], u['bias'], u['coef']), (1, 0.5, 1.0))

    def test_values_match_documented_formula_and_are_reproducible(self):
        c = compile_spec(self.seeded(dist='uniform'))
        a, b = c.buffers[0].materialize(), compile_spec(self.seeded(dist='uniform')).buffers[0].materialize()
        self.assertEqual(a, b)
        u0 = (initgen.draw(7, 0) >> 40) / 2.0 ** 24
        self.assertEqual(a[0], ref_ops.f32((u0 - 0.5) * 1.0))
        self.assertTrue(all(-0.5 <= x < 0.5 for x in a))
        self.assertNotEqual(a, compile_spec(self.seeded(dist='uniform', seed=8)).buffers[0].materialize())

    def test_normal_statistics_and_counter_independence(self):
        n = 20000
        vals = initgen.seeded_values(3, n, *initgen.params('normal', 2.0))
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        self.assertAlmostEqual(mean, 0.0, delta=0.05)
        self.assertAlmostEqual(std, 2.0, delta=0.05)
        head = initgen.seeded_values(3, 10, *initgen.params('normal', 2.0))
        self.assertEqual(vals[:10], head)                                      # element i never depends on n

    def test_vm_initialises_from_seed(self):
        runtime, _ = vm.run_spec(self.seeded(dist='uniform'), deterministic=True)
        self.assertEqual(list(runtime.buffers['w']), compile_spec(self.seeded(dist='uniform')).buffers[0].materialize())

    def test_validation(self):
        for kw, needle in [({'seed': -1}, 'seed must be'), ({'seed': 2**63}, 'seed must be'), ({'seed': 1.5}, 'seed must be'),
                           ({'scale': -1.0}, 'scale must be >= 0'), ({'scale': float('nan')}, 'non-finite'),
                           ({'dist': 'cauchy'}, 'dist must be one of')]:
            with self.subTest(kw=kw), self.assertRaises(CompileError) as cm:
                compile_spec(self.seeded(**kw))
            self.assertIn(needle, str(cm.exception))
        with self.assertRaises(CompileError) as cm:
            compile_spec(wl([task('t')], buffers=[{'name': 'w', 'dtype': 'i32', 'init': {'kind': 'seeded', 'seed': 1, 'scale': 1}}]))
        self.assertIn('dtype f32', str(cm.exception))


class FileInitTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='kvm-file-'))
        self.addCleanup(shutil.rmtree, self.dir)
        (self.dir / 'data').mkdir()
        (self.dir / 'data' / 'w.f32').write_bytes(struct.pack('<4f', 1, 2, 3, 4.5))
        (self.dir / 'data' / 'ids.i32').write_bytes(struct.pack('<2i', 5, -6))

    def spec(self, path='data/w.f32', rows=2, cols=2, dtype='f32'):
        return wl([task('t')], buffers=[{'name': 'w', 'rows': rows, 'cols': cols, 'dtype': dtype, 'init': {'kind': 'file', 'path': path}}])

    def test_compile_validates_and_materializes(self):
        c = compile_spec(self.spec(), data_dir=self.dir)
        self.assertEqual(c.metadata['buffers'][0]['init'], {'kind': 'file', 'path': 'data/w.f32'})
        self.assertEqual(c.buffers[0].materialize(self.dir), [1, 2, 3, 4.5])
        i = compile_spec(self.spec('data/ids.i32', 1, 2, 'i32'), data_dir=self.dir)
        self.assertEqual(i.buffers[0].materialize(self.dir), [5, -6])

    def test_path_and_size_errors(self):
        for path, needle in [('/etc/passwd', 'relative'), ('../x', 'relative'), ('data/../../x', 'relative'), ('data/missing', 'not found')]:
            with self.subTest(path=path), self.assertRaises(CompileError) as cm:
                compile_spec(self.spec(path), data_dir=self.dir)
            self.assertIn(needle, str(cm.exception))
        with self.assertRaises(CompileError) as cm:
            compile_spec(self.spec(rows=3), data_dir=self.dir)
        self.assertIn('has 16 bytes, expected 24', str(cm.exception))

    def test_without_data_dir_compiles_but_cannot_materialize_or_write(self):
        c = compile_spec(self.spec())
        with self.assertRaises(CompileError):
            c.buffers[0].materialize()
        with self.assertRaises(CompileError):
            cw.write_bundle(c, self.dir / 'out')

    def test_bundle_is_self_contained_and_vm_reads_it(self):
        c = compile_spec(self.spec(), data_dir=self.dir)
        out = cw.write_bundle(c, self.dir / 'out')
        self.assertEqual((out / 'data' / 'w.f32').read_bytes(), (self.dir / 'data' / 'w.f32').read_bytes())
        runtime, _ = vm.run_spec(self.spec(), deterministic=True, data_dir=self.dir)
        self.assertEqual(list(runtime.buffers['w']), [1, 2, 3, 4.5])

    def test_cli_defaults_data_dir_to_workload_directory(self):
        import json
        (self.dir / 'wl.json').write_text(json.dumps(self.spec()))
        self.assertEqual(cw.main([str(self.dir / 'wl.json'), str(self.dir / 'bundle'), '--emit-golden', '--log-level', 'CRITICAL']), 0)
        golden = json.loads((self.dir / 'bundle' / 'golden.json').read_text())
        self.assertEqual(golden['buffers']['w'], [1, 2, 3, 4.5])


class OpaqueTests(unittest.TestCase):
    SPEC = wl([task('f', 'doom_present', {'fb': 'fb', 'row0': 0, 'rows': 10}), task('p', 'signal', {'flag': 'fb', 'var': 0, 'ver': 1}, after=['f'])],
              buffers=[{'name': 'fb', 'rows': 127, 'cols': 128}])

    def test_refused_without_the_flag_and_noop_with_it(self):
        ops = all_ops()
        with self.assertRaises(vm.GraphError) as cm:
            vm.run_spec(self.SPEC, deterministic=True, ops=ops)
        self.assertIn('doom_present', str(cm.exception))
        runtime, trace = vm.run_spec(self.SPEC, deterministic=True, allow_opaque=True, ops=ops)
        self.assertEqual(len(trace), 2)
        self.assertEqual(set(runtime.buffers['fb']), {0.0})

    def test_declared_extents_still_checked(self):
        spec = wl([task('f', 'doom_present', {'fb': 'fb', 'row0': 0, 'rows': 10})], buffers=[{'name': 'fb', 'rows': 4, 'cols': 4}])
        with self.assertRaises(CompileError) as cm:
            compile_spec(spec, ops=all_ops())
        self.assertIn('16256', str(cm.exception))

    def test_op_without_any_semantics_is_refused_by_the_vm(self):
        from unittest import mock
        c = compile_spec(wl([task('t')]))
        with mock.patch.dict(ref_ops.OPS):
            del ref_ops.OPS['noop']
            with self.assertRaises(vm.GraphError) as cm:
                vm.Runtime(workers=1).submit_compiled(c)
        self.assertIn('no CPU semantics', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
