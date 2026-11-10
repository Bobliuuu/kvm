"""OP_WRITES (runtime/partition.py) must equal what the ops actually store to: zoo workloads run in the CPU VM with buffers that
record every store, and the recorded written-argument sets are compared with the table. The place and dp planners rely on it for
coherence (a write invalidates every other copy), so a missing entry would silently read stale data on another device."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'dist'), str(ROOT / 'frontend'), str(ROOT / 'examples/models')]
import commops            # noqa: E402
import compile_workload as cw  # noqa: E402
import partition as pt    # noqa: E402
import ref_ops            # noqa: E402
import workload_vm        # noqa: E402
from ref_types import Buf  # noqa: E402

MODELS = ['gpt_tiny', 'llama_tiny', 'moe_tiny', 'vit_tiny', 'mlp_tiny',
          '{"family": "mlp", "name": "mlp_bf16s", "precision": "bf16s"}', '{"family": "llama", "name": "llama_bf16", "precision": "bf16"}']


class _RBuf(Buf):
    __slots__ = ('hit',)

    def __setitem__(self, k, v):
        self.hit = True
        return super().__setitem__(k, v)


def recorded_writes(models):
    import gen_bundle as zoo
    seen = {}

    def execute(self, task):
        bufs = {}
        for a in task.op.args:
            if a.type != 'buf':
                continue
            n = task.args[a.name]
            if n is None:
                bufs[a.name] = None
                continue
            b = self.buffers[n]
            if not isinstance(b, _RBuf):
                b = self.buffers[n] = _RBuf(b, b.rows, b.cols, b.dtype)
            b.hit = False
            bufs[a.name] = b
        ref_ops.OPS[task.op.name].run(bufs, {a.name: task.args[a.name] for a in task.op.args if a.type != 'buf'})
        seen.setdefault(task.op.name, set()).update(k for k, b in bufs.items() if b is not None and b.hit)

    orig = workload_vm.Runtime._execute
    workload_vm.Runtime._execute = execute
    try:
        for m in models:
            c, _ = zoo.build(m, 2, accum=2 if m == 'llama_tiny' else 1)
            workload_vm.run_golden(cw.compile_spec(c.spec), timeout=900)
    finally:
        workload_vm.Runtime._execute = orig
    return seen


class OpWrites(unittest.TestCase):
    def test_table_matches_recorded_stores(self):
        seen = recorded_writes(MODELS)
        self.assertGreater(len(seen), 40)
        for op, args in sorted(seen.items()):
            self.assertIn(op, pt.OP_WRITES, f'{op} missing from OP_WRITES')
            # an argument may write only under some flags (matmul 'pre', norm 'dbeta'): recorded must be a subset, and every
            # table entry must have been observed in at least one of the models
            self.assertTrue(args <= set(pt.OP_WRITES[op]), f'{op}: stores to {sorted(args - set(pt.OP_WRITES[op]))} not in the table')
            self.assertEqual(set(pt.OP_WRITES[op]) - args - {'pre', 'dbeta', 'x'}, set(), f'{op}: table lists writes never observed')

    def test_unknown_op_is_refused(self):
        planner = pt.PlacePlanner.__new__(pt.PlacePlanner)
        with self.assertRaises(pt.PartitionError):
            planner.writes({'op': 'no_such_op', 'args': {}})


if __name__ == '__main__':
    unittest.main()
