import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from apply import integrate
from apply_queue import integrate_queue
from test_integration import UPSTREAM, KERNEL

HERE = Path(__file__).resolve().parents[1] / 'integration'
QUEUE = (HERE / 'ready_queue_kernel.inc').read_text()
LOWERING = (HERE / 'queue_lowering.inc').read_text()

# The original fixture omits parts of main; supply the actual anchors of the
# source sections the second patcher needs to modify.
PREFIX = '#include <vector>\n'

class QueueIntegrationTests(unittest.TestCase):
    def source(self):
        scan = integrate(UPSTREAM, KERNEL)
        # Original fixture lacks the extra host lines, append them.
        return PREFIX + scan.replace("  Instr* dtapeP;", "// ================= host =================\n  Instr* dtapeP;", 1)

    def test_integration(self):
        out = integrate_queue(self.source(), QUEUE, LOWERING)
        self.assertIn('ready_queue_kernel<<<queue_grid', out)
        self.assertIn('best(run_queue, 5);', out)
        self.assertIn('KVM queue vs claim', out)
        self.assertIn('for (int mode : {3, 4, 5, 6})', out)
        self.assertIn('qgraph = kvm_compile_ready_graph(tapeP, steps, SC)', out)
        self.assertIn('for (int i = 0; i < 7; i++)', out)
        self.assertIn('best(run_vm, 3)', out)

    def test_double_patch_rejected(self):
        out = integrate_queue(self.source(), QUEUE, LOWERING)
        with self.assertRaises(ValueError): integrate_queue(out, QUEUE, LOWERING)

    def test_missing_anchor_rejected(self):
        with self.assertRaises(ValueError): integrate_queue(self.source().replace('  double t[5]; Out o[5];',''), QUEUE, LOWERING)

    def test_queue_publication_order(self):
        self.assertLess(QUEUE.index('queue_task[slot] = succ;'), QUEUE.index('kvm_q_publish(queue_ready + slot, 1);'))
        self.assertLess(QUEUE.index('exec_instr(in, sm, loss);'), QUEUE.index('kvm_q_dec(remaining + succ)'))

    def test_queue_capacity_invariant(self):
        self.assertIn('capacity >= n', QUEUE)
        self.assertIn('threshold != (int)p.size()', LOWERING)
        self.assertIn('dependency cycle', LOWERING)

if __name__ == '__main__': unittest.main()
