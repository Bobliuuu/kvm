import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'integration'))
from apply import integrate

KERNEL = (Path(__file__).resolve().parents[1]/'integration'/'ready_scan_kernel.inc').read_text()
# Exact upstream source fragments, enough to check the patch's assumptions.
UPSTREAM = '''// ---- baseline: one launch per phase, block i runs tape[off+i], no waits
  Instr* dtapeP; int *dhead, *dctr; float* dloss;
  CK(cudaFuncSetAttribute(vm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  int grid = prop.multiProcessorCount * occ;
  auto reset = [&]() {
  auto run_vm = [&]() { vm_kernel<<<grid, NT, DSMEM>>>(dtapeP, ntape, dhead, dctr, dloss, nullptr); };
  double t[4]; Out o[4];
  best(run_seq, 0); best(run_seqgraph, 1); best(run_batched, 2); best(run_vm, 3);
  const char* nm[4] = {"sequential, launch-per-op", "sequential, CUDA graph/run", "batched across runs (vmap-like graph)", "KVM: all runs, ONE kernel"};
  for (int i = 0; i < 4; i++) printf("%-40s %10.2f %14.2f %11.0f %8.2fx\\n",
  printf("KVM vs best baseline: %.2fx\\n", std::min(std::min(t[0], t[1]), t[2]) / t[3]);
  for (int i = 0; i < 4; i++) { e_ref =
  for (int i = 0; i < 3; i++) for (size_t j = 0; j < o[3].w.size(); j++)
  bool dec = true; for (int r = 0; r < R; r++) dec = dec && o[3].L[(size_t)r * steps + steps - 1] < o[3].L[(size_t)r * steps];
  bool ok = e_ref < 2e-2 && e_mode < 1e-3 && dec;
'''

class IntegrationTests(unittest.TestCase):
    def test_patch_injects_kernel(self):
        output = integrate(UPSTREAM, KERNEL)
        self.assertIn('ready_scan_kernel<<<ready_grid',output)
        self.assertIn('best(run_ready, 4)',output)
        self.assertIn('for (int i = 0; i < 5; i++)',output)
        self.assertIn('cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ready_occ',output)
    def test_preserves_original_baseline(self):
        output = integrate(UPSTREAM,KERNEL)
        self.assertIn('best(run_vm, 3)',output)
        self.assertIn('vm_kernel<<<grid',output)
    def test_idempotence_rejected(self):
        with self.assertRaises(ValueError): integrate(integrate(UPSTREAM,KERNEL),KERNEL)
    def test_upstream_drift_rejected(self):
        with self.assertRaises(ValueError): integrate(UPSTREAM.replace('  double t[4]; Out o[4];',''),KERNEL)
    def test_uniform_exit(self):
        self.assertIn('__syncthreads();\n    if (finished) return;',KERNEL)
    def test_claim_only_after_ready(self):
        self.assertLess(KERNEL.index('if (!kvm_ready(in, ctr)) continue;'),KERNEL.index('atomicCAS(states + idx, 0, 1)'))
    def test_signals_after_exec(self):
        self.assertLess(KERNEL.index('exec_instr(in, sm, loss);'),KERNEL.index('atomicAdd(ctr + in.s0, 1);'))

if __name__ == '__main__': unittest.main()
