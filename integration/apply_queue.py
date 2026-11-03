#!/usr/bin/env python3
"""Add the dependency-driven queue to the ready-scan patched KVM source.

Usage: python3 integration/apply.py upstream/kvm.cu
       python3 integration/apply_queue.py upstream/kvm_ready.cu
Writes upstream/kvm_queued.cu without modifying inputs.
"""
from pathlib import Path
import sys

HERE = Path(__file__).parent

def integrate_queue(source: str, kernel: str, lowering: str, windowing: str = "", runner: str = "") -> str:
    if 'ready_queue_kernel' in source:
        raise ValueError('queue already integrated')
    def replace(old: str, new: str):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'expected one anchor, found {source.count(old)}: {old[:100]}')
        source = source.replace(old, new, 1)
    replace('#include <vector>', '#include <vector>\n#include <algorithm>\n#include <stdexcept>')
    replace('// ---- baseline: one launch per phase, block i runs tape[off+i], no waits',
            kernel + '\n// ---- baseline: one launch per phase, block i runs tape[off+i], no waits')
    replace('// ================= host =================',
            lowering + '\n' + windowing + '\n// ================= host =================')
    replace('  Instr* dtapeP; int *dhead, *dctr; float* dloss;', '''  KvmReadyGraph qgraph = kvm_compile_ready_graph(tapeP, steps, SC);
  printf("queue DAG: %d tasks, %zu edges, %zu initially ready\\n", ntape,
         qgraph.successors.size(), qgraph.initial.size());
  int *q_offsets, *q_successors, *q_remaining, *q_tasks, *q_ready;
  int *q_head, *q_tail, *q_done;
  auto qalloc = [&](int** dst, size_t count) {
    CK(cudaMalloc(dst, std::max((size_t)1, count) * sizeof(int)));
  };
  qalloc(&q_offsets, qgraph.offsets.size());
  qalloc(&q_successors, qgraph.successors.size());
  qalloc(&q_remaining, ntape); qalloc(&q_tasks, ntape);
  qalloc(&q_ready, ntape);
  qalloc(&q_head, 1); qalloc(&q_tail, 1); qalloc(&q_done, 1);
  CK(cudaMemcpy(q_offsets, qgraph.offsets.data(),
                qgraph.offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  if (!qgraph.successors.empty())
    CK(cudaMemcpy(q_successors, qgraph.successors.data(),
                  qgraph.successors.size() * sizeof(int), cudaMemcpyHostToDevice));
  Instr* dtapeP; int *dhead, *dctr; float* dloss;''')
    replace('  CK(cudaFuncSetAttribute(ready_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));',
            '  CK(cudaFuncSetAttribute(ready_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));\n  CK(cudaFuncSetAttribute(ready_queue_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));')
    replace('  int ready_grid = prop.multiProcessorCount * ready_occ;', '''  int ready_grid = prop.multiProcessorCount * ready_occ;
  int queue_occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &queue_occ, ready_queue_kernel, NT, DSMEM));
  int queue_grid = prop.multiProcessorCount * queue_occ;''')
    replace('    CK(cudaMemset(dready_done, 0, sizeof(unsigned int)));', '''    CK(cudaMemset(dready_done, 0, sizeof(unsigned int)));
    CK(cudaMemcpy(q_remaining, qgraph.indegree.data(),
                  (size_t)ntape * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(q_ready, 0, (size_t)ntape * sizeof(int)));
    // Initial queue slots are fully populated before kernel launch.
    if (!qgraph.initial.empty()) {
      CK(cudaMemcpy(q_tasks, qgraph.initial.data(),
                    qgraph.initial.size() * sizeof(int), cudaMemcpyHostToDevice));
      std::vector<int> ones(qgraph.initial.size(), 1);
      CK(cudaMemcpy(q_ready, ones.data(), ones.size() * sizeof(int),
                    cudaMemcpyHostToDevice));
    }
    int zero = 0, initial_count = (int)qgraph.initial.size();
    CK(cudaMemcpy(q_head, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(q_done, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(q_tail, &initial_count, sizeof(int), cudaMemcpyHostToDevice));''')
    replace('  auto run_ready = [&]() { ready_scan_kernel<<<ready_grid, NT, DSMEM>>>(dtapeP, ntape, dready_states, dready_cursor, dready_done, dctr, dloss); };', '''  auto run_ready = [&]() { ready_scan_kernel<<<ready_grid, NT, DSMEM>>>(dtapeP, ntape, dready_states, dready_cursor, dready_done, dctr, dloss); };
  auto run_queue = [&]() { ready_queue_kernel<<<queue_grid, NT, DSMEM>>>(
      dtapeP, ntape, q_offsets, q_successors, q_remaining, q_tasks,
      q_ready, q_head, q_tail, q_done, dloss, 0); };''')
    replace('  double t[5]; Out o[5];', (runner or '// window runner omitted by fixture') + '\n  double t[7]; Out o[7];')

    replace('  best(run_ready, 4);', '  best(run_ready, 4);\n  best(run_queue, 5);\n  best(run_queue_window, 6);')
    replace('  const char* nm[5] = {"sequential, launch-per-op", "sequential, CUDA graph/run", "batched across runs (vmap-like graph)", "KVM: claim-and-wait", "KVM: ready-only scan"};',
            '  const char* nm[7] = {"sequential, launch-per-op", "sequential, CUDA graph/run", "batched across runs (vmap-like graph)", "KVM: claim-and-wait", "KVM: ready-only scan", "KVM: ready queue", "KVM: windowed queue"};')
    replace('  for (int i = 0; i < 5; i++) printf("%-40s %10.2f %14.2f %11.0f %8.2fx\\n",',
            '  for (int i = 0; i < 7; i++) printf("%-40s %10.2f %14.2f %11.0f %8.2fx\\n",')
    replace('  printf("KVM ready vs claim: %.2fx\\n", t[3] / t[4]);',
            '  printf("KVM ready vs claim: %.2fx\\n", t[3] / t[4]);\n  printf("KVM queue vs claim: %.2fx\\n", t[3] / t[5]);')
    replace('  for (int i = 0; i < 5; i++) { e_ref =', '  for (int i = 0; i < 7; i++) { e_ref =')
    replace('  for (int i = 0; i < 5; i++) for (size_t j = 0; j < o[3].w.size(); j++)',
            '  for (int i = 0; i < 7; i++) for (size_t j = 0; j < o[3].w.size(); j++)')
    replace('for (int mode : {3, 4})', 'for (int mode : {3, 4, 5, 6})')
    replace('  for (size_t j = 0; j < o[4].w.size(); j++) if (fabs(o[4].w[j] - o[3].w[j]) > 1e-3f) ok = false;',
            '  for (int mode : {4, 5, 6}) for (size_t j = 0; j < o[mode].w.size(); j++) if (fabs(o[mode].w[j] - o[3].w[j]) > 1e-3f) ok = false;')
    return source

def main():
    src = Path(sys.argv[1])
    output = src.with_name('kvm_queued.cu')
    output.write_text(integrate_queue(src.read_text(),
                                     (HERE / 'ready_queue_kernel.inc').read_text(),
                                     (HERE / 'queue_lowering.inc').read_text(),
                                     (HERE / 'window_graph.inc').read_text(),
                                     (HERE / 'window_runner.inc').read_text()))
    print('Wrote', output)

if __name__ == '__main__':
    main()
