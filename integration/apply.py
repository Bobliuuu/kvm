#!/usr/bin/env python3
"""Apply an opt-in ready-only scheduling A/B experiment to upstream kvm.cu.

Usage: python3 integration/apply.py /path/to/Bobliuuu/kvm/kvm.cu
Writes kvm_ready.cu beside original; never overwrites original.
"""
import pathlib, sys

def integrate(source: str, kernel: str) -> str:
    if 'ready_scan_kernel' in source:
        raise ValueError('input already appears patched')
    anchors = ['// ---- baseline: one launch per phase, block i runs tape[off+i], no waits',
               '  Instr* dtapeP; int *dhead, *dctr; float* dloss;',
               '  CK(cudaFuncSetAttribute(vm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));',
               '  auto reset = [&]() {',
               '  auto run_vm = [&]() { vm_kernel<<<grid, NT, DSMEM>>>(dtapeP, ntape, dhead, dctr, dloss, nullptr); };',
               '  double t[4]; Out o[4];',
               '  best(run_seq, 0); best(run_seqgraph, 1); best(run_batched, 2); best(run_vm, 3);',
               '  const char* nm[4] =',
               '  for (int i = 0; i < 4; i++) printf("%-40s %10.2f %14.2f %11.0f %8.2fx\\n",',
               '  for (int i = 0; i < 4; i++) { e_ref =',
               '  for (int i = 0; i < 3; i++) for (size_t j = 0; j < o[3].w.size(); j++)']
    for anchor in anchors:
        if source.count(anchor) != 1:
            raise ValueError(f'expected unique upstream anchor, found {source.count(anchor)}: {anchor}')
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1: raise ValueError(f'expected one occurrence: {old}')
        source = source.replace(old, new, 1)
    replace(anchors[0], kernel + '\n' + anchors[0])
    replace(anchors[1], anchors[1] + '\n  int* dready_states; unsigned int *dready_cursor, *dready_done;\n  CK(cudaMalloc(&dready_states, (size_t)ntape * sizeof(int)));\n  CK(cudaMalloc(&dready_cursor, sizeof(unsigned int)));\n  CK(cudaMalloc(&dready_done, sizeof(unsigned int)));')
    replace(anchors[2], anchors[2] + '\n  CK(cudaFuncSetAttribute(ready_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));')
    replace('  int grid = prop.multiProcessorCount * occ;', '  int grid = prop.multiProcessorCount * occ;\n  int ready_occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ready_occ, ready_scan_kernel, NT, DSMEM));\n  int ready_grid = prop.multiProcessorCount * ready_occ;')
    replace(anchors[3], anchors[3] + '\n    CK(cudaMemset(dready_states, 0, (size_t)ntape * sizeof(int)));\n    CK(cudaMemset(dready_cursor, 0, sizeof(unsigned int)));\n    CK(cudaMemset(dready_done, 0, sizeof(unsigned int)));')
    replace(anchors[4], anchors[4] + '\n  auto run_ready = [&]() { ready_scan_kernel<<<ready_grid, NT, DSMEM>>>(dtapeP, ntape, dready_states, dready_cursor, dready_done, dctr, dloss); };')
    replace(anchors[5], '  double t[5]; Out o[5];')
    replace(anchors[6], anchors[6] + '\n  best(run_ready, 4);')
    replace('  const char* nm[4] = {"sequential, launch-per-op", "sequential, CUDA graph/run", "batched across runs (vmap-like graph)", "KVM: all runs, ONE kernel"};', '  const char* nm[5] = {"sequential, launch-per-op", "sequential, CUDA graph/run", "batched across runs (vmap-like graph)", "KVM: claim-and-wait", "KVM: ready-only scan"};')
    replace('  for (int i = 0; i < 4; i++) printf("%-40s %10.2f %14.2f %11.0f %8.2fx\\n",', '  for (int i = 0; i < 5; i++) printf("%-40s %10.2f %14.2f %11.0f %8.2fx\\n",')
    replace('  printf("KVM vs best baseline: %.2fx\\n", std::min(std::min(t[0], t[1]), t[2]) / t[3]);', '  printf("KVM claim vs best baseline: %.2fx\\n", std::min(std::min(t[0], t[1]), t[2]) / t[3]);\n  printf("KVM ready vs claim: %.2fx\\n", t[3] / t[4]);')
    replace('  for (int i = 0; i < 4; i++) { e_ref =', '  for (int i = 0; i < 5; i++) { e_ref =')
    replace('  for (int i = 0; i < 3; i++) for (size_t j = 0; j < o[3].w.size(); j++)', '  for (int i = 0; i < 5; i++) for (size_t j = 0; j < o[3].w.size(); j++)')
    replace('  bool dec = true; for (int r = 0; r < R; r++) dec = dec && o[3].L[(size_t)r * steps + steps - 1] < o[3].L[(size_t)r * steps];', '  bool dec = true; for (int mode : {3, 4}) for (int r = 0; r < R; r++) dec = dec && o[mode].L[(size_t)r * steps + steps - 1] < o[mode].L[(size_t)r * steps];')
    replace('  bool ok = e_ref < 2e-2 && e_mode < 1e-3 && dec;', '  bool ok = e_ref < 2e-2 && e_mode < 1e-3 && dec;\n  for (size_t j = 0; j < o[4].w.size(); j++) if (fabs(o[4].w[j] - o[3].w[j]) > 1e-3f) ok = false;')
    return source

def main():
    path = pathlib.Path(sys.argv[1])
    kernel = (pathlib.Path(__file__).parent/'ready_scan_kernel.inc').read_text()
    result = integrate(path.read_text(), kernel)
    out = path.with_name('kvm_ready.cu')
    out.write_text(result)
    print(f'Wrote {out} ({len(result)} bytes)')

if __name__ == '__main__': main()
