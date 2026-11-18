// Operator-pack smoke test: build with a pack directory on the include path to prove sparse ids and the buf?/i32 plumbing
// against the real packs, e.g.  nvcc ... -I<dir with mvm_ops_gpt.def/.cuh> -DMVM_EXEC_NO_MAIN tests/gpu/pack_smoke.cu
// Runs ops of the GPT pack (ids 59/60) next to core ops; skipped (prints SKIP) when the pack is absent.
#include "../../mvm_exec.cu"
int main() {
  if (!mvm_op(59) || !mvm_op(60)) { printf("SKIP: no GPT pack on the include path\n"); return 0; }
  MvmConfig c; c.blocks = 3; c.max_tasks = 1024; c.live_cap = 256; c.ring_words = 4096; c.timeout_us = 10ull * 1000000;
  MvmExec ex(c);
  int f = ex.add_buffer(1, 8, std::vector<float>(8, 3.f));
  MvmInit one; one.kind = MvmInit::FILL; one.value = 5; int ib = ex.add_buffer(1, 8, one, 1);
  ex.start();
  MvmBundleBuilder bb;
  int z = bb.add("zero_buf", {MvmArg::buf(f), MvmArg::i32(2), MvmArg::i32(5)});
  int a = bb.add("counter_inc", {MvmArg::buf(ib), MvmArg::i32(1), MvmArg::i32(4), MvmArg::i32(10)});
  int s = bb.add("scale", {MvmArg::buf(f), MvmArg::buf(f), MvmArg::f32(2.f), MvmArg::i32(8)});
  bb.edge(z, s); (void)a;
  if (ex.submit(bb.build()) != MvmSubmit::OK) return 1;
  ex.close(); const MvmResult& r = ex.finish();
  auto fv = ex.read_buffer(f), iv = ex.read_buffer(ib); bool ok = r.ok() && r.done == 3;
  for (int i = 0; i < 8; ++i) { ok = ok && fv[i] == ((i >= 2 && i < 5) ? 0.f : 6.f); int v; memcpy(&v, &iv[i], 4); ok = ok && v == ((i >= 1 && i < 4) ? 15 : 5); }
  printf("%s: GPT-pack ops zero_buf(59) and counter_inc(60) ran beside core ops through the sparse-id dispatcher\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
