// Distributed-execution tests on real GPUs. Build (per-op IEEE arithmetic so results compare bitwise with the CPU reference):
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -DKVM_EXEC_NO_MAIN tests/dist/test_dist.cu -o test_dist -lpthread
//   ./test_dist [suite...]            suites: ops pingpong errors backpressure bw plan   (default: all but plan)
//   ./test_dist plan DIR [--net loopback|tcp] [--fault drop=0.1,dup=0.1,delay=0.2] [--blocks N]
//        runs every rank of a partitioner plan (runtime/partition.py --plan) as threads of this process
//   ./test_dist plan DIR --rank R --peers host:port,host:port,...      one rank of a MULTI-PROCESS / multi-node run
//   ./test_dist --expect-fail=stale pingpong     negative test (-DKVM_DIST_BREAK_ORDER): exits 0 only if the suite DETECTS stale data;
//   ./test_dist --expect-fail=timeout backpressure   (-DKVM_DIST_NO_BP_POLL) exits 0 only if the receiver hits the device stall timeout (TIMEOUT / PARK_STALL)
// Everything that needs more than one physical GPU prints "SKIP: needs 2 GPUs" when fewer are present; with >= 2 GPUs the same
// suites place rank r on device r % ngpus and use real peer-to-peer.
#include "../../kvm_exec.cu"
#include "../../dist/cuda_dist.hpp"

#include <functional>
#include <random>
#include <sstream>

using namespace kvmd;

static int g_fail = 0, g_checks = 0, g_skip = 0, g_stale = 0, g_timeouts = 0;   // g_stale / g_timeouts: WHICH way a negative test failed
static bool g_mute = false;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; if (!g_mute) { fprintf(stderr, "  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #c); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)
static int g_ngpu = 1;
static KvmArg Bf(int i) { return KvmArg::buf(i); }
static KvmArg I(int v) { return KvmArg::i32(v); }
static KvmArg F(float v) { return KvmArg::f32(v); }
static bool same_bits(float a, float b) { return memcmp(&a, &b, 4) == 0; }

static KvmConfig small_cfg(int blocks = 4, int num_vars = 64) {
  KvmConfig c; c.blocks = blocks; c.max_tasks = 1 << 14; c.live_cap = 1 << 11; c.ring_words = 1 << 14; c.timeout_us = 30ull * 1000000; c.num_vars = num_vars; return c;
}
static std::vector<float> download(const float* d, size_t n) { std::vector<float> v(n); CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v; }

// ==================================================================================================== ops (one executor)
// Local buffers only: the comm ops are ordinary ops, so a "peer" buffer on the same device is just another allocation.
struct OpsEnv {
  KvmExec ex; DistDevice dd; float *a, *b, *c; int ia, ib, ic, iin, iq;
  static constexpr int N = 4096;
  OpsEnv(int sq_slots = 4, int sq_bytes = 1024) : ex(small_cfg()), dd(0, 1, 0, 64, sq_slots, sq_bytes, 300ull * 1000000) {
    std::mt19937 g(5);
    std::vector<float> va(N), vb(N);
    for (auto& x : va) x = (float)((int)(g() % 17) - 8) * 0.37f;
    for (auto& x : vb) x = (float)((int)(g() % 255) - 100) * 1.3f;
    ia = ex.add_buffer(64, 64, va); ib = ex.add_buffer(64, 64, vb); ic = ex.add_buffer(64, 64);
    iin = ex.add_view(dd.inbox_buf(), 1, 64);
    iq = ex.add_view(dd.sendq_buf(), 1, (int)dd.sendq_floats());
  }
};

static uint16_t bf16_bits(float x) {
  uint32_t u; memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_to_f(uint32_t b16) { uint32_t u = b16 << 16; float f; memcpy(&f, &u, 4); return f; }

static int run_one(const std::function<void(OpsEnv&, KvmBundleBuilder&)>& build, OpsEnv& env, std::vector<float>* ra = nullptr, std::vector<float>* rb = nullptr, std::vector<float>* rc = nullptr) {
  KvmBundleBuilder bb; build(env, bb);
  env.ex.start();
  KvmSubmit s = env.ex.submit(bb.build());
  CHECK(s == KvmSubmit::OK, "submit %s", kvm_submit_name(s));
  const KvmResult& r = env.ex.finish();
  if (r.ok()) { if (ra) *ra = env.ex.read_buffer(env.ia); if (rb) *rb = env.ex.read_buffer(env.ib); if (rc) *rc = env.ex.read_buffer(env.ic); }
  return r.err_code;
}

static void suite_ops() {
  fprintf(stderr, "== ops: comm operators vs CPU reference\n");
  std::mt19937 g(11);
  // --- put / put_signal: random tiles, strided, unaligned (exercise the float4 and scalar paths)
  int mismatch = 0, tiles = 0;
  const int trials = getenv("KVM_DIST_QUICK") ? 6 : 40;   // sanitizer runs: every launch is slow under instrumentation
  for (int trial = 0; trial < trials; ++trial) {
    OpsEnv env;
    std::vector<float> a0 = env.ex.read_buffer(env.ia), c0 = env.ex.read_buffer(env.ic), want = c0;
    struct T { int so, dof, rows, cols, sld, dld, var, ver; bool sig; };
    std::vector<T> ts;
    for (int k = 0; k < 6; ++k) {
      T t; t.cols = 1 + g() % 40; t.rows = 1 + g() % 20; t.sld = t.cols + g() % 24; t.dld = t.cols + g() % 24; t.sig = g() % 2;
      if (trial % 3 == 0) { t.cols = 4 * (1 + g() % 10); t.sld = t.cols + 4 * (g() % 6); t.dld = t.cols + 4 * (g() % 6); }   // aligned -> vector path
      int sext = (t.rows - 1) * t.sld + t.cols, dext = (t.rows - 1) * t.dld + t.cols;
      if (sext > 4096 || dext > 4096) { --k; continue; }
      t.so = (trial % 3 == 0) ? 4 * (g() % ((4096 - sext) / 4 + 1)) : g() % (4096 - sext + 1);
      t.dof = (trial % 3 == 0) ? 4 * (g() % ((4096 - dext) / 4 + 1)) : g() % (4096 - dext + 1);
      t.var = 1 + g() % 60; t.ver = 1 + g() % 1000;
      // chain the tasks, so overlapping destinations have a defined order; the CPU model applies them in the same order
      ts.push_back(t);
      for (int r = 0; r < t.rows; ++r) for (int c = 0; c < t.cols; ++c) want[t.dof + r * t.dld + c] = a0[t.so + r * t.sld + c];
      ++tiles;
    }
    std::vector<float> got;
    int err = run_one([&](OpsEnv& e, KvmBundleBuilder& bb) {
      int prev = -1;
      for (auto& t : ts) {
        int id = t.sig ? bb.add("put_signal", {Bf(e.ia), Bf(e.ic), I(t.so), I(t.dof), I(t.rows), I(t.cols), I(t.sld), I(t.dld), Bf(e.iin), I(t.var), I(t.ver)})
                       : bb.add("put", {Bf(e.ia), Bf(e.ic), I(t.so), I(t.dof), I(t.rows), I(t.cols), I(t.sld), I(t.dld)});
        if (prev >= 0) bb.edge(prev, id);
        prev = id;
      }
    }, env, nullptr, nullptr, &got);
    CHECK(err == 0, "device error %s", kvm_err_name(err));
    if (!err) for (size_t i = 0; i < got.size(); ++i) mismatch += !same_bits(got[i], want[i]);
    std::vector<unsigned> in = env.dd.read_inbox();
    std::vector<unsigned> win(64, 0);
    for (auto& t : ts) if (t.sig) win[t.var] = t.ver;                // later signals to the same var overwrite: the chain order is the task order
    CHECK(in == win, "inbox mismatch in trial %d", trial);
  }
  CHECK(mismatch == 0, "%d wrong elements over %d tiles", mismatch, tiles);
  fprintf(stderr, "  put/put_signal: %d random tiles bitwise equal (scalar and 16-byte paths, strided)\n", tiles);

  // --- reduce2 / reduce_put_signal
  {
    OpsEnv env; std::vector<float> a0 = env.ex.read_buffer(env.ia), b0 = env.ex.read_buffer(env.ib), want = env.ex.read_buffer(env.ic), got;
    int n1 = 1000, n2 = 777;
    for (int i = 0; i < n1; ++i) want[10 + i] = a0[5 + i] + b0[20 + i];
    for (int i = 0; i < n2; ++i) want[2000 + i] = a0[100 + i] + b0[3 + i];
    int err = run_one([&](OpsEnv& e, KvmBundleBuilder& bb) {
      int t1 = bb.add("reduce2", {Bf(e.ia), Bf(e.ib), Bf(e.ic), I(5), I(20), I(10), I(n1)});
      int t2 = bb.add("reduce_put_signal", {Bf(e.ia), Bf(e.ib), Bf(e.ic), I(100), I(3), I(2000), I(n2), Bf(e.iin), I(7), I(3)});
      bb.edge(t1, t2);
    }, env, nullptr, nullptr, &got);
    CHECK(err == 0, "device error %s", kvm_err_name(err));
    int bad = 0; if (!err) for (size_t i = 0; i < got.size(); ++i) bad += !same_bits(got[i], want[i]);
    CHECK(bad == 0, "%d wrong elements", bad);
    CHECK(env.dd.read_inbox()[7] == 3u, "reduce_put_signal did not signal");
    fprintf(stderr, "  reduce2 / reduce_put_signal: bitwise equal\n");
  }
  // --- bf16 pack / unpack (+accumulate), odd n, subnormals, huge, negative zero
  {
    OpsEnv env; std::vector<float> a0 = env.ex.read_buffer(env.ia);
    a0[0] = -0.f; a0[1] = 1e-40f; a0[2] = 3.4e38f; a0[3] = 1.00390625f; a0[4] = 1.00390625f + 1.1920929e-7f;   // tie / just above tie
    env.ex.set_version(0, 0);
    int n = 1001;
    std::vector<float> got_c; std::vector<float> want(4096, 0.f);
    // we upload the patched a0 through a fresh executor
    KvmExec ex(small_cfg()); DistDevice dd(0, 1, 0, 8);
    int ia = ex.add_buffer(64, 64, a0), ib = ex.add_buffer(64, 64), ic = ex.add_buffer(64, 64, std::vector<float>(4096, 2.f));
    KvmBundleBuilder bb;
    int p = bb.add("pack_bf16", {Bf(ia), Bf(ib), I(0), I(0), I(n)});
    int u = bb.add("unpack_bf16", {Bf(ib), Bf(ic), I(0), I(10), I(n), I(1)});   // acc: 2 + bf16(a)
    int u2 = bb.add("unpack_bf16", {Bf(ib), Bf(ic), I(0), I(2000), I(n), I(0)});
    bb.edge(p, u); bb.edge(u, u2);
    ex.start(); CHECK(ex.submit(bb.build()) == KvmSubmit::OK, "submit"); const KvmResult& r = ex.finish();
    CHECK(r.ok(), "device error %s", kvm_err_name(r.err_code));
    std::vector<float> got = ex.read_buffer(ic); int bad = 0;
    for (int i = 0; i < n; ++i) {
      float v = bf16_to_f(bf16_bits(a0[i]));
      bad += !same_bits(got[10 + i], 2.f + v) || !same_bits(got[2000 + i], v);
    }
    CHECK(bad == 0, "%d bf16 mismatches", bad);
    fprintf(stderr, "  pack_bf16/unpack_bf16: %d values bitwise equal (RNE, tie, subnormal, odd n)\n", n);
  }
  // --- validation: every misuse is a decoded device error, never UB
  struct Neg { const char* what; int want; std::function<void(OpsEnv&, KvmBundleBuilder&)> f; };
  std::vector<Neg> negs = {
    {"put source tile out of range", KVM_ERR_BOUNDS, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("put", {Bf(e.ia), Bf(e.ic), I(4090), I(0), I(2), I(10), I(10), I(10)}); }},
    {"put ld < cols", KVM_ERR_BOUNDS, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("put", {Bf(e.ia), Bf(e.ic), I(0), I(0), I(2), I(10), I(10), I(5)}); }},
    {"put negative offset", KVM_ERR_BOUNDS, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("put", {Bf(e.ia), Bf(e.ic), I(-1), I(0), I(1), I(10), I(10), I(10)}); }},
    {"signal var outside the mailbox", KVM_ERR_DIST_BAD_SIGNAL, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("signal", {Bf(e.iin), I(64), I(1)}); }},
    {"signal version 0", KVM_ERR_DIST_BAD_SIGNAL, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("signal", {Bf(e.iin), I(3), I(0)}); }},
    {"send_proxy to a buffer that is not a queue", KVM_ERR_DIST_BAD_QUEUE, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("send_proxy", {Bf(e.ia), Bf(e.ib), I(0), I(1), I(8), I(8), I(1), I(0), I(0), I(8), I(1), I(1)}); }},
    {"send_proxy payload larger than a slot", KVM_ERR_DIST_MSG_TOO_BIG, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("send_proxy", {Bf(e.ia), Bf(e.iq), I(0), I(1), I(512), I(512), I(1), I(0), I(0), I(512), I(1), I(1)}); }},
    {"send_proxy negative destination rank", KVM_ERR_DIST_BAD_MSG, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("send_proxy", {Bf(e.ia), Bf(e.iq), I(0), I(1), I(8), I(8), I(-1), I(0), I(0), I(8), I(1), I(1)}); }},
    {"reduce2 out of range", KVM_ERR_BOUNDS, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("reduce2", {Bf(e.ia), Bf(e.ib), Bf(e.ic), I(0), I(0), I(4000), I(200)}); }},
    {"unpack_bf16 source too short", KVM_ERR_BOUNDS, [](OpsEnv& e, KvmBundleBuilder& b) { b.add("unpack_bf16", {Bf(e.ia), Bf(e.ib), I(4090), I(0), I(100), I(0)}); }},
  };
  for (auto& n : negs) {
    OpsEnv env; g_mute = true; int err = run_one(n.f, env); g_mute = false;
    CHECK(err == n.want, "%s: got %s (%d), want %s", n.what, kvm_err_name(err), err, kvm_err_name(n.want));
  }
  fprintf(stderr, "  %zu misuse cases end the kernel with the right decoded device error\n", negs.size());
}

// ==================================================================================================== multi-rank scaffolding
// gen = a block of a counter-based seeded matrix (runtime/initgen.py: SplitMix64 at counter j, same values as the CPU reference):
// logical matrix GR x GC, element (r, c) = draw index tr ? c*GR + r : r*GC + c; this buffer holds rows r0.., cols c0.. of it.
struct PlanBuf { enum Kind { LOCAL, PEER, INBOX, SENDQ } kind = LOCAL; int rows = 1, cols = 1, rank = -1, idx = -1; bool fill = true; float fillv = 0; std::vector<float> vals;
  bool i32 = false, gen = false; unsigned long long seed = 0; int draws = 1, gr = 0, gc = 0, tr = 0, r0 = 0, c0 = 0; float bias = 0, coef = 1; };
struct DumpSpec { int idx; std::string name; int r0, c0, gr, gc; };
struct PlanRank {
  int rank = 0, world = 1, nvars = 1;
  std::vector<PlanBuf> bufs; std::vector<std::pair<int, int>> vars; std::vector<int> in_vars; std::map<int, std::vector<float>> expect; std::vector<DumpSpec> dumps;
  KvmBundle bundle;
  bool uses_sendq() const { for (auto& b : bufs) if (b.kind == PlanBuf::SENDQ) return true; return false; }
  bool uses_peer() const { for (auto& b : bufs) if (b.kind == PlanBuf::PEER || b.kind == PlanBuf::INBOX) return true; return false; }
};

template <class T> static std::vector<T> slurp(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb"); if (!f) throw std::runtime_error("cannot read " + path);
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<T> v((size_t)n / sizeof(T)); if (n && fread(v.data(), 1, (size_t)n, f) != (size_t)n) { fclose(f); throw std::runtime_error("short read " + path); }
  fclose(f); return v;
}
static PlanRank load_rank(const std::string& dir) {
  PlanRank pr;
  pr.bundle.tasks = slurp<KvmTaskDesc>(dir + "/tasks.bin");
  pr.bundle.succ = slurp<int32_t>(dir + "/successors.bin");
  pr.bundle.args = slurp<int32_t>(dir + "/args.bin");
  std::ifstream f(dir + "/dist.txt"); if (!f) throw std::runtime_error("cannot read " + dir + "/dist.txt");
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ls(line); std::string key; ls >> key;
    if (key == "rank") { std::string w; ls >> pr.rank >> w >> pr.world; }
    else if (key == "nvars") ls >> pr.nvars;
    else if (key == "buf") {
      int idx; std::string kind; ls >> idx >> kind; PlanBuf b;
      if (idx != (int)pr.bufs.size()) throw std::runtime_error("buffer lines must be dense and ascending");
      if (kind == "local") {
        std::string init; ls >> b.rows >> b.cols >> init;
        if (init == "fill") { b.fill = true; ls >> b.fillv; }
        else if (init == "gen") { b.fill = false; b.gen = true; ls >> b.seed >> b.draws >> b.bias >> b.coef >> b.gr >> b.gc >> b.tr >> b.r0 >> b.c0; }
        else { b.fill = false; b.vals.resize((size_t)b.rows * b.cols); std::string t; for (auto& x : b.vals) { ls >> t; x = strtof(t.c_str(), nullptr); } }
        { std::string t; while (ls >> t) if (t == "i32") b.i32 = true; }
      } else if (kind == "peer") { b.kind = PlanBuf::PEER; ls >> b.rank >> b.idx >> b.rows >> b.cols; }
      else if (kind == "inbox") { b.kind = PlanBuf::INBOX; ls >> b.rank >> b.cols; b.rows = 1; }
      else if (kind == "sendq") b.kind = PlanBuf::SENDQ;
      else throw std::runtime_error("unknown buffer kind " + kind);
      pr.bufs.push_back(std::move(b));
    } else if (key == "var") { int id, base; ls >> id >> base; pr.vars.push_back({id, base}); }
    else if (key == "inbox_var") { int v; ls >> v; pr.in_vars.push_back(v); }
    else if (key == "dump") { DumpSpec d; ls >> d.idx >> d.name >> d.r0 >> d.c0 >> d.gr >> d.gc; pr.dumps.push_back(d); }
    else if (key == "expect") { int idx, n; ls >> idx >> n; std::vector<float> v(n); std::string t; for (auto& x : v) { ls >> t; x = strtof(t.c_str(), nullptr); } pr.expect[idx] = std::move(v); }
  }
  pr.nvars = std::max(pr.nvars, 1);
  return pr;
}

static inline unsigned long long sm64(unsigned long long seed, unsigned long long j) {
  unsigned long long z = seed + (j + 1) * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31);
}
static inline float gen_u(unsigned long long seed, unsigned long long j) { return (float)(sm64(seed, j) >> 40) * (1.0f / 16777216.0f); }
// Fills h[0 .. rows*cols) with the block of the seeded matrix described by b (threads: the box is shared, keep it small).
static void gen_block(const PlanBuf& b, std::vector<float>& h) {
  size_t n = (size_t)b.rows * b.cols; h.resize(n);
  int T = (int)std::min<size_t>(8, std::max<size_t>(1, n >> 20)); std::vector<std::thread> th;
  for (int t = 0; t < T; ++t) th.emplace_back([&, t] {
    size_t lo = n * t / T, hi = n * (t + 1) / T;
    for (size_t i = lo; i < hi; ++i) {
      unsigned long long lr = i / b.cols, lc = i % b.cols, r = b.r0 + lr, c = b.c0 + lc;
      unsigned long long idx = b.tr ? c * (unsigned long long)b.gr + r : r * (unsigned long long)b.gc + c, j = idx * b.draws;
      float acc = b.draws == 1 ? gen_u(b.seed, j) : ((gen_u(b.seed, j) + gen_u(b.seed, j + 1)) + (gen_u(b.seed, j + 2) + gen_u(b.seed, j + 3)));
      h[i] = (acc - b.bias) * b.coef;
    }
  });
  for (auto& x : th) x.join();
}

struct Net {   // the inter-node fabric of an in-process run: one transport per rank
  std::unique_ptr<LoopbackHub> hub;
  std::vector<std::unique_ptr<Transport>> base, top;
  std::vector<std::unique_ptr<Reliable>> rel;
  FaultSpec fault; bool faulty = false;
  void build(int world, bool tcp, FaultSpec f, int self = -1, std::vector<std::string> hosts = {}, std::vector<int> ports = {}) {
    fault = f; faulty = f.drop > 0 || f.dup > 0 || f.delay > 0;
    base.resize(world); top.resize(world); rel.resize(world);
    if (self >= 0 && !hosts.empty()) {   // multi-process: only our own rank exists here
      TcpConfig c; c.rank = self; c.hosts = hosts; c.ports = ports; c.cluster_id = 0x4b564d;
      base[self] = std::make_unique<TcpTransport>(c);
    } else if (tcp) {
      auto ports2 = pick_free_ports(world); std::vector<std::thread> th;
      for (int r = 0; r < world; ++r) th.emplace_back([&, r] { TcpConfig c; c.rank = r; c.hosts.assign(world, "127.0.0.1"); c.ports = ports2; c.cluster_id = 1; base[r] = std::make_unique<TcpTransport>(c); });
      for (auto& t : th) t.join();
    } else {
      hub = std::make_unique<LoopbackHub>(world);
      for (int r = 0; r < world; ++r) base[r] = std::make_unique<LoopbackTransport>(*hub, r);
    }
    ReliableConfig rc; rc.rto_ms = 50; rc.rto_max_ms = 500; rc.max_retries = 20; rc.dead_ms = 20000;
    for (int r = 0; r < world; ++r) {
      if (!base[r]) continue;
      top[r] = faulty ? std::unique_ptr<Transport>(new FaultyTransport(*base[r], f)) : nullptr;
      rel[r] = std::make_unique<Reliable>(top[r] ? *top[r] : *base[r], rc);
    }
  }
};

struct RankOut { KvmResult res; bool ok = false; std::string msg; double ms = 0, start_ms = 0, submit_ms = 0, finish_ms = 0; };

// Runs a set of ranks (all of them in-process, or just `only`). `devs[r]` = CUDA device of rank r.
struct World {
  std::vector<PlanRank> pr;
  std::vector<int> dev;
  std::vector<std::unique_ptr<DistDevice>> dd;
  std::vector<std::vector<float*>> local;
  std::vector<std::unique_ptr<KvmExec>> ex;
  std::vector<std::unique_ptr<CudaSink>> sink;
  std::vector<std::unique_ptr<Proxy>> px;
  Net net;
  int only = -1;
  KvmConfig cfg = small_cfg(4);
  uint32_t sq_slots = 16, sq_slot_bytes = 256 * 1024;      // slot = the partitioner's max_msg_bytes
  std::vector<float*> allocs;
  std::map<int, size_t> local_bytes, local_count;

  ~World() { px.clear(); sink.clear(); ex.clear(); for (float* p : allocs) cudaFree(p); dd.clear(); }

  bool active(int r) const { return only < 0 || only == r; }
  void build(bool tcp, FaultSpec f, std::vector<std::string> hosts = {}, std::vector<int> ports = {}) {
    int W = (int)pr.size();
    local.assign(W, {});
    dd.resize(W); ex.resize(W); sink.resize(W); px.resize(W);
    std::set<int> devset;
    for (int r = 0; r < W; ++r) if (active(r)) devset.insert(dev[r]);
    bool peers = false; for (auto& p : pr) peers |= p.uses_peer();
    if (peers && devset.size() > 1) enable_peer_access(std::vector<int>(devset.begin(), devset.end()));
    for (int r = 0; r < W; ++r) {
      if (!active(r)) continue;
      PlanRank& p = pr[r];
      CK(cudaSetDevice(dev[r]));
      dd[r] = std::make_unique<DistDevice>(r, W, dev[r], p.nvars, sq_slots, sq_slot_bytes, 10ull * 1000000000ull);
      std::vector<int> iv(p.in_vars.begin(), p.in_vars.end()); dd[r]->set_in_vars(iv);
      local[r].assign(p.bufs.size(), nullptr);
      for (size_t i = 0; i < p.bufs.size(); ++i) {
        PlanBuf& b = p.bufs[i];
        if (b.kind != PlanBuf::LOCAL) continue;
        size_t n = (size_t)b.rows * b.cols; float* d; CK(cudaMalloc(&d, n * 4)); allocs.push_back(d); local_bytes[r] += n * 4; ++local_count[r];
        std::vector<float> h; if (b.gen) gen_block(b, h); else h = b.fill ? std::vector<float>(n, b.fillv) : b.vals;
        if (b.i32) for (auto& x : h) { int iv = (int)x; memcpy(&x, &iv, 4); }
        CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice)); local[r][i] = d;
      }
    }
    // second pass: peers' pointers are known now
    for (int r = 0; r < W; ++r) {
      if (!active(r)) continue;
      PlanRank& p = pr[r];
      CK(cudaSetDevice(dev[r]));
      KvmConfig c = cfg; c.num_vars = p.nvars;
      ex[r] = std::make_unique<KvmExec>(c);
      for (size_t i = 0; i < p.bufs.size(); ++i) {
        PlanBuf& b = p.bufs[i]; float* ptr = nullptr; int rows = b.rows, cols = b.cols;
        switch (b.kind) {
          case PlanBuf::LOCAL: ptr = local[r][i]; break;
          case PlanBuf::PEER:
            if (!active(b.rank)) throw std::runtime_error("plan needs peer-to-peer pointers to a rank that lives in another process; use the proxy (send_proxy) topology for multi-process runs");
            ptr = local[b.rank][b.idx]; break;
          case PlanBuf::INBOX:
            if (!active(b.rank)) throw std::runtime_error("plan writes another rank's inbox directly (put_signal); multi-process runs need send_proxy links");
            ptr = dd[b.rank]->inbox_buf(); break;
          case PlanBuf::SENDQ: ptr = dd[r]->sendq_buf(); rows = 1; cols = (int)dd[r]->sendq_floats(); break;
        }
        if (ex[r]->add_view(ptr, rows, cols, b.i32 ? 1 : 0) != (int)i) throw std::logic_error("buffer index mismatch");
      }
      for (auto& v : p.vars) if (v.second) ex[r]->set_version(v.first, v.second);
      ex[r]->set_dist(dd[r]->dev());
    }
    // every executor allocates BEFORE any kernel runs: cudaMalloc may wait for all kernels on the device (see KvmExec::prepare)
    for (int r = 0; r < W; ++r) if (active(r)) { CK(cudaSetDevice(dev[r])); ex[r]->prepare(); }
    bool need_net = false; for (auto& p : pr) need_net |= p.uses_sendq();
    if (need_net) {
      net.build(W, tcp, f, only, hosts, ports);
      for (int r = 0; r < W; ++r) {
        if (!active(r)) continue;
        std::vector<DevBuf> tbl; for (size_t i = 0; i < pr[r].bufs.size(); ++i) tbl.push_back({pr[r].bufs[i].kind == PlanBuf::LOCAL ? local[r][i] : nullptr, pr[r].bufs[i].rows, pr[r].bufs[i].cols});
        sink[r] = std::make_unique<CudaSink>(dev[r], tbl, *dd[r]);
        px[r] = std::make_unique<Proxy>(r, W, dd[r]->make_sendq(), *net.rel[r], *sink[r]);
        px[r]->start();
      }
    }
  }

  // A timed-out rank: which tasks never ran, what they waited for, and what the inbox and the executor's version said. Tells a lost
  // signal (inbox 0) from a lost wakeup (inbox >= need, version < need) from a lost wake after the version moved.
  void diagnose(int r, const KvmResult& res) {
    std::vector<unsigned> inbox = dd[r]->read_inbox();
    int shown = 0, stuck = 0;
    for (size_t i = 0; i < res.tasks.size() && i < res.trace.size(); ++i) {
      if (res.trace[i].exec_count != 0) continue;
      ++stuck;
      const KvmTaskDesc& t = res.tasks[i];
      if (shown++ >= 6) continue;
      const bool w = t.wait_var >= 0;
      fprintf(stderr, "  diagnosis rank %d: task %zu (%s) never ran; wait var %d >= %d; executor version %d, inbox %u%s\n", r, i, kvm_op(t.op_id)->name, t.wait_var,
              t.wait_ver, w ? res.version[t.wait_var] : -1, w ? inbox[t.wait_var] : 0u,
              w && (int)inbox[t.wait_var] >= t.wait_ver && res.version[t.wait_var] < t.wait_ver ? "   <- signal arrived, pump did not publish it" :
              w && res.version[t.wait_var] >= t.wait_ver ? "   <- version reached, task never woken (lost wakeup)" : "");
    }
    const KvmDistCtl* c = dd[r]->ctl();
    fprintf(stderr, "  diagnosis rank %d: %d tasks never ran; pump polled %llu times, last publish var %d ver %d; in_vars:", r, stuck, (unsigned long long)c->polls, c->last_pub_var, c->last_pub_ver);
    for (int v : pr[r].in_vars) fprintf(stderr, " %d(inbox %u, version %d)", v, inbox[v], res.version[v]);
    fprintf(stderr, "\n");
  }

  std::vector<RankOut> run() {
    int W = (int)pr.size();
    std::vector<RankOut> out(W); std::vector<std::thread> th;
    std::vector<std::atomic<int>> where(W);       // 0 not started, 1 in start(), 2 in submit(), 3 in finish(), 4 done: for the watchdog
    for (auto& x : where) x = 0;
    std::atomic<bool> all_done{false};
    std::thread watchdog([&] {                      // a rank thread stuck outside the kernel is the failure we are hunting: say where
      for (int i = 0; i < 150 && !all_done; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (all_done) return;
      for (int r = 0; r < W; ++r) if (active(r)) fprintf(stderr, "  watchdog (15 s): rank %d thread is %s\n", r, (const char*[]){"before start()", "in start()", "in submit()", "in finish()", "done"}[where[r].load()]);
    });
    // Kernels are launched from THIS thread, one executor after another, before any rank thread runs. Launching the persistent kernels
    // of several executors concurrently from one host thread each left one executor's kernel unscheduled until its peers timed out
    // in ~7% of runs (6/80 vs 0/80 for sequential or 3 ms-staggered launches; see DISTRIBUTED.md). KVM_DIST_LAUNCH=threads restores
    // the concurrent launch for experiments; KVM_DIST_STAGGER_MS=n sleeps n ms between launches.
    const bool main_launch = !(getenv("KVM_DIST_LAUNCH") && std::string(getenv("KVM_DIST_LAUNCH")) == "threads");
    const int stagger = getenv("KVM_DIST_STAGGER_MS") ? atoi(getenv("KVM_DIST_STAGGER_MS")) : 0;
    if (main_launch) for (int r = 0; r < W; ++r) if (active(r)) { CK(cudaSetDevice(dev[r])); ex[r]->start(); if (stagger) std::this_thread::sleep_for(std::chrono::milliseconds(stagger)); }
    for (int r = 0; r < W; ++r) {
      if (!active(r)) continue;
      if (stagger && !main_launch) std::this_thread::sleep_for(std::chrono::milliseconds(stagger));
      th.emplace_back([&, r] {
        try {
          CK(cudaSetDevice(dev[r]));
          auto t0 = Clock::now();
          where[r] = 1;
          if (!main_launch) ex[r]->start();
          where[r] = 2;
          out[r].start_ms = (double)ms_since(t0);
          KvmSubmit s = ex[r]->submit(pr[r].bundle);
          out[r].submit_ms = (double)ms_since(t0);
          if (s != KvmSubmit::OK && s != KvmSubmit::DEVICE_ERROR) throw std::runtime_error(std::string("submit: ") + kvm_submit_name(s));   // DEVICE_ERROR: the device already failed (e.g. dead peer); finish() reports which
          where[r] = 3;
          out[r].res = ex[r]->finish();
          where[r] = 4;
          out[r].ms = (double)ms_since(t0);
          out[r].finish_ms = out[r].ms;
          if (px[r]) px[r]->kernel_done();
          out[r].ok = out[r].res.ok();
          if (!out[r].ok) out[r].msg = std::string(kvm_err_name(out[r].res.err_code)) + " (" + kvm_err_text(out[r].res.err_code) + ") slot " + std::to_string(out[r].res.err_slot) + " a=" + std::to_string(out[r].res.err_a);
        } catch (const std::exception& e) { out[r].ok = false; out[r].msg = e.what(); if (px[r]) px[r]->kernel_done(); }
      });
    }
    for (auto& t : th) t.join();
    all_done = true; watchdog.join();
    for (int r = 0; r < W; ++r) if (px[r]) { if (!px[r]->wait_finished(out[r].ok ? 60000 : 3000)) { out[r].msg += " [proxy shutdown incomplete]"; } }
    for (int r = 0; r < W; ++r) if (px[r]) px[r]->stop();
    bool any_timeout = false;
    for (int r = 0; r < W; ++r) any_timeout |= active(r) && !out[r].ok && out[r].res.err_code == KVM_ERR_TIMEOUT;
    if (any_timeout) for (int r = 0; r < W; ++r) if (active(r))
      fprintf(stderr, "  host timeline rank %d: start() returned at %.0f ms, submit() at %.0f ms, finish() at %.0f ms; kernel %.3f ms; err %s\n", r, out[r].start_ms, out[r].submit_ms,
              out[r].finish_ms, out[r].res.kernel_ms, kvm_err_name(out[r].res.err_code));
    for (int r = 0; r < W; ++r) if (active(r) && !out[r].ok && out[r].res.err_code == KVM_ERR_TIMEOUT) diagnose(r, out[r].res);
    return out;
  }
};

static void report(const char* label, std::vector<RankOut>& out, World& w) {
  for (size_t r = 0; r < out.size(); ++r) if (w.active((int)r)) {
    CHECK(out[r].ok, "%s rank %zu: %s", label, r, out[r].msg.c_str());
    g_timeouts += out[r].res.err_code == KVM_ERR_TIMEOUT;
  }
}

// ==================================================================================================== ping-pong across executors
// Rank 0 and rank 1 alternate R rounds. Round r: rank0 computes work = r * src and publishes it into rank 1's landing region r
// (a FRESH region, so a stale read is visible as zeros); rank 1 waits on its inbox variable (version >= r), copies the landing
// region back into rank 0's `back` region r and signals; rank 0's next round waits for that. At the end rank 0 checks that back
// == r * src for every r. If the signal could overtake the data (or the wait did not acquire), some back[r] holds stale zeros.
static std::vector<PlanRank> make_pingpong(int R, int N, bool proxy) {
  std::vector<PlanRank> pr(2);
  for (int r = 0; r < 2; ++r) { pr[r].rank = r; pr[r].world = 2; pr[r].nvars = 4; pr[r].vars = {{0, 0}, {1, 0}}; pr[r].in_vars = {0}; }
  auto local = [](int rows, int cols, float v = 0) { PlanBuf b; b.rows = rows; b.cols = cols; b.fillv = v; return b; };
  auto peer = [](int rank, int idx, int rows, int cols) { PlanBuf b; b.kind = PlanBuf::PEER; b.rank = rank; b.idx = idx; b.rows = rows; b.cols = cols; return b; };
  auto inbox = [](int rank) { PlanBuf b; b.kind = PlanBuf::INBOX; b.rank = rank; b.rows = 1; b.cols = 4; return b; };
  PlanBuf sq; sq.kind = PlanBuf::SENDQ;
  // rank 0: 0 src[N] (values 1..), 1 work[N], 2 back[R*N], 3 landing1 (peer of rank1 buf 0), 4 inbox of rank 1, 5 sendq
  PlanBuf src = local(1, N); src.fill = false; src.vals.resize(N); for (int i = 0; i < N; ++i) src.vals[i] = (float)(i % 13 + 1);
  pr[0].bufs = {src, local(1, N), local(1, R * N), peer(1, 0, 1, R * N), inbox(1), sq};
  // rank 1: 0 landing[R*N], 1 back0 (peer of rank0 buf 2), 2 inbox of rank 0, 3 sendq
  pr[1].bufs = {local(1, R * N), peer(0, 2, 1, R * N), inbox(0), sq};
  if (!proxy) { pr[0].bufs.pop_back(); pr[1].bufs.pop_back(); }   // peer pointers only: no send queue, so no proxy threads either
  if (proxy) { pr[0].bufs[3] = local(1, 1); pr[0].bufs[4] = local(1, 1); pr[1].bufs[1] = local(1, 1); pr[1].bufs[2] = local(1, 1); }   // no peer pointers: all traffic via the queue
  KvmBundleBuilder b0, b1;
  for (int r = 1; r <= R; ++r) {
    KvmTaskOpts o; o.wait_var = r > 1 ? 0 : -1; o.wait_ver = r - 1;
    int s = b0.add("scale", {Bf(0), Bf(1), F((float)r), I(N)}, o);
    int p = proxy ? b0.add("send_proxy", {Bf(1), Bf(5), I(0), I(1), I(N), I(N), I(1), I(0), I((r - 1) * N), I(N), I(0), I(r)})
                  : b0.add("put_signal", {Bf(1), Bf(3), I(0), I((r - 1) * N), I(1), I(N), I(N), I(N), Bf(4), I(0), I(r)});
    b0.edge(s, p);
    KvmTaskOpts o1; o1.wait_var = 0; o1.wait_ver = r;
    if (proxy) b1.add("send_proxy", {Bf(0), Bf(3), I((r - 1) * N), I(1), I(N), I(N), I(0), I(2), I((r - 1) * N), I(N), I(0), I(r)}, o1);
    else b1.add("put_signal", {Bf(0), Bf(1), I((r - 1) * N), I((r - 1) * N), I(1), I(N), I(N), I(N), Bf(2), I(0), I(r)}, o1);
  }
  // rank 0 final waiter so the kernel only ends after the last reply arrived (and nothing is left in flight)
  KvmTaskOpts ow; ow.wait_var = 0; ow.wait_ver = R; b0.add("noop", {}, ow);
  pr[0].bundle = b0.build(); pr[1].bundle = b1.build();
  return pr;
}

static void check_pingpong(World& w, int R, int N, const char* label) {
  std::vector<float> back(R * N); CK(cudaSetDevice(w.dev[0])); CK(cudaMemcpy(back.data(), w.local[0][2], back.size() * 4, cudaMemcpyDeviceToHost));
  long stale = 0, zeros = 0;
  for (int r = 1; r <= R; ++r) for (int i = 0; i < N; ++i) {
    float want = (float)r * (float)(i % 13 + 1), got = back[(size_t)(r - 1) * N + i];
    if (!same_bits(got, want)) { ++stale; zeros += got == 0.f; }
  }
  g_stale += stale > 0;
  CHECK(stale == 0, "%s: %ld of %d values stale/wrong (%ld are zeros: the signal overtook the data)", label, stale, R * N, zeros);
}

static int device_of(int r) { return r % std::max(g_ngpu, 1); }

static void suite_pingpong(bool proxy, FaultSpec f = {}, bool tcp = false, int R = 200, int N = 4096) {
  char label[160]; snprintf(label, sizeof label, "%s%s%s", proxy ? "proxy" : "p2p", tcp ? "+tcp" : "", (f.drop > 0 || f.dup > 0 || f.delay > 0) ? "+faults" : "");
  fprintf(stderr, "== pingpong[%s]: %d rounds x %d floats between two executors (%d device%s)\n", label, R, N, std::min(g_ngpu, 2), g_ngpu > 1 ? "s" : "");
  World w; w.pr = make_pingpong(R, N, proxy); w.dev = {device_of(0), device_of(1)};
  w.cfg.blocks = 4; w.cfg.timeout_us = 20ull * 1000000;
  w.build(tcp, f);
  auto t0 = Clock::now();
  auto out = w.run();
  double ms = ms_since(t0);
  report(label, out, w);
  if (!out[0].ok || !out[1].ok) {      // diagnosis for the log: how far did each side get?
    std::vector<unsigned> i0 = w.dd[0]->read_inbox(), i1 = w.dd[1]->read_inbox();
    fprintf(stderr, "  diagnosis: inbox[0] = %u on rank 0, %u on rank 1 (rounds published to each side); executor version[0] = %d / %d; tasks done %d / %d\n",
            i0[0], i1[0], out[0].res.version.empty() ? -1 : out[0].res.version[0], out[1].res.version.empty() ? -1 : out[1].res.version[0], out[0].res.done, out[1].res.done);
  }
  if (out[0].ok && out[1].ok) {
    check_pingpong(w, R, N, label);
    std::vector<unsigned> in0 = w.dd[0]->read_inbox(), in1 = w.dd[1]->read_inbox();
    CHECK(in0[0] == (unsigned)R && in1[0] == (unsigned)R, "inbox versions %u/%u, want %d", in0[0], in1[0], R);
    CHECK(out[0].res.version[0] == R && out[1].res.version[0] == R, "executor versions %d/%d, want %d", out[0].res.version[0], out[1].res.version[0], R);
    fprintf(stderr, "  %d round trips in %.1f ms = %.1f us per round trip including setup (two cross-executor version hops + two %d-float tile copies each)\n", R, ms, ms * 1000 / R, N);
    if (w.dev[0] == w.dev[1] && !proxy) {   // one GPU = one %globaltimer: the executors' own timestamps give the hop latency
      std::vector<double> h01, h10;
      const auto &t0 = out[0].res.trace, &t1 = out[1].res.trace;
      for (int r = 0; r < R; ++r) {
        const KvmTrace &put = t0[2 * r + 1], &back = t1[r];                 // rank0: scale, put_signal per round; rank1: one put_signal
        h01.push_back((double)((long long)back.start_ns - (long long)put.end_ns) / 1e3);
        if (r + 1 < R) h10.push_back((double)((long long)t0[2 * (r + 1)].start_ns - (long long)back.end_ns) / 1e3);
      }
      std::sort(h01.begin(), h01.end()); std::sort(h10.begin(), h10.end());
      fprintf(stderr, "  cross-executor version hop (publisher task end -> waiter task start, device timestamps): rank0->1 median %.1f us (p90 %.1f), rank1->0 median %.1f us (p90 %.1f)\n",
              h01[h01.size() / 2], h01[h01.size() * 9 / 10], h10[h10.size() / 2], h10[h10.size() * 9 / 10]);
    }
  }
}

// ==================================================================================================== hang-free failure behaviour
static void suite_errors() {
  fprintf(stderr, "== errors: nothing hangs\n");
  {   // a wait nobody will ever satisfy: the executor's own timeout ends it (device-side), with a decoded code
    KvmConfig c = small_cfg(3, 8); c.timeout_us = 800 * 1000;
    KvmExec ex(c); DistDevice dd(0, 1, 0, 8); dd.set_in_vars({1}); ex.set_dist(dd.dev());
    ex.add_buffer(1, 4);
    KvmBundleBuilder bb; KvmTaskOpts o; o.wait_var = 1; o.wait_ver = 1; bb.add("noop", {}, o);
    ex.start(); ex.submit(bb.build()); auto t0 = Clock::now(); const KvmResult& r = ex.finish();
    CHECK(r.err_code == KVM_ERR_TIMEOUT, "unsatisfiable remote wait: got %s", kvm_err_name(r.err_code));
    CHECK(ms_since(t0) < 5000, "took %d ms", ms_since(t0));
    fprintf(stderr, "  never-signalled remote wait -> %s after %d ms\n", kvm_err_name(r.err_code), ms_since(t0));
  }
  {   // send queue full and nobody draining it: the producing task times out on the device
    KvmExec ex(small_cfg(3, 8)); DistDevice dd(0, 1, 0, 8, 2, 256, 300ull * 1000000);
    int src = ex.add_buffer(1, 64), q = ex.add_view(dd.sendq_buf(), 1, (int)dd.sendq_floats());
    KvmBundleBuilder bb; int prev = -1;
    for (int i = 0; i < 4; ++i) { int t = bb.add("send_proxy", {Bf(src), Bf(q), I(0), I(1), I(8), I(8), I(1), I(0), I(0), I(8), I(0), I(i + 1)}); if (prev >= 0) bb.edge(prev, t); prev = t; }
    ex.start(); ex.submit(bb.build()); auto t0 = Clock::now(); const KvmResult& r = ex.finish();
    CHECK(r.err_code == KVM_ERR_DIST_SENDQ_TIMEOUT, "full queue: got %s", kvm_err_name(r.err_code));
    fprintf(stderr, "  full send queue, no proxy -> %s after %d ms\n", kvm_err_name(r.err_code), ms_since(t0));
  }
  {   // proxy stops heartbeating: the pump fails the device
    KvmConfig c = small_cfg(3, 8); c.timeout_us = 10ull * 1000000;
    KvmExec ex(c); DistDevice dd(0, 1, 0, 8, 4, 256, 1000000000ull, 300ull * 1000000); dd.set_in_vars({1}); ex.set_dist(dd.dev());
    ex.add_buffer(1, 4);
    dd.ctl()->heartbeat = 1;                    // a proxy that beat once and then died
    KvmBundleBuilder bb; KvmTaskOpts o; o.wait_var = 1; o.wait_ver = 1; bb.add("noop", {}, o);
    ex.start(); ex.submit(bb.build()); auto t0 = Clock::now(); const KvmResult& r = ex.finish();
    CHECK(r.err_code == KVM_ERR_DIST_PROXY_DEAD, "dead proxy: got %s", kvm_err_name(r.err_code));
    CHECK(ms_since(t0) < 5000, "took %d ms", ms_since(t0));
    fprintf(stderr, "  silent proxy -> %s after %d ms\n", kvm_err_name(r.err_code), ms_since(t0));
  }
  {   // peer rank dies: the proxy tells the device, the device ends with PEER_DOWN
    World w; w.pr = make_pingpong(5, 256, true); w.dev = {device_of(0), device_of(1)}; w.only = 0; w.cfg.timeout_us = 20ull * 1000000;
    w.build(false, FaultSpec{});
    w.net.hub->kill(1);
    auto t0 = Clock::now(); auto out = w.run();
    CHECK(!out[0].ok && out[0].res.err_code == KVM_ERR_DIST_PEER_DOWN, "dead peer: rank0 %s", out[0].ok ? "succeeded" : out[0].msg.c_str());
    CHECK(ms_since(t0) < 8000, "took %d ms", ms_since(t0));
    fprintf(stderr, "  peer rank killed -> %s after %d ms\n", kvm_err_name(out[0].res.err_code), ms_since(t0));
  }
}

// ==================================================================================================== pump during admission backpressure
// Parked (version-waiting) tasks have their own budget of live_cap. Device 1 admits two bundles of 8 tasks parked on a remote
// variable (16 parked > live_cap 8); a third bundle then waits for the parked count to drop. The signal that releases them arrives
// while the admitter is blocked in that wait: the pump must keep running there.
static void suite_backpressure() {
  fprintf(stderr, "== backpressure: remote publishes keep flowing while admission waits for room\n");
  DistDevice d0(0, 2, device_of(0), 4), d1(1, 2, device_of(1), 4);
  d1.set_in_vars({0});
  if (device_of(0) != device_of(1)) enable_peer_access({device_of(0), device_of(1)});
  CK(cudaSetDevice(device_of(1)));
  KvmConfig c1 = small_cfg(3, 4); c1.live_cap = 8; c1.timeout_us = 8ull * 1000000;
  KvmExec e1(c1); e1.add_buffer(1, 4); e1.set_dist(d1.dev());
  CK(cudaSetDevice(device_of(0)));
  KvmExec e0(small_cfg(3, 4)); int flag = e0.add_view(d1.inbox_buf(), 1, 4);
  CK(cudaSetDevice(device_of(1))); e1.prepare();
  CK(cudaSetDevice(device_of(0))); e0.prepare();          // before either kernel runs (cudaMalloc can wait for running kernels)
  KvmBundleBuilder a, a2, b, s;
  for (int i = 0; i < 8; ++i) { KvmTaskOpts o; o.wait_var = 0; o.wait_ver = 1; a.add("noop", {}, o); a2.add("noop", {}, o); }
  b.add("noop", {});
  s.add("signal", {Bf(flag), I(0), I(1)});
  KvmResult r1;
  std::thread t1([&] {
    CK(cudaSetDevice(device_of(1)));
    e1.start(); e1.submit(a.build()); e1.submit(a2.build()); e1.submit(b.build()); r1 = e1.finish();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));     // let the window fill and admission block
  CK(cudaSetDevice(device_of(0)));
  e0.start(); e0.submit(s.build()); const KvmResult& r0 = e0.finish();
  t1.join();
  CHECK(r0.ok(), "signaller: %s", kvm_err_name(r0.err_code));
  g_timeouts += r1.err_code == KVM_ERR_TIMEOUT || r1.err_code == KVM_ERR_PARK_STALL;
  CHECK(r1.ok() && r1.done == 17, "receiver: %s, %d/17 tasks done", kvm_err_name(r1.err_code), r1.done);
}

// ==================================================================================================== bandwidth (needs 2 GPUs for the NVLink number)
static void suite_bw() {
  fprintf(stderr, "== bw: device-initiated tile copies (put) between two buffers\n");
  const size_t BYTES = 256u << 20, TILE = 512u << 10;     // 512 KiB tiles: 512 tasks keep every SM busy
  auto measure = [&](int src_dev, int dst_dev, const char* label) {
    float *a, *b;
    CK(cudaSetDevice(dst_dev)); CK(cudaMalloc(&b, BYTES));
    CK(cudaSetDevice(src_dev)); CK(cudaMalloc(&a, BYTES)); CK(cudaMemset(a, 1, BYTES));
    if (src_dev != dst_dev) enable_peer_access({src_dev, dst_dev});
    CK(cudaSetDevice(src_dev));
    KvmConfig c = small_cfg(0, 8); c.blocks = 0; KvmExec ex(c);
    int ia = ex.add_view(a, 1, (int)(BYTES / 4)), ib = ex.add_view(b, 1, (int)(BYTES / 4));
    KvmBundleBuilder bb; int n = (int)(BYTES / TILE);
    for (int i = 0; i < n; ++i) bb.add("put", {Bf(ia), Bf(ib), I((int)(i * TILE / 4)), I((int)(i * TILE / 4)), I(1), I((int)(TILE / 4)), I((int)(TILE / 4)), I((int)(TILE / 4))});
    ex.start(); ex.submit(bb.build()); const KvmResult& r = ex.finish();
    CHECK(r.ok(), "device error %s", kvm_err_name(r.err_code));
    double gbs = (double)BYTES / (r.kernel_ms * 1e-3) / 1e9;
    fprintf(stderr, "  %-34s %6.1f GB/s  (%zu MiB in %.2f ms, %d blocks)\n", label, gbs, BYTES >> 20, r.kernel_ms, r.grid);
    CK(cudaSetDevice(src_dev)); cudaFree(a); CK(cudaSetDevice(dst_dev)); cudaFree(b);
  };
  measure(0, 0, "same device (HBM read+write)");
  if (g_ngpu >= 2) {
    int can = 0; cudaDeviceCanAccessPeer(&can, 0, 1);
    if (can) { measure(0, 1, "device 0 -> 1 (peer stores)"); measure(1, 0, "device 1 -> 0 (peer stores)"); }
    else { fprintf(stderr, "  SKIP: devices 0 and 1 cannot access each other peer-to-peer\n"); ++g_skip; }
  } else { fprintf(stderr, "  SKIP: needs 2 GPUs (peer-to-peer / NVLink bandwidth)\n"); ++g_skip; }
}

// ==================================================================================================== plan runner
static int suite_plan(int argc, char** argv) {
  std::string dir; bool tcp = false; FaultSpec f; int only = -1, blocks = 4; std::vector<std::string> hosts; std::vector<int> ports; double tol = -1; double timeout_s = -1; std::string dump_dir;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--net" && i + 1 < argc) tcp = std::string(argv[++i]) == "tcp";
    else if (a == "--rank" && i + 1 < argc) only = atoi(argv[++i]);
    else if (a == "--blocks" && i + 1 < argc) blocks = atoi(argv[++i]);
    else if (a == "--tol" && i + 1 < argc) tol = atof(argv[++i]);
    else if (a == "--timeout" && i + 1 < argc) timeout_s = atof(argv[++i]);
    else if (a == "--dump" && i + 1 < argc) dump_dir = argv[++i];
    else if (a == "--fault" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) { if (!t.rfind("drop=", 0)) f.drop = atof(t.c_str() + 5); else if (!t.rfind("dup=", 0)) f.dup = atof(t.c_str() + 4); else if (!t.rfind("delay=", 0)) f.delay = atof(t.c_str() + 6); } }
    else if (a == "--peers" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) { auto c = t.rfind(':'); hosts.push_back(t.substr(0, c)); ports.push_back(atoi(t.c_str() + c + 1)); } }
    else if (dir.empty()) dir = a;
  }
  if (dir.empty()) { fprintf(stderr, "usage: plan DIR [--net tcp] [--fault ...] [--rank R --peers h:p,...]\n"); return 2; }
  World w; int W = 0;
  while (access((dir + "/rank" + std::to_string(W)).c_str(), F_OK) == 0) ++W;
  if (!W) { fprintf(stderr, "no rank* directories in %s\n", dir.c_str()); return 2; }
  for (int r = 0; r < W; ++r) w.pr.push_back(load_rank(dir + "/rank" + std::to_string(r)));
  for (int r = 0; r < W; ++r) w.dev.push_back(device_of(r));
  w.only = only; w.cfg = small_cfg(blocks); w.cfg.timeout_us = (only >= 0 ? 180ull : 20ull) * 1000000;
  w.cfg.max_tasks = 1 << 16; w.cfg.live_cap = 1 << 12; w.cfg.ring_words = 1 << 20;   // big plans: thousands of tasks per rank (the comment on the line above used to swallow these)   // processes/hosts start with skew w.cfg.max_tasks = 1 << 16; w.cfg.live_cap = 1 << 13;
  if (timeout_s > 0) w.cfg.timeout_us = (unsigned long long)(timeout_s * 1e6);
  fprintf(stderr, "== plan %s: %d ranks on %d GPU(s), net %s%s\n", dir.c_str(), W, g_ngpu, only >= 0 ? "tcp (multi-process)" : tcp ? "tcp" : "loopback", (f.drop > 0 || f.dup > 0 || f.delay > 0) ? " + faults" : "");
  if (only >= 0 && hosts.size() != (size_t)W) { fprintf(stderr, "--rank needs --peers with %d entries\n", W); return 2; }
  w.build(tcp || only >= 0, f, hosts, ports);
  { size_t tot = 0; for (int r = 0; r < W; ++r) { if (!w.active(r)) continue; tot += w.local_bytes[r];
      fprintf(stderr, "  MEM rank %d: %zu local buffers, %zu bytes\n", r, w.local_count[r], w.local_bytes[r]); }
    fprintf(stderr, "  MEM total local buffer bytes %zu\n", tot);
    size_t fr = 0, tt = 0; cudaMemGetInfo(&fr, &tt); fprintf(stderr, "  MEM device used after allocation (cudaMemGetInfo, whole device incl. other users) %zu bytes\n", tt - fr); }
  auto out = w.run();
  report("plan", out, w);
  long cmp = 0, bad = 0, exact = 0; double worst = 0;
  for (int r = 0; r < W; ++r) {
    if (!w.active(r) || !out[r].ok) continue;
    fprintf(stderr, "  rank %d: %d tasks, %.2f ms kernel, device %d\n", r, out[r].res.done, out[r].ms, w.dev[r]);
    for (auto& kv : w.pr[r].expect) {
      std::vector<float> got = download(w.local[r][kv.first], kv.second.size());
      for (size_t i = 0; i < got.size(); ++i) {
        ++cmp; if (same_bits(got[i], kv.second[i])) { ++exact; continue; }
        double rel = std::fabs((double)got[i] - kv.second[i]) / std::max(1.0, (double)std::fabs(kv.second[i])); worst = std::max(worst, rel);
        if (tol < 0 || rel > tol) { if (bad++ < 5) fprintf(stderr, "  MISMATCH rank %d buf %d [%zu] = %.9g want %.9g\n", r, kv.first, i, got[i], kv.second[i]); }
      }
    }
  }
  if (!dump_dir.empty()) {
    FILE* man = fopen((dump_dir + "/manifest.txt").c_str(), "w");
    for (int r = 0; r < W; ++r) {
      if (!w.active(r) || !out[r].ok) continue;
      for (auto& d : w.pr[r].dumps) {
        const PlanBuf& b = w.pr[r].bufs[d.idx]; size_t n = (size_t)b.rows * b.cols;
        std::vector<float> got = download(w.local[r][d.idx], n);
        std::string fn = "rank" + std::to_string(r) + "_" + d.name + ".bin";
        FILE* f = fopen((dump_dir + "/" + fn).c_str(), "wb"); fwrite(got.data(), 4, n, f); fclose(f);
        fprintf(man, "%d %s %d %d %d %d %d %d %s\n", r, d.name.c_str(), b.rows, b.cols, d.r0, d.c0, d.gr, d.gc, fn.c_str());
      }
    }
    fclose(man);
  }
  CHECK(bad == 0, "%ld of %ld expected values differ", bad, cmp);
  fprintf(stderr, "  %ld values compared, %ld bitwise equal, worst relative difference %.3g\n", cmp, exact, worst);
  return 0;
}

// ==================================================================================================== main
int main(int argc, char** argv) {
  g_ngpu = device_count();
  bool expect_fail = false; std::string kind; std::vector<std::string> suites;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--expect-fail") expect_fail = true;
    else if (a.rfind("--expect-fail=", 0) == 0) { expect_fail = true; kind = a.substr(14); }   // stale | timeout: the way it must fail
    else suites.push_back(a);
  }
  if (g_ngpu < 1) { fprintf(stderr, "no CUDA device\n"); return 3; }
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  fprintf(stderr, "test_dist: %d GPU(s); device 0 = %s, %d SMs, sm_%d%d\n", g_ngpu, p.name, p.multiProcessorCount, p.major, p.minor);
  try {
    if (!suites.empty() && suites[0] == "plan") { suite_plan(argc, argv); }
    else {
      bool all = suites.empty();
      auto want = [&](const char* s) { return all || std::find(suites.begin(), suites.end(), s) != suites.end(); };
      if (want("ops")) suite_ops();
      if (want("pingpong")) {
        suite_pingpong(false);
        if (expect_fail) { /* the negative build must already have failed above */ }
        else {
          suite_pingpong(true, FaultSpec{}, false, 100);
          suite_pingpong(true, FaultSpec{}, true, 100);
          FaultSpec f; f.drop = 0.1; f.dup = 0.1; f.delay = 0.2; f.delay_ms_max = 10; f.seed = 3;
          suite_pingpong(true, f, true, 60);
        }
      }
      if (want("errors") && !expect_fail) suite_errors();
      if (want("backpressure")) suite_backpressure();
      if (want("bw") && !expect_fail) suite_bw();
    }
  } catch (const std::exception& e) { fprintf(stderr, "EXCEPTION: %s\n", e.what()); ++g_fail; }
  if (expect_fail) {
    bool right = g_fail > 0 && (kind.empty() || (kind == "stale" && g_stale > 0) || (kind == "timeout" && g_timeouts > 0));
    printf("%s: negative test %s (%d checks failed; stale-data detections %d, device timeouts %d%s%s)\n", right ? "PASS" : "FAIL",
           right ? "detected the violation" : g_fail ? "failed, but not in the expected way" : "did NOT detect the violation", g_fail, g_stale, g_timeouts, kind.empty() ? "" : "; expected ", kind.c_str());
    return right ? 0 : 1;
  }
  printf("%s: %d checks, %d failed, %d skipped stage(s)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail, g_skip);
  return g_fail ? 1 : 0;
}
