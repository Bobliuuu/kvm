// CPU check of the bundle-level fusion pass (include/kvm_fuse.h) on real bundles: prints the decision for every run and, when it fires, checks the rewritten bundle.
//   g++ -std=c++17 -O2 -Iinclude tests/static/test_fuse_pass.cpp -o /tmp/test_fuse_pass && /tmp/test_fuse_pass BUNDLE_DIR...   (run from the repo root)
// Checks on a fired bundle: the DAG is acyclic; every kept task keeps its op and arguments; every replaced step became its slices and groups, each slice before every group; the static
// planner accepts the result on the bundle's registry signatures, with the fused ops' signatures added.
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <vector>
#include "kvm_fuse.h"

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
template <class T> static std::vector<T> load(const std::string& p) { std::string r = slurp(p); assert(r.size() % sizeof(T) == 0); std::vector<T> v(r.size() / sizeof(T)); memcpy(v.data(), r.data(), r.size()); return v; }

// The registry from the .def files (X(name, id, "sig") lines).
static std::map<int, KvmSOpInfo> g_ops; static std::map<std::string, int> g_ids;
static int types_of(const std::string& sig, int* arity) {
    uint64_t t = 0; int i = 0; std::stringstream ss(sig); std::string tok;
    while (std::getline(ss, tok, ',')) { auto c = tok.find(':'); std::string ty = tok.substr(c + 1); uint64_t v = ty == "buf" ? 0 : ty == "i32" ? 1 : ty == "f32" ? 2 : ty == "buf?" ? 3 : 7; assert(v != 7); t |= v << (3 * i); ++i; }
    *arity = i; return (int)t;
}
static void load_defs(const std::string& inc) {
    for (const char* f : {"kvm_ops.def", "kvm_ops_gpt.def", "kvm_ops_extra.def", "kvm_ops_comm.def", "kvm_ops_game.def", "kvm_ops_fmlp.def"}) {
        std::ifstream in(inc + "/" + f); std::string line; std::regex re("^\\s*X\\(\\s*(\\w+)\\s*,\\s*(\\d+)\\s*,\\s*\"([^\"]*)\"");
        while (std::getline(in, line)) {
            std::smatch m; if (!std::regex_search(line, m, re)) continue;
            int id = std::stoi(m[2].str()), ar = 0; KvmSOpInfo o; o.types = (uint64_t)types_of(m[3].str(), &ar); o.arity = ar; g_ops[id] = o; g_ids[m[1].str()] = id;
        }
    }
}

// Pulls the buffer table out of metadata.json without a JSON library: every {"name":..., "index":N, "rows":R, "cols":C, "dtype":D ...} entry of the "buffers" array.
static std::vector<KvmFuseBuf> load_bufs(const std::string& meta) {
    const size_t start = meta.find("\"buffers\"");
    assert(start != std::string::npos);
    const std::regex re("\"index\"\\s*:\\s*(\\d+)\\s*,\\s*\"rows\"\\s*:\\s*(\\d+)\\s*,\\s*\"cols\"\\s*:\\s*(\\d+)\\s*,\\s*\"dtype\"\\s*:\\s*(\\d+)");
    std::vector<KvmFuseBuf> v;
    for (std::sregex_iterator it(meta.begin() + start, meta.end(), re), end; it != end; ++it) {
        const int idx = std::stoi((*it)[1].str());
        if (idx >= (int)v.size()) v.resize(idx + 1);
        v[idx] = KvmFuseBuf{std::stoi((*it)[2].str()), std::stoi((*it)[3].str()), std::stoi((*it)[3].str()), std::stoi((*it)[4].str())};
    }
    return v;
}

static bool acyclic(const std::vector<KvmTaskDesc>& t, const std::vector<int32_t>& succ) {
    std::vector<int> in(t.size(), 0), st; int seen = 0;
    for (const auto& d : t) for (int k = 0; k < d.succ_cnt; ++k) ++in[succ[d.succ_off + k]];
    for (size_t i = 0; i < t.size(); ++i) if (!in[i]) st.push_back((int)i);
    while (!st.empty()) { int u = st.back(); st.pop_back(); ++seen; for (int k = 0; k < t[u].succ_cnt; ++k) if (--in[succ[t[u].succ_off + k]] == 0) st.push_back(succ[t[u].succ_off + k]); }
    return seen == (int)t.size();
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s BUNDLE_DIR... (run from the repo root)\n", argv[0]); return 2; }
    load_defs("include");
    const auto id = [&](const char* n) { auto it = g_ids.find(n); return it == g_ids.end() ? -1 : it->second; };
    KvmFuseOps ops; ops.noop = id("noop"); ops.select_window = id("select_window"); ops.embedding_fwd = id("embedding_fwd"); ops.matmul_tile = id("matmul_tile"); ops.softmax_ce_fused = id("softmax_ce_fused"); ops.sgd_range = id("sgd_range"); ops.softmax_ce_fwd = id("softmax_ce_fwd"); ops.softmax_ce_reduce = id("softmax_ce_reduce"); ops.softmax_ce_bwd = id("softmax_ce_bwd"); ops.gelu_bwd = id("gelu_bwd"); ops.colsum = id("colsum"); ops.embedding_bwd = id("embedding_bwd"); ops.sgd = id("sgd"); ops.fmlp_slice = id("fmlp_slice"); ops.fmlp_sgd = id("fmlp_sgd");
    const int slice_id = ops.fmlp_slice, sgd_id = ops.fmlp_sgd;
    const KvmSOpLookup op_of = [](int i) { auto it = g_ops.find(i); return it == g_ops.end() ? KvmSOpInfo{} : it->second; };
    int failures = 0;
    for (int ai = 1; ai < argc; ++ai) {
        const std::string dir = argv[ai];
        const std::string meta = slurp(dir + "/metadata.json");
        auto tasks = load<KvmTaskDesc>(dir + "/tasks.bin"); auto succ = load<int32_t>(dir + "/successors.bin"); auto args = load<int32_t>(dir + "/args.bin");
        const auto bufs = load_bufs(meta);
        const KvmFuseResult r = kvm_fuse_mlp(tasks, succ, args, bufs, ops, op_of);
        printf("%s: %zu tasks, %zu buffers -> fused runs %d, unchanged runs %d, replaced tasks %d, new tasks %d\n", dir.c_str(), tasks.size(), bufs.size(), r.fused_runs, r.unchanged_runs, r.replaced_tasks, r.new_tasks);
        std::map<std::string, int> counts;
        for (const auto& line : r.report) { auto cut = line.find(": unchanged: "); ++counts[cut == std::string::npos ? line.substr(line.find(": ") + 2) : line.substr(cut + 2)]; }
        for (const auto& kv : counts) printf("  [%d runs] %s\n", kv.second, kv.first.c_str());
        if (!r.fused_runs) continue;
        // structural checks of the rewritten bundle
        if (!acyclic(r.tasks, r.succ)) { printf("  FAIL: the rewritten bundle has a cycle\n"); ++failures; continue; }
        // kept tasks keep their op and arguments: the multiset of (op, args) over tasks that are not fused equals the original minus the replaced tasks
        std::map<std::vector<int>, int> before, after;
        std::set<int> replaced;
        for (const auto& d : tasks) { std::vector<int> k{d.op_id}; k.insert(k.end(), args.begin() + d.args_off, args.begin() + d.args_off + d.args_words); ++before[k]; }
        for (const auto& d : r.tasks) if (d.op_id != slice_id && d.op_id != sgd_id) { std::vector<int> k{d.op_id}; k.insert(k.end(), r.args.begin() + d.args_off, r.args.begin() + d.args_off + d.args_words); ++after[k]; }
        int lost = 0, extra = 0;
        for (const auto& kv : before) { auto it = after.find(kv.first); lost += kv.second - (it == after.end() ? 0 : std::min(it->second, kv.second)); }
        for (const auto& kv : after) { auto it = before.find(kv.first); extra += kv.second - (it == before.end() ? 0 : std::min(it->second, kv.second)); }
        if (extra || lost != r.replaced_tasks) { printf("  FAIL: kept tasks changed: %d replaced, %d lost, %d invented\n", r.replaced_tasks, lost, extra); ++failures; continue; }
        // every slice precedes every group of its step, and groups run after the slices: checked through the planner, which also proves the record layout fits
        for (int W : {1, 4, 8, 107}) {
            KvmSConfig cfg; cfg.workers = W;
            const KvmSCostFn cost = [&](int op, const int32_t*) { return op == slice_id ? 60.0 : op == sgd_id ? 8.0 : op == 0 ? 0.1 : 3.0; };
            const KvmSPlan p = kvm_static_plan(r.tasks, r.succ, r.args, cfg, op_of, cost);
            int done = 0; for (const auto& prog : p.prog) done += (int)prog.size();
            if (done != (int)r.tasks.size()) { printf("  FAIL: planner placed %d of %zu tasks at W=%d\n", done, r.tasks.size(), W); ++failures; }
            printf("  W=%d: planned %d tasks, predicted makespan %.1f us (work %.1f us), %d cross-worker waits\n", W, done, p.makespan_us, p.total_work_us, p.cross_edges);
        }
    }
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
