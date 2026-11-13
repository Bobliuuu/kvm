// CPU check of the bundle-level fusion pass (include/kvm_fuse.h) on real bundles: prints the decision for every run and, when it fires, checks the rewritten bundle.
//   g++ -std=c++17 -O2 -Iinclude tests/static/test_fuse_pass.cpp -o /tmp/test_fuse_pass && /tmp/test_fuse_pass BUNDLE_DIR[@FORM]...   (build from the repo root)
// FORM, when given, is what the pass must make of every run of the bundle at the largest worker count (108, an A100's default): none (no run fused), block (the one-block
// form, fmlp_slice and fmlp_sgd), split (the split form's fmlp_split gangs) or grid (the grid form's fmlp_grid gangs). A bundle given without one is only reported.
// Checks on a fired bundle: the DAG is acyclic; every kept task keeps its op and arguments; every replaced step became its slices and groups, each slice before every group; the static
// planner accepts the result on the bundle's registry signatures, with the fused ops' signatures added; every gang's members sit on distinct workers; and the emitted records
// replay as the device runs them (tests/static/replay.h): no deadlock through any gang's barrier, every edge of the DAG honoured.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <vector>
#include "kvm_fuse.h"
#include "registry.h"
#include "replay.h"

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
template <class T> static std::vector<T> load(const std::string& p) { std::string r = slurp(p); assert(r.size() % sizeof(T) == 0); std::vector<T> v(r.size() / sizeof(T)); memcpy(v.data(), r.data(), r.size()); return v; }

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
    // --smem FLOATS: the shared-memory budget of a fused block (default: the pass's, an A100's); a smaller GPU's budget changes the forms
    int smem_floats = KvmFuseOptions{}.smem_floats, first_arg = 1;
    if (argc > 2 && !strcmp(argv[1], "--smem")) { smem_floats = atoi(argv[2]); first_arg = 3; }
    if (argc <= first_arg || smem_floats <= 0) { fprintf(stderr, "usage: %s [--smem FLOATS] BUNDLE_DIR[@none|block|split|grid]...\n", argv[0]); return 2; }
    const kvm_test_registry::Registry& R = kvm_test_registry::registry();
    KvmFuseOps ops;
    ops.noop = R.require("noop");
    ops.select_window = R.require("select_window");
    ops.embedding_fwd = R.require("embedding_fwd");
    ops.matmul_tile = R.require("matmul_tile");
    ops.softmax_ce_fused = R.require("softmax_ce_fused");
    ops.sgd_range = R.require("sgd_range");
    ops.softmax_ce_fwd = R.require("softmax_ce_fwd");
    ops.softmax_ce_reduce = R.require("softmax_ce_reduce");
    ops.softmax_ce_bwd = R.require("softmax_ce_bwd");
    ops.gelu_bwd = R.require("gelu_bwd");
    ops.colsum = R.require("colsum");
    ops.embedding_bwd = R.require("embedding_bwd");
    ops.sgd = R.require("sgd");
    ops.fmlp_slice = R.require("fmlp_slice");
    ops.fmlp_sgd = R.require("fmlp_sgd");
    ops.fmlp_split = R.require("fmlp_split");
    ops.fmlp_grid = R.require("fmlp_grid");
    const int slice_id = ops.fmlp_slice, sgd_id = ops.fmlp_sgd, split_id = ops.fmlp_split, grid_id = ops.fmlp_grid;
    const auto gang_op = [&](int op) { return op == split_id || op == grid_id; };
    const KvmSOpLookup op_of = R.lookup();
    static constexpr int kWorkers[] = {1, 4, 8, 48, 107, 108};   // 108: the A100's grid, the executor's default worker count
    int failures = 0;
    for (int ai = first_arg; ai < argc; ++ai) {
        // BUNDLE_DIR[@FORM]: the form follows the last @, unless a / follows it (a directory name with an @ in it)
        const std::string spec = argv[ai]; const size_t at = spec.rfind('@');
        const bool has_form = at != std::string::npos && spec.find('/', at) == std::string::npos;
        const std::string dir = has_form ? spec.substr(0, at) : spec, expect = has_form ? spec.substr(at + 1) : "";
        if (has_form && expect != "none" && expect != "block" && expect != "split" && expect != "grid") { fprintf(stderr, "%s: unknown form \"%s\" (none, block, split or grid)\n", spec.c_str(), expect.c_str()); return 2; }
        const std::string meta = slurp(dir + "/metadata.json");
        auto tasks = load<KvmTaskDesc>(dir + "/tasks.bin"); auto succ = load<int32_t>(dir + "/successors.bin"); auto args = load<int32_t>(dir + "/args.bin");
        const auto bufs = load_bufs(meta);
        // the pass sizes its runs for the workers the plan will have (a split step's gang must fit them), so each worker count runs the pass afresh
        for (int W : kWorkers) {
        KvmFuseOptions opts; opts.workers = W; opts.smem_floats = smem_floats;
        const KvmFuseResult r = kvm_fuse_mlp(tasks, succ, args, bufs, ops, op_of, opts);
        printf("%s W=%d: %zu tasks, %zu buffers -> fused runs %d, unchanged runs %d, replaced tasks %d, new tasks %d\n", dir.c_str(), W, tasks.size(), bufs.size(), r.fused_runs, r.unchanged_runs, r.replaced_tasks, r.new_tasks);
        std::map<std::string, int> counts;
        for (const auto& line : r.report) { auto cut = line.find(": unchanged: "); ++counts[cut == std::string::npos ? line.substr(line.find(": ") + 2) : line.substr(cut + 2)]; }
        for (const auto& kv : counts) printf("  [%d runs] %s\n", kv.second, kv.first.c_str());
        if (!r.dropped.empty()) printf("  nothing fused: %s\n", r.dropped.c_str());
        if (!expect.empty() && W == kWorkers[sizeof kWorkers / sizeof kWorkers[0] - 1]) {
            int block = 0, split = 0, grid = 0;
            for (const auto& d : r.tasks) { block += d.op_id == slice_id || d.op_id == sgd_id; split += d.op_id == split_id; grid += d.op_id == grid_id; }
            const char* got = !r.fused_runs ? "none" : r.unchanged_runs || (block > 0) + (split > 0) + (grid > 0) > 1 ? "mixed" : block ? "block" : split ? "split" : "grid";
            if (expect != got) { printf("  FAIL: expected %s at W=%d, got %s (%d of %d runs fused)\n", expect.c_str(), W, got, r.fused_runs, r.fused_runs + r.unchanged_runs); ++failures; }
            else printf("  expected %s at W=%d: ok\n", expect.c_str(), W);
            // the static kernel entry the executor launches (kvm_fuse_entry): each fused form on its own entry, an unfused bundle on the generic one
            const KvmFuseEntry::Kind want = expect == "split" ? KvmFuseEntry::split : expect == "grid" ? KvmFuseEntry::grid : expect == "block" ? KvmFuseEntry::block : KvmFuseEntry::generic;
            if (r.entry.kind != want) { printf("  FAIL: entry at W=%d is %s\n", W, kvm_fuse_entry_name(r.entry).c_str()); ++failures; }
            else printf("  entry at W=%d: %s: ok\n", W, kvm_fuse_entry_name(r.entry).c_str());
        }
        if (!r.fused_runs) continue;
        // a fused bundle runs on one entry: never `mixed` (the pass leaves such a bundle unfused). A split entry is instantiated for its runs'
        // virtual share count, and each gang carries it on its own run's physical shares (kvm_fused_split.h): a member's `members` argument is
        // Q x its run's S, with 1 <= S <= Sv = the entry's count (kvm_s_fmlp_split refuses anything else)
        if (r.entry.kind == KvmFuseEntry::mixed) { printf("  FAIL: W=%d: a fused bundle with no entry (%s)\n", W, r.entry.why.c_str()); ++failures; }
        if (r.entry.kind == KvmFuseEntry::split) {
            int bad = 0;
            std::set<int> phys;
            for (const auto& d : r.tasks) {
                if (d.op_id != split_id) continue;
                const int run = r.args[d.args_off], members = r.args[d.args_off + 3];
                const bool ok = run >= 0 && run < (int)r.runs.size() && r.runs[run].Sv == r.entry.S && r.runs[run].S >= 1 && r.runs[run].S <= r.runs[run].Sv &&
                                members == KVM_SPLIT_Q * r.runs[run].S;
                bad += !ok;
                if (ok) phys.insert(r.runs[run].S);
            }
            if (bad) { printf("  FAIL: W=%d: %d split member(s) whose gang is not %d x its run's physical shares, up to the entry's %d virtual\n", W, bad, KVM_SPLIT_Q, r.entry.S); ++failures; }
            else { std::string ps; for (int x : phys) ps += (ps.empty() ? "" : ", ") + std::to_string(x); printf("  split gangs at W=%d: %d virtual shares on %s physical: ok\n", W, r.entry.S, ps.c_str()); }
        }
        // structural checks of the rewritten bundle
        if (!acyclic(r.tasks, r.succ)) { printf("  FAIL: the rewritten bundle has a cycle\n"); ++failures; continue; }
        // kept tasks keep their op and arguments: the multiset of (op, args) over tasks that are not fused equals the original minus the replaced tasks
        std::map<std::vector<int>, int> before, after;
        for (const auto& d : tasks) { std::vector<int> k{d.op_id}; k.insert(k.end(), args.begin() + d.args_off, args.begin() + d.args_off + d.args_words); ++before[k]; }
        for (const auto& d : r.tasks) if (d.op_id != slice_id && d.op_id != sgd_id && !gang_op(d.op_id)) { std::vector<int> k{d.op_id}; k.insert(k.end(), r.args.begin() + d.args_off, r.args.begin() + d.args_off + d.args_words); ++after[k]; }
        int lost = 0, extra = 0;
        for (const auto& kv : before) { auto it = after.find(kv.first); lost += kv.second - (it == after.end() ? 0 : std::min(it->second, kv.second)); }
        for (const auto& kv : after) { auto it = before.find(kv.first); extra += kv.second - (it == before.end() ? 0 : std::min(it->second, kv.second)); }
        if (extra || lost != r.replaced_tasks) { printf("  FAIL: kept tasks changed: %d replaced, %d lost, %d invented\n", r.replaced_tasks, lost, extra); ++failures; continue; }
        // the planner places every task, and every gang's members on distinct workers
        KvmSConfig cfg; cfg.workers = W;
        const KvmSCostFn cost = [&](int op, const int32_t*) { return op == slice_id ? 60.0 : op == sgd_id ? 8.0 : op == split_id ? 40.0 : op == grid_id ? 60.0 : op == 0 ? 0.1 : 3.0; };
        KvmSPlan p;
        try { p = kvm_static_plan(r.tasks, r.succ, r.args, cfg, op_of, cost); }
        catch (const std::exception& e) { printf("  FAIL: the planner refused the rewritten bundle at W=%d: %s\n", W, e.what()); ++failures; continue; }
        int done = 0; for (const auto& prog : p.prog) done += (int)prog.size();
        if (done != (int)r.tasks.size()) { printf("  FAIL: planner placed %d of %zu tasks at W=%d\n", done, r.tasks.size(), W); ++failures; }
        std::map<int, std::set<int>> gang_workers; std::map<int, int> gang_size;
        for (size_t t = 0; t < r.tasks.size(); ++t) if (gang_op(r.tasks[t].op_id)) { const int g = r.args[r.tasks[t].args_off + 1]; gang_workers[g].insert(p.worker_of[t]); ++gang_size[g]; }
        for (const auto& kv : gang_workers) if ((int)kv.second.size() != gang_size[kv.first]) { printf("  FAIL: gang %d shares a worker\n", kv.first); ++failures; break; }
        // the emitted records replay as the device runs them: no deadlock through any gang's barrier, every edge (successors and versions) honoured
        const kvm_test_replay::Result rp = kvm_test_replay::replay(r.tasks, r.args, kvm_static::build_dag(r.tasks, r.succ), p, op_of);
        if (!rp.ok) { printf("  FAIL: the records' replay at W=%d: %s\n", W, rp.why.c_str()); ++failures; }
        printf("  W=%d: planned %d tasks (%d gangs), predicted makespan %.1f us (work %.1f us), %d cross-worker waits; replayed %zu records%s\n", W, done, p.gangs,
               p.makespan_us, p.total_work_us, p.cross_edges, rp.records, rp.ok ? (rp.gangs ? ", every gang's barrier crossed" : "") : ", FAILED");
        }
    }
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
