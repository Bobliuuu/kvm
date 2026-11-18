// CPU check of the bundle-level fusion pass (include/mvm_fuse.h) on real bundles: prints the decision for every run and, when it fires, checks the rewritten bundle.
//   g++ -std=c++17 -O2 -Iinclude tests/static/test_fuse_pass.cpp -o /tmp/test_fuse_pass && /tmp/test_fuse_pass BUNDLE_DIR[@FORM]...   (build from the repo root)
// FORM, when given, is what the pass must make of every run of the bundle at the largest worker count (108, an A100's default): none (no run fused), block (the one-block
// form, fmlp_slice and fmlp_sgd), split (the split form's fmlp_split gangs, its members holding their weights), split-streamed (the same gangs, their
// members streaming their weights: mvm_fused_split.h), grid (the grid form's fmlp_grid gangs) or batch (the batched form's fmlp_batch gangs:
// every run of the bundle in one gang a step). A bundle given without one is only reported.
// Checks on a fired bundle: the DAG is acyclic; every kept task keeps its op and arguments; every replaced step became its slices and groups, each slice before every group; the static
// planner accepts the result on the bundle's registry signatures, with the fused ops' signatures added; every gang's members sit on distinct workers; and the emitted records
// replay as the device runs them (tests/static/replay.h): no deadlock through any gang's barrier, every edge of the DAG honoured.
#include <algorithm>
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
#include "mvm_fuse.h"
#include "registry.h"
#include "replay.h"

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
template <class T> static std::vector<T> load(const std::string& p) { std::string r = slurp(p); assert(r.size() % sizeof(T) == 0); std::vector<T> v(r.size() / sizeof(T)); memcpy(v.data(), r.data(), r.size()); return v; }

// Pulls the buffer table out of metadata.json without a JSON library: every {"name":..., "index":N, "rows":R, "cols":C, "dtype":D ...} entry of the "buffers" array.
static std::vector<MvmFuseBuf> load_bufs(const std::string& meta) {
    const size_t start = meta.find("\"buffers\"");
    assert(start != std::string::npos);
    const std::regex re("\"index\"\\s*:\\s*(\\d+)\\s*,\\s*\"rows\"\\s*:\\s*(\\d+)\\s*,\\s*\"cols\"\\s*:\\s*(\\d+)\\s*,\\s*\"dtype\"\\s*:\\s*(\\d+)");
    std::vector<MvmFuseBuf> v;
    for (std::sregex_iterator it(meta.begin() + start, meta.end(), re), end; it != end; ++it) {
        const int idx = std::stoi((*it)[1].str());
        if (idx >= (int)v.size()) v.resize(idx + 1);
        v[idx] = MvmFuseBuf{std::stoi((*it)[2].str()), std::stoi((*it)[3].str()), std::stoi((*it)[3].str()), std::stoi((*it)[4].str())};
    }
    return v;
}

// FNV-1a over the pass's whole result: every task descriptor, successor and argument word of the rewritten bundle, every fused run's
// descriptor (its shapes, buffers, slices, form and gang) and the interior buffers, and the entry the bundle runs on. Two bundles with one digest run the same tasks on the same runs.
static uint64_t digest(const MvmFuseResult& r) {
    uint64_t h = 1469598103934665603ull;
    const auto eat = [&](const void* p, size_t n) { for (size_t i = 0; i < n; ++i) { h ^= static_cast<const unsigned char*>(p)[i]; h *= 1099511628211ull; } };
    const auto num = [&](int32_t v) { eat(&v, 4); };
    eat(r.tasks.data(), r.tasks.size() * sizeof(MvmTaskDesc)); eat(r.succ.data(), r.succ.size() * 4); eat(r.args.data(), r.args.size() * 4);
    for (const MvmFuseRun& u : r.runs) {
        for (int v : {u.L, u.E, u.V, u.T, u.slice_rows, u.nsl, u.groups, u.smem_floats, u.wte, u.head, u.steps, u.split, u.S, u.Q, u.Sv, u.grid, u.P}) num(v);
        eat(&u.lr, 4);
        for (int l = 0; l < MVM_FUSE_MAXL; ++l) { num(u.H[l]); num(u.w[l]); num(u.b[l]); }
    }
    for (int b : r.interior) num(b);
    num((int32_t)r.entry.kind); num(r.entry.S);
    return h;
}

static bool acyclic(const std::vector<MvmTaskDesc>& t, const std::vector<int32_t>& succ) {
    std::vector<int> in(t.size(), 0), st; int seen = 0;
    for (const auto& d : t) for (int k = 0; k < d.succ_cnt; ++k) ++in[succ[d.succ_off + k]];
    for (size_t i = 0; i < t.size(); ++i) if (!in[i]) st.push_back((int)i);
    while (!st.empty()) { int u = st.back(); st.pop_back(); ++seen; for (int k = 0; k < t[u].succ_cnt; ++k) if (--in[succ[t[u].succ_off + k]] == 0) st.push_back(succ[t[u].succ_off + k]); }
    return seen == (int)t.size();
}

// The loss lowered again in row blocks of `rows` (a tiling's Tiling.ce_rows), from a bundle whose loss is one softmax_ce_fused task over
// rows [0, T) followed by its reduce. Form 0: softmax_ce_fused blocks, then the reduce. Form 1: softmax_ce_fwd blocks, the reduce,
// softmax_ce_bwd blocks. Every block takes the replaced task's place and edges: its predecessors, and its successors (in form 1 the fwd
// blocks feed the reduce, and the bwd blocks follow it and feed the fused task's other successors). `breakage` damages block 1 of every
// loss: 1 a gap before it, 2 an overlap with block 0, 3 a cover that ends a row short (on the last block), 4 another vocabulary.
// Returns false (and leaves `out` alone) when the bundle has no such loss.
struct Bundle { std::vector<MvmTaskDesc> tasks; std::vector<int32_t> succ, args; };
static bool relower_loss(const Bundle& in, const MvmFuseOps& ops, int form, int breakage, int rows, Bundle& out) {
    const int n = (int)in.tasks.size();
    std::vector<std::vector<int>> pred(n), next(n);
    for (int i = 0; i < n; ++i) for (int k = 0; k < in.tasks[i].succ_cnt; ++k) { const int j = in.succ[in.tasks[i].succ_off + k]; next[i].push_back(j); pred[j].push_back(i); }
    const auto argv_of = [&](int t) { return std::vector<int32_t>(in.args.begin() + in.tasks[t].args_off, in.args.begin() + in.tasks[t].args_off + in.tasks[t].args_words); };
    // The new tasks, in order: (old task it comes from, op, args); edges are added below on new indices.
    struct New { int from; int op; std::vector<int32_t> args; };
    std::vector<New> nt; std::vector<std::vector<int>> first_last(n);   // old task -> its new tasks
    std::vector<int> reduce_of(n, -1);                                   // a fused CE -> its reduce
    for (int t = 0; t < n; ++t) {
        if (in.tasks[t].op_id != ops.softmax_ce_fused) continue;
        const auto f = argv_of(t);
        for (int j : next[t]) if (in.tasks[j].op_id == ops.softmax_ce_reduce) reduce_of[t] = j;
        if (f.size() != 11 || f[5] != 0 || reduce_of[t] < 0 || in.tasks[t].wait_var != MVM_NO_VAR || in.tasks[t].pub_var != MVM_NO_VAR) return false;
    }
    if (std::count_if(reduce_of.begin(), reduce_of.end(), [](int r) { return r >= 0; }) == 0) return false;
    std::vector<int> bwd_after(n, -1);   // a reduce -> the fused CE whose bwd blocks follow it (form 1)
    for (int t = 0; t < n; ++t) if (reduce_of[t] >= 0) bwd_after[reduce_of[t]] = t;
    const auto blocks = [&](int T) {
        std::vector<std::pair<int, int>> b;
        for (int r0 = 0; r0 < T; r0 += rows) b.push_back({r0, std::min(T, r0 + rows)});
        if (b.size() < 2) return b;
        if (breakage == 1) b[1].first += 1;                          // a gap
        if (breakage == 2) b[1].first -= 1;                          // an overlap
        if (breakage == 3) b.back().second -= 1;                     // a short cover
        return b;
    };
    for (int t = 0; t < n; ++t) {
        if (reduce_of[t] >= 0) {   // a fused CE: its blocks
            const auto f = argv_of(t);
            const auto bl = blocks(f[6]);
            for (size_t i = 0; i < bl.size(); ++i) {
                std::vector<int32_t> a = form == 0 ? f : std::vector<int32_t>{f[0], f[1], f[2], f[3], 0, 0, f[7], f[8]};
                const int ri = form == 0 ? 5 : 4;
                a[ri] = bl[i].first; a[ri + 1] = bl[i].second;
                if (breakage == 4 && i == 1) a[ri + 2] -= 4;          // another vocabulary
                first_last[t].push_back((int)nt.size()); nt.push_back({t, form == 0 ? ops.softmax_ce_fused : ops.softmax_ce_fwd, a});
            }
            continue;
        }
        first_last[t].push_back((int)nt.size()); nt.push_back({t, in.tasks[t].op_id, argv_of(t)});
        if (form == 1 && bwd_after[t] >= 0) {   // the reduce of a fused CE: the bwd blocks follow it
            const int c = bwd_after[t]; const auto f = argv_of(c); const auto r = argv_of(t);
            for (const auto& b : blocks(f[6])) {
                std::vector<int32_t> a{f[0], f[1], f[3], r[2], f[4], b.first, b.second, f[7], f[8], f[10]};
                first_last[c].push_back((int)nt.size()); nt.push_back({c, ops.softmax_ce_bwd, a});
            }
        }
    }
    // Edges on new indices.
    std::vector<std::set<int>> e(nt.size());
    const auto is_bwd = [&](int k) { return form == 1 && nt[k].op == ops.softmax_ce_bwd; };
    for (int t = 0; t < n; ++t) for (int j : next[t]) for (int a : first_last[t]) for (int b : first_last[j]) {
        if (form == 1 && reduce_of[t] >= 0) {   // fwd blocks -> the reduce only; bwd blocks -> the fused CE's other successors
            if (j == reduce_of[t]) { if (!is_bwd(a)) e[a].insert(b); }
            else if (is_bwd(a)) e[a].insert(b);
            continue;
        }
        if (form == 1 && reduce_of[j] >= 0) { e[a].insert(b); continue; }   // the CE's predecessors feed its fwd and bwd blocks
        e[a].insert(b);
    }
    if (form == 1) for (int t = 0; t < n; ++t) if (reduce_of[t] >= 0) for (int b : first_last[t]) if (is_bwd(b)) e[first_last[reduce_of[t]][0]].insert(b);   // reduce -> bwd blocks
    out = Bundle{};
    std::vector<int> indeg(nt.size(), 0);
    for (size_t k = 0; k < nt.size(); ++k) for (int j : e[k]) ++indeg[j];
    for (size_t k = 0; k < nt.size(); ++k) {
        MvmTaskDesc d = in.tasks[nt[k].from];
        d.op_id = nt[k].op; d.args_off = (int32_t)out.args.size(); d.args_words = (int32_t)nt[k].args.size();
        d.succ_off = (int32_t)out.succ.size(); d.succ_cnt = (int32_t)e[k].size(); d.indegree = indeg[k];
        out.args.insert(out.args.end(), nt[k].args.begin(), nt[k].args.end());
        out.succ.insert(out.succ.end(), e[k].begin(), e[k].end());
        out.tasks.push_back(d);
    }
    return true;
}

int main(int argc, char** argv) {
    // --smem FLOATS: the shared-memory budget of a fused block (default: the pass's, an A100's); a smaller GPU's budget changes the forms.
    // --virtual-over-batch N: the most grid runs the virtual form takes ahead of the batched form (default: the pass's; 0: the batched form first).
    int smem_floats = MvmFuseOptions{}.smem_floats, virtual_over_batch = MvmFuseOptions{}.grid_virtual_over_batch, first_arg = 1;
    for (; first_arg + 1 < argc; first_arg += 2) {
        if (!strcmp(argv[first_arg], "--smem")) smem_floats = atoi(argv[first_arg + 1]);
        else if (!strcmp(argv[first_arg], "--virtual-over-batch")) virtual_over_batch = atoi(argv[first_arg + 1]);
        else break;
    }
    if (argc <= first_arg || smem_floats <= 0 || virtual_over_batch < 0) {
        fprintf(stderr, "usage: %s [--smem FLOATS] [--virtual-over-batch N] BUNDLE_DIR[@none|block|split|split-streamed|grid|grid-virtual|batch]...\n", argv[0]); return 2; }
    printf("options: smem %d floats, virtual form ahead of the batched form up to %d grid runs\n", smem_floats, virtual_over_batch);
    const mvm_test_registry::Registry& R = mvm_test_registry::registry();
    MvmFuseOps ops;
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
    ops.fmlp_batch = R.require("fmlp_batch");
    const int slice_id = ops.fmlp_slice, sgd_id = ops.fmlp_sgd, split_id = ops.fmlp_split, grid_id = ops.fmlp_grid, batch_id = ops.fmlp_batch;
    const auto gang_op = [&](int op) { return op == split_id || op == grid_id || op == batch_id; };
    const MvmSOpLookup op_of = R.lookup();
    static constexpr int kWorkers[] = {1, 4, 8, 48, 107, 108};   // 108: the A100's grid, the executor's default worker count
    int failures = 0;
    for (int ai = first_arg; ai < argc; ++ai) {
        // BUNDLE_DIR[@FORM]: the form follows the last @, unless a / follows it (a directory name with an @ in it)
        const std::string spec = argv[ai]; const size_t at = spec.rfind('@');
        const bool has_form = at != std::string::npos && spec.find('/', at) == std::string::npos;
        const std::string dir = has_form ? spec.substr(0, at) : spec, expect = has_form ? spec.substr(at + 1) : "";
        if (has_form && expect != "none" && expect != "block" && expect != "split" && expect != "split-streamed" && expect != "grid" && expect != "grid-virtual" && expect != "batch") { fprintf(stderr, "%s: unknown form \"%s\" (none, block, split, split-streamed, grid or batch)\n", spec.c_str(), expect.c_str()); return 2; }
        const std::string meta = slurp(dir + "/metadata.json");
        auto tasks = load<MvmTaskDesc>(dir + "/tasks.bin"); auto succ = load<int32_t>(dir + "/successors.bin"); auto args = load<int32_t>(dir + "/args.bin");
        const auto bufs = load_bufs(meta);
        // the pass sizes its runs for the workers the plan will have (a split step's gang must fit them), so each worker count runs the pass afresh
        for (int W : kWorkers) {
        MvmFuseOptions opts; opts.workers = W; opts.smem_floats = smem_floats; opts.grid_virtual_over_batch = virtual_over_batch;
        const MvmFuseResult r = mvm_fuse_mlp(tasks, succ, args, bufs, ops, op_of, opts);
        printf("%s W=%d: %zu tasks, %zu buffers -> fused runs %d, unchanged runs %d, replaced tasks %d, new tasks %d; rewritten bundle %016llx\n", dir.c_str(), W, tasks.size(), bufs.size(), r.fused_runs, r.unchanged_runs, r.replaced_tasks, r.new_tasks, (unsigned long long)digest(r));
        for (const auto& line : r.warnings) printf("  warning: %s\n", line.c_str());
        std::map<std::string, int> counts;
        for (const auto& line : r.report) { auto cut = line.find(": unchanged: "); ++counts[cut == std::string::npos ? line.substr(line.find(": ") + 2) : line.substr(cut + 2)]; }
        for (const auto& kv : counts) printf("  [%d runs] %s\n", kv.second, kv.first.c_str());
        if (!r.dropped.empty()) printf("  nothing fused: %s\n", r.dropped.c_str());
        if (!expect.empty() && W == kWorkers[sizeof kWorkers / sizeof kWorkers[0] - 1]) {
            int block = 0, split = 0, grid = 0, batch = 0;
            for (const auto& d : r.tasks) { block += d.op_id == slice_id || d.op_id == sgd_id; split += d.op_id == split_id; grid += d.op_id == grid_id; batch += d.op_id == batch_id; }
            const char* got = !r.fused_runs ? "none" : r.unchanged_runs || (block > 0) + (split > 0) + (grid > 0) + (batch > 0) > 1 ? "mixed" : block ? "block" : split ? (r.entry.stream ? "split-streamed" : "split") : grid ? (r.entry.kind == MvmFuseEntry::grid_virtual ? "grid-virtual" : "grid") : "batch";
            if (expect != got) { printf("  FAIL: expected %s at W=%d, got %s (%d of %d runs fused)\n", expect.c_str(), W, got, r.fused_runs, r.fused_runs + r.unchanged_runs); ++failures; }
            else printf("  expected %s at W=%d: ok\n", expect.c_str(), W);
            // the static kernel entry the executor launches (mvm_fuse_entry): each fused form on its own entry, an unfused bundle on the generic one
            const MvmFuseEntry::Kind want = expect == "split" || expect == "split-streamed" ? MvmFuseEntry::split : expect == "grid" ? MvmFuseEntry::grid : expect == "grid-virtual" ? MvmFuseEntry::grid_virtual : expect == "block" ? MvmFuseEntry::block : expect == "batch" ? MvmFuseEntry::batch : MvmFuseEntry::generic;
            // "batch" is either batched entry: the split arithmetic's or the grid arithmetic's (the batched check below reads each run's)
            if (r.entry.kind != want && !(want == MvmFuseEntry::batch && r.entry.kind == MvmFuseEntry::batch_grid)) { printf("  FAIL: entry at W=%d is %s\n", W, mvm_fuse_entry_name(r.entry).c_str()); ++failures; }
            else printf("  entry at W=%d: %s: ok\n", W, mvm_fuse_entry_name(r.entry).c_str());
        }
        if (!r.fused_runs) continue;
        // a fused bundle runs on one entry: never `mixed` (the pass leaves such a bundle unfused). A split entry is instantiated for its runs'
        // virtual share count, and each gang carries it on its own run's physical shares (mvm_fused_split.h): a member's `members` argument is
        // Q x its run's S, with 1 <= S <= Sv = the entry's count (mvm_s_fmlp_split refuses anything else)
        if (r.entry.kind == MvmFuseEntry::mixed) { printf("  FAIL: W=%d: a fused bundle with no entry (%s)\n", W, r.entry.why.c_str()); ++failures; }
        if (r.entry.kind == MvmFuseEntry::split) {
            int bad = 0;
            std::set<int> phys;
            for (const auto& d : r.tasks) {
                if (d.op_id != split_id) continue;
                const int run = r.args[d.args_off], members = r.args[d.args_off + 3];
                bool ok = run >= 0 && run < (int)r.runs.size() && r.runs[run].Sv == r.entry.S && r.runs[run].S >= 1 && r.runs[run].S <= r.runs[run].Sv &&
                          members == MVM_SPLIT_Q * r.runs[run].S && (r.runs[run].kc > 0) == r.entry.stream;
                if (ok) {   // the descriptor the executor lays out for the run (mvm_exec.cu) fits the budget, as the member its entry runs reads it
                    const MvmFuseRun& u = r.runs[run];
                    MvmSplitDesc sd{}; sd.V = u.V; sd.E = u.E; sd.L = u.L; sd.H = u.H[0]; sd.S = u.S; sd.Sv = u.Sv; sd.Q = u.Q;
                    const int f = mvm_split_layout(sd, u.kc);
                    ok = f > 0 && f <= u.smem_floats && mvm_split_is_streamed(sd) == (u.kc > 0) && (!u.kc || mvm_split_stream_built(u.Sv));
                }
                bad += !ok;
                if (ok) phys.insert(r.runs[run].S);
            }
            if (bad) { printf("  FAIL: W=%d: %d split member(s) whose gang is not %d x its run's physical shares, up to the entry's %d virtual, with the entry's member and a layout that fits\n", W, bad, MVM_SPLIT_Q, r.entry.S); ++failures; }
            else { std::string ps; for (int x : phys) ps += (ps.empty() ? "" : ", ") + std::to_string(x); printf("  split gangs at W=%d: %d virtual shares on %s physical, weights %s: ok\n", W, r.entry.S, ps.c_str(), r.entry.stream ? "streamed" : "resident"); }
        }
        if (r.entry.kind == MvmFuseEntry::grid || r.entry.kind == MvmFuseEntry::grid_virtual) {
            // a grid member's `members` argument is its run's blocks: P resident, phys (1 <= phys < P) on virtual members, every run alike, and the
            // descriptor the executor lays out for it fits the budget (mvm_s_fmlp_grid and mvm_s_fmlp_grid_virtual refuse anything else)
            const bool virt = r.entry.kind == MvmFuseEntry::grid_virtual;
            int bad = 0; std::set<int> blocks;
            for (const auto& d : r.tasks) {
                if (d.op_id != grid_id) continue;
                const int run = r.args[d.args_off], members = r.args[d.args_off + 3];
                bool ok = run >= 0 && run < (int)r.runs.size();
                if (ok) {
                    const MvmFuseRun& u = r.runs[run];
                    MvmGridDesc gd{}; gd.V = u.V; gd.E = u.E; gd.L = u.L; gd.H = u.H[0]; gd.T = u.T; gd.P = u.P; gd.phys = u.phys;
                    const int f = virt ? mvm_grid_layout_chunked(gd, u.smem_floats) : mvm_grid_layout(gd);
                    ok = (virt ? u.phys >= 1 && u.phys < u.P && members == u.phys : u.phys == 0 && members == u.P) && f > 0 && f <= u.smem_floats;
                    if (ok) blocks.insert(members);
                }
                bad += !ok;
            }
            if (bad) { printf("  FAIL: W=%d: %d grid member(s) whose block count is not their run's %s, or whose layout does not fit\n", W, bad, virt ? "phys (1 to P - 1)" : "P"); ++failures; }
            else { std::string bs; for (int x : blocks) bs += (bs.empty() ? "" : ", ") + std::to_string(x); printf("  grid gangs at W=%d: %s blocks a run%s: ok\n", W, bs.c_str(), virt ? " (virtual members)" : ""); }
        }
        if (W == kWorkers[sizeof kWorkers / sizeof kWorkers[0] - 1]) {   // the loss in row blocks (a tiling's Tiling.ce_rows), at the A100's workers:
            // either form fuses to this very bundle; a broken covering stays unfused, with a warning
            static const char* kBreak[] = {"", "a gap", "an overlap", "a short cover", "another vocabulary"};
            const Bundle orig{tasks, succ, args};
            Bundle rb;
            for (int form = 0; form < 2; ++form) {
                if (!relower_loss(orig, ops, form, 0, 16, rb)) { printf("  loss in row blocks: no single-task loss to lower again\n"); break; }
                const MvmFuseResult q = mvm_fuse_mlp(rb.tasks, rb.succ, rb.args, bufs, ops, op_of, opts);
                const bool ok = q.fused_runs == r.fused_runs && q.unchanged_runs == r.unchanged_runs && digest(q) == digest(r) && q.warnings.empty();
                printf("  loss in 16-row blocks, %s form: %zu tasks -> fused runs %d, rewritten bundle %016llx: %s\n", form ? "per-op" : "fused", rb.tasks.size(),
                       q.fused_runs, (unsigned long long)digest(q), ok ? "the same bundle" : "FAIL: not the same rewritten bundle");
                failures += !ok;
            }
            for (int brk = 1; brk <= 4 && relower_loss(orig, ops, 0, brk, 16, rb); ++brk) {
                const MvmFuseResult q = mvm_fuse_mlp(rb.tasks, rb.succ, rb.args, bufs, ops, op_of, opts);
                const bool ok = q.fused_runs == 0 && (int)q.warnings.size() == r.fused_runs;
                printf("  loss in 16-row blocks with %s: fused runs %d, %zu warning(s)%s: %s\n", kBreak[brk], q.fused_runs, q.warnings.size(),
                       q.warnings.empty() ? "" : (" (" + q.warnings[0].substr(q.warnings[0].find('(') + 1, q.warnings[0].rfind(')') - q.warnings[0].find('(') - 1) + ")").c_str(),
                       ok ? "refused, with a warning" : "FAIL: expected every run refused with a warning");
                failures += !ok;
            }
            // A softmax CE task one word short of its op's arity, or one word long (its words moved to the argument array's end), must leave the
            // bundle as it came, with no warning: the pass never reads past a task's words, nor fuses a malformed task away into a bundle that
            // then passes validation. Prints only on failure.
            for (int delta : {-1, 1}) {
                int ce = -1;
                for (int i = 0; i < (int)orig.tasks.size() && ce < 0; ++i) if (orig.tasks[i].op_id == ops.softmax_ce_fused) ce = i;
                if (ce < 0) break;
                Bundle wb = orig; MvmTaskDesc& t = wb.tasks[ce];
                const std::vector<int32_t> words(wb.args.begin() + t.args_off, wb.args.begin() + t.args_off + t.args_words);
                t.args_off = (int32_t)wb.args.size(); t.args_words += delta;
                for (int k = 0; k < t.args_words; ++k) wb.args.push_back(k < (int)words.size() ? words[k] : 0);
                const MvmFuseResult q = mvm_fuse_mlp(wb.tasks, wb.succ, wb.args, bufs, ops, op_of, opts);
                const bool alone = q.fused_runs == 0 && q.warnings.empty() && q.args == wb.args && q.succ == wb.succ && q.tasks.size() == wb.tasks.size() &&
                                   std::equal(q.tasks.begin(), q.tasks.end(), wb.tasks.begin(), [](const MvmTaskDesc& a, const MvmTaskDesc& b) { return std::memcmp(&a, &b, sizeof a) == 0; });
                if (!alone) { printf("  FAIL: a softmax CE task with %d argument words (its op takes %d) was not left alone: fused runs %d, %zu warning(s)\n", t.args_words, t.args_words - delta, q.fused_runs, q.warnings.size()); ++failures; }
            }
        }
        // A batched entry: the bundle's one batch carries every run; each run's arithmetic is the one its shape takes alone at these workers, the
        // split form's Sv or the grid form's P (gate 2: a run's arithmetic is its shape's alone, so R copies here compute the bits one run computes
        // on its form); each run names an i32 token and target buffer of T and an f32 loss pair for every step; and each step is one gang of all
        // W members, numbered 0..W-1.
        if (r.entry.kind == MvmFuseEntry::batch || r.entry.kind == MvmFuseEntry::batch_grid) {
            std::string bad;
            const auto fail = [&](const std::string& w) { if (bad.empty()) bad = w; };
            if (r.batches.size() != 1) fail(std::to_string(r.batches.size()) + " batches, not 1");
            else {
                const MvmFuseBatch& bt = r.batches[0];
                if (bt.runs.size() != r.runs.size() || bt.steps.size() != bt.runs.size()) fail("the batch carries " + std::to_string(bt.runs.size()) + " of " + std::to_string(r.runs.size()) + " runs");
                for (size_t i = 0; i < bt.runs.size() && i < bt.steps.size(); ++i) {
                    const MvmFuseRun& run = r.runs[bt.runs[i]];
                    std::string w; const std::vector<int> fit = mvm_fuse_split_fits(run.L, run.E, run.V, run.H, run.T, ops, opts, w);
                    if (!run.batch || run.B != W) fail("run " + std::to_string(i) + " is not a batched run of " + std::to_string(W) + " members");
                    if (run.Sv > 0) {   // the split form's arithmetic: the share count the shape takes alone
                        if (fit.empty() || run.Sv != fit.back()) fail("run " + std::to_string(i) + " at Sv " + std::to_string(run.Sv) + ", its shape alone takes " + (fit.empty() ? std::string("none (") + w + ")" : std::to_string(fit.back())));
                    } else {   // the grid form's: the gang the shape takes alone, when the split form cannot hold it
                        std::string gw; const int P = mvm_fuse_grid_fits(run.L, run.E, run.V, run.H, run.T, ops, opts, gw);
                        if (!fit.empty() || run.P != P || P <= 0) fail("run " + std::to_string(i) + " at a grid of " + std::to_string(run.P) + ", its shape alone takes " + (fit.empty() ? "a grid of " + std::to_string(P) : "the split form"));
                    }
                    if ((int)bt.steps[i].size() != run.steps) fail("run " + std::to_string(i) + " names buffers for " + std::to_string(bt.steps[i].size()) + " of " + std::to_string(run.steps) + " steps");
                    for (const auto& io : bt.steps[i]) for (int k = 0; k < 3; ++k) {
                        const int b = io[k], dt = k < 2 ? 1 : 0; const long long n = k < 2 ? run.T : 2;
                        if (b < 0 || b >= (int)bufs.size() || bufs[b].dtype != dt || (long long)bufs[b].rows * bufs[b].cols < n) fail("run " + std::to_string(i) + ": buffer " + std::to_string(b) + " is not its step's " + (k == 0 ? "tokens" : k == 1 ? "targets" : "loss"));
                    }
                }
                std::map<int, std::set<int>> members; std::map<int, int> gang_step;
                for (const auto& d : r.tasks) {
                    if (d.op_id != batch_id) continue;
                    const int32_t* a = &r.args[d.args_off];
                    if (d.args_words != 5 || a[0] != 0 || a[3] != W || a[2] < 0 || a[2] >= W || a[4] < 0 || a[4] >= r.runs[0].steps) { fail("a member's arguments are not (0, gang, 0..W-1, W, step)"); continue; }
                    if (!members[a[1]].insert(a[2]).second) fail("gang " + std::to_string(a[1]) + " holds member " + std::to_string(a[2]) + " twice");
                    if (gang_step.count(a[1]) && gang_step[a[1]] != a[4]) fail("gang " + std::to_string(a[1]) + " spans two steps");
                    gang_step[a[1]] = a[4];
                }
                if ((int)members.size() != r.runs[0].steps) fail(std::to_string(members.size()) + " gangs for " + std::to_string(r.runs[0].steps) + " steps");
                for (const auto& kv : members) if ((int)kv.second.size() != W) fail("gang " + std::to_string(kv.first) + " has " + std::to_string(kv.second.size()) + " of " + std::to_string(W) + " members");
            }
            if (!bad.empty()) { printf("  FAIL: W=%d: batched: %s\n", W, bad.c_str()); ++failures; }
            else if (r.runs[0].Sv) printf("  batched gangs at W=%d: %zu runs, %d virtual shares, one gang of %d a step: ok\n", W, r.runs.size(), r.runs[0].Sv, W);
            else printf("  batched gangs at W=%d: %zu runs, the grid's arithmetic at %d members, one gang of %d a step: ok\n", W, r.runs.size(), r.runs[0].P, W);
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
        MvmSConfig cfg; cfg.workers = W;
        const MvmSCostFn cost = [&](int op, const int32_t*) { return op == slice_id ? 60.0 : op == sgd_id ? 8.0 : op == split_id ? 40.0 : op == grid_id ? 60.0 : op == batch_id ? 80.0 : op == 0 ? 0.1 : 3.0; };
        MvmSPlan p;
        try { p = mvm_static_plan(r.tasks, r.succ, r.args, cfg, op_of, cost); }
        catch (const std::exception& e) { printf("  FAIL: the planner refused the rewritten bundle at W=%d: %s\n", W, e.what()); ++failures; continue; }
        int done = 0; for (const auto& prog : p.prog) done += (int)prog.size();
        if (done != (int)r.tasks.size()) { printf("  FAIL: planner placed %d of %zu tasks at W=%d\n", done, r.tasks.size(), W); ++failures; }
        std::map<int, std::set<int>> gang_workers; std::map<int, int> gang_size;
        for (size_t t = 0; t < r.tasks.size(); ++t) if (gang_op(r.tasks[t].op_id)) { const int g = r.args[r.tasks[t].args_off + 1]; gang_workers[g].insert(p.worker_of[t]); ++gang_size[g]; }
        for (const auto& kv : gang_workers) if ((int)kv.second.size() != gang_size[kv.first]) { printf("  FAIL: gang %d shares a worker\n", kv.first); ++failures; break; }
        // the emitted records replay as the device runs them: no deadlock through any gang's barrier, every edge (successors and versions) honoured
        const mvm_test_replay::Result rp = mvm_test_replay::replay(r.tasks, r.args, mvm_static::build_dag(r.tasks, r.succ), p, op_of);
        if (!rp.ok) { printf("  FAIL: the records' replay at W=%d: %s\n", W, rp.why.c_str()); ++failures; }
        printf("  W=%d: planned %d tasks (%d gangs), predicted makespan %.1f us (work %.1f us), %d cross-worker waits; replayed %zu records%s\n", W, done, p.gangs,
               p.makespan_us, p.total_work_us, p.cross_edges, rp.records, rp.ok ? (rp.gangs ? ", every gang's barrier crossed" : "") : ", FAILED");
        }
    }
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
