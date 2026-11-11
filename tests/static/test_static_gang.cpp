// CPU tests of gang tasks in the static-schedule compiler (include/kvm_static.h): a gang op's members run at once on distinct workers and
// meet at barriers inside the op.
//   g++ -std=c++17 -O2 -Iinclude tests/static/test_static_gang.cpp -o /tmp/test_static_gang && /tmp/test_static_gang
// Bundles are synthetic: chains of gangs (a fused step's members, step after step) with ordinary tasks around them, in three wirings
// (member m of a step depends on member m of the last, on every member of the last, or through ordinary joins). The checks:
//  * every gang's members are on distinct workers;
//  * the EMITTED RECORDS replay without deadlock when a gang member, at its worker's head with its waits met, holds that worker until every
//    member of its gang is there (the barrier);
//  * every dependency edge is honoured by the replay's order;
//  * plans of bundles without gangs are unchanged by the unit contraction (the same programs and waits as tasks planned one by one);
//  * malformed gangs are refused: a member depending on another member (directly or through other tasks), more members than workers,
//    a member missing or given twice, members that disagree on their count.
#include <cassert>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "kvm_static.h"
#include "replay.h"

namespace {

constexpr int OP_NOOP = 0, OP_WORK = 1, OP_GANG = 2;
int g_fail = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++g_fail; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

KvmSOpInfo op_info(int id) {
    KvmSOpInfo o;
    if (id == OP_NOOP) { o.arity = 0; }
    else if (id == OP_WORK) { o.arity = 1; o.types = 1; }   // x:i32
    else if (id == OP_GANG) { o.arity = 3; o.types = 1 | (1 << 3) | (1 << 6); kvm_static_gang_args("gang:i32,member:i32,members:i32", o); }
    return o;
}
double op_cost(int id, const int32_t* a) { return id == OP_GANG ? 20.0 : id == OP_WORK ? 1.0 + (double)((uint32_t)a[0] % 40) : 0.1; }

struct Bundle {
    std::vector<KvmTaskDesc> tasks;
    std::vector<int32_t> succ, args;
    std::vector<std::vector<int>> out;   // edges as built (before the CSR)
    int add(int op, std::vector<int32_t> a) {
        KvmTaskDesc d{}; d.op_id = op; d.args_off = (int)args.size(); d.args_words = (int)a.size();
        d.wait_var = d.pub_var = KVM_NO_VAR;
        for (int32_t x : a) args.push_back(x);
        tasks.push_back(d); out.emplace_back();
        return (int)tasks.size() - 1;
    }
    void edge(int u, int v) { out[u].push_back(v); }
    void finish() {
        succ.clear();
        for (size_t i = 0; i < tasks.size(); ++i) {
            std::sort(out[i].begin(), out[i].end()); out[i].erase(std::unique(out[i].begin(), out[i].end()), out[i].end());
            tasks[i].succ_off = (int)succ.size(); tasks[i].succ_cnt = (int)out[i].size();
            for (int v : out[i]) succ.push_back(v);
        }
        std::vector<int> indeg(tasks.size(), 0);
        for (auto& e : out) for (int v : e) ++indeg[v];
        for (size_t i = 0; i < tasks.size(); ++i) tasks[i].indegree = indeg[i];
    }
};

// Steps of one fused op spread over `members` blocks, with `extra` ordinary tasks around each step; `wiring` 0: member m <- member m of the
// last step; 1: every member <- every member of the last step; 2: the last step's members -> one ordinary join -> every member; 3 (stress):
// members of unequal rank, each with its own predecessors among the last step's tasks and its own chain of 1-4 tasks after it, while long
// tasks (up to 40 against a member's 20) hold workers, so planning members one at a time stacks two on a worker or orders one behind a task
// that waits on another member.
Bundle make(int steps, int members, int extra, int wiring, unsigned seed) {
    std::mt19937 rng(seed);
    Bundle b;
    std::vector<int> prev;
    int gang = 0;
    for (int s = 0; s < steps; ++s) {
        std::vector<int> pre;
        for (int e = 0; e < extra; ++e) { pre.push_back(b.add(OP_WORK, {(int32_t)rng()})); if (!prev.empty()) b.edge(prev[rng() % prev.size()], pre.back()); }
        int join = -1;
        if (wiring == 2 && !prev.empty()) { join = b.add(OP_NOOP, {}); for (int q : prev) b.edge(q, join); }
        (void)join;
        std::vector<int> cur;
        for (int m = 0; m < members; ++m) cur.push_back(b.add(OP_GANG, {gang, m, members}));
        for (int m = 0; m < members; ++m) {
            if (!prev.empty()) {
                if (wiring == 0) b.edge(prev[m], cur[m]);
                else if (wiring == 1) for (int q : prev) b.edge(q, cur[m]);
                else if (wiring == 2) b.edge(join, cur[m]);
                else { const int k = 1 + (int)(rng() % prev.size()); for (int j = 0; j < k; ++j) b.edge(prev[rng() % prev.size()], cur[m]); }
            }
            if (!pre.empty()) b.edge(pre[rng() % pre.size()], cur[m]);
        }
        if (wiring == 3) {   // each member its own chain; long tasks fed by random members
            std::vector<int> tails;
            for (int m = 0; m < members; ++m) {
                int last = cur[m];
                const int len = 1 + (int)(rng() % 4);
                for (int k = 0; k < len; ++k) { const int t = b.add(OP_WORK, {(int32_t)rng()}); b.edge(last, t); last = t; }
                tails.push_back(last);
            }
            for (int e = 0; e < extra; ++e) { const int t = b.add(OP_WORK, {(int32_t)(39 + 40 * (rng() % 1000))}); b.edge(cur[rng() % cur.size()], t); tails.push_back(t); }
            prev = tails;   // the next step's members draw their predecessors from these
            ++gang;
            continue;
        }
        for (int e = 0; e < extra; ++e) { const int t = b.add(OP_WORK, {(int32_t)rng()}); b.edge(cur[rng() % cur.size()], t); }
        prev = cur; ++gang;
    }
    b.finish();
    return b;
}

// Gang membership read from the bundle itself, never from the plan under test: per task its gang id, -1 for none
std::vector<int> gangs_of(const Bundle& b) {
    std::vector<int> g(b.tasks.size(), -1);
    for (size_t t = 0; t < b.tasks.size(); ++t) if (b.tasks[t].op_id == OP_GANG) g[t] = b.args[b.tasks[t].args_off];
    return g;
}

// Replays the emitted records as the device runs them (tests/static/replay.h). Returns false on a deadlock or a broken edge.
bool replay(const Bundle& b, const KvmSPlan& p, int W, std::string& why) {
    if (p.workers != W) { why = "the plan has " + std::to_string(p.workers) + " workers, not " + std::to_string(W); return false; }
    const kvm_test_replay::Result r = kvm_test_replay::replay(b.tasks, b.args, b.out, p, op_info);
    why = r.why;
    return r.ok;
}

bool distinct_workers(const KvmSPlan& p, const Bundle& b, std::string& why) {
    const std::vector<int> gang_of = gangs_of(b);
    std::map<int, std::set<int>> ws;
    std::map<int, int> count;
    for (size_t t = 0; t < b.tasks.size(); ++t) if (gang_of[t] >= 0) { ws[gang_of[t]].insert(p.worker_of[t]); ++count[gang_of[t]]; }
    for (auto& kv : ws) if ((int)kv.second.size() != count[kv.first]) { why = "gang " + std::to_string(kv.first) + " shares a worker"; return false; }
    return true;
}

bool refused(const Bundle& b, int W, const char* expect) {
    KvmSConfig cfg; cfg.workers = W;
    try { kvm_static_plan(b.tasks, b.succ, b.args, cfg, op_info, op_cost); }
    catch (const std::invalid_argument& e) {
        const bool ok = std::string(e.what()).find(expect) != std::string::npos;
        if (!ok) printf("  refused with \"%s\", expected \"%s\"\n", e.what(), expect);
        return ok;
    }
    return false;
}

}  // namespace

int main() {
    int plans = 0;
    // 1. valid bundles: every wiring, gang widths 1-8, worker counts from the widest gang up to the A100's 108
    for (int wiring = 0; wiring < 4; ++wiring)
        for (int members : {1, 2, 3, 5, 8})
            for (int extra : {0, 3})
                for (int W : {members, members + 1, 2 * members + 3, 16, 108})
                  for (unsigned seed = 0; seed < (wiring == 3 ? 8u : 1u); ++seed) {
                    if (W < members) continue;
                    const Bundle b = make(6, members, extra, wiring, 1000u * wiring + 10u * members + extra + 100000u * seed);
                    KvmSConfig cfg; cfg.workers = W;
                    KvmSPlan p;
                    try { p = kvm_static_plan(b.tasks, b.succ, b.args, cfg, op_info, op_cost); }
                    catch (const std::exception& e) { EXPECT(false, "wiring %d members %d extra %d W %d: refused: %s", wiring, members, extra, W, e.what()); continue; }
                    std::string why;
                    EXPECT(p.gangs == 6, "wiring %d members %d W %d: %d gangs, expected 6", wiring, members, W, p.gangs);
                    EXPECT(distinct_workers(p, b, why), "wiring %d members %d W %d: %s", wiring, members, W, why.c_str());
                    EXPECT(replay(b, p, W, why), "wiring %d members %d extra %d W %d: %s", wiring, members, extra, W, why.c_str());
                    ++plans;
                }
    printf("valid bundles: %d plans checked (distinct workers, deadlock-free replay of the emitted records, every edge honoured)\n", plans);

    // 2. a bundle without gangs plans as before: the same bundle with its gang ops turned into ordinary ones (members > 1 widths only need the
    // argument count to match, so OP_WORK with one word stands in) gives identical programs and waits to planning with gangs of width 1
    {
        Bundle b = make(8, 1, 3, 1, 7u);   // width-1 gangs: each a unit of one task
        Bundle o = b;
        for (auto& t : o.tasks) if (t.op_id == OP_GANG) { t.op_id = OP_WORK; t.args_words = 1; }   // reads the gang id as x
        // costs differ (a gang costs 20, the stand-in 1 + x % 7), so compare against a cost function that charges the same
        KvmSCostFn same = [&](int id, const int32_t* a) { return id == OP_GANG || id == OP_WORK ? 1.0 + (a[0] % 7) : 0.1; };
        for (int W : {1, 4, 108}) {
            KvmSConfig cfg; cfg.workers = W;
            const KvmSPlan pg = kvm_static_plan(b.tasks, b.succ, b.args, cfg, op_info, same);
            const KvmSPlan po = kvm_static_plan(o.tasks, o.succ, o.args, cfg, op_info, same);
            bool equal = pg.prog == po.prog && pg.worker_of == po.worker_of;
            for (size_t t = 0; equal && t < b.tasks.size(); ++t) {
                equal = pg.waits[t].size() == po.waits[t].size();
                for (size_t k = 0; equal && k < pg.waits[t].size(); ++k) equal = pg.waits[t][k].worker == po.waits[t][k].worker && pg.waits[t][k].need == po.waits[t][k].need;
            }
            EXPECT(equal, "W %d: width-1 gangs plan differently from the same tasks without a gang", W);
        }
        printf("width-1 gangs plan exactly as ordinary tasks (W = 1, 4, 108)\n");
    }

    // 3. malformed gangs are refused, each with its reason
    {
        Bundle b = make(2, 4, 0, 0, 3u);
        b.out[b.tasks.size() - 4].push_back((int)b.tasks.size() - 3);   // member 0 -> member 1 of the last gang
        b.finish();
        EXPECT(refused(b, 8, "depends on member task"), "a member depending on another member is accepted");
    }
    {
        Bundle b = make(2, 4, 0, 0, 3u);
        const int x = b.add(OP_WORK, {1});
        const int n = (int)b.tasks.size() - 1;   // x itself
        b.out[n - 4].push_back(x);   // last gang's member 0 -> x
        b.out[x].push_back(n - 3);   // x -> last gang's member 1
        b.finish();
        EXPECT(refused(b, 8, "dependency cycle"), "a gang reaching itself through another task is accepted");
    }
    {
        const Bundle b = make(2, 8, 0, 0, 3u);
        EXPECT(refused(b, 7, "needs 8 workers"), "a gang wider than the workers is accepted");
    }
    {
        Bundle b = make(1, 3, 0, 0, 3u);
        b.args[b.tasks[2].args_off + 1] = 1;   // member 2 claims to be member 1
        EXPECT(refused(b, 4, "two tasks are member 1"), "a member given twice is accepted");
    }
    {
        Bundle b = make(1, 3, 0, 0, 3u);
        b.args[b.tasks[2].args_off + 2] = 4;   // member 2 says the gang has 4 members
        EXPECT(refused(b, 8, "disagree"), "members disagreeing on their count are accepted");
    }
    {
        Bundle b = make(1, 3, 0, 0, 3u);
        for (int k = 0; k < 3; ++k) b.args[b.tasks[k].args_off + 2] = 4;   // all say 4, member 3 is absent
        EXPECT(refused(b, 8, "member 3 is missing"), "a gang with a missing member is accepted");
    }
    printf("malformed gangs: 6 refusal cases checked\n");
    if (g_fail) { printf("FAILED: %d check(s)\n", g_fail); return 1; }
    printf("all gang checks passed\n");
    return 0;
}
