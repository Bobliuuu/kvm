#pragma once
// Replays a static plan's EMITTED RECORDS as the device runs them, for the static path's CPU tests (g++ -Iinclude):
//  * each worker runs its program in order, and a record starts when every progress watermark it waits for is published;
//  * a gang op's member (kvm_static_plan's gang tasks), at its worker's head with its waits met, holds that worker until every member of its
//    gang is there: the barrier inside the op. Gang membership is read from the bundle's own arguments through the registry signature
//    (`gang` and `members`), never from the plan under test.
// It fails on a deadlock (records that never run) and on a dependency edge the order breaks: `edges` is the DAG the plan must honour,
// kvm_static::build_dag's successors plus version orderings for a real bundle.
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "kvm_static.h"

namespace kvm_test_replay {

struct Result { bool ok = false; size_t records = 0; int gangs = 0; std::string why; };

inline Result replay(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& args, const std::vector<std::vector<int>>& edges,
                     const KvmSPlan& p, const KvmSOpLookup& op_of) {
    Result res;
    const int W = p.workers, nt = (int)tasks.size();
    // per task: its gang id and member count, from the bundle (-1: no gang)
    std::vector<int> gang_of(nt, -1), members_of(nt, 0);
    int nbufs = 0;
    for (int t = 0; t < nt; ++t) {
        const KvmSOpInfo o = op_of(tasks[t].op_id);
        const int32_t* a = args.data() + tasks[t].args_off;
        if (o.gang_arg >= 0) { gang_of[t] = a[o.gang_arg]; members_of[t] = a[o.members_arg]; }
        for (int j = 0; j < tasks[t].args_words; ++j) {
            const int ty = (int)((o.types >> (3 * j)) & 7u);
            if ((ty == 0 || ty == 3) && a[j] >= nbufs) nbufs = a[j] + 1;
        }
    }
    // buffer views only fill the records: any distinct, well-formed view will do
    std::vector<KvmBuf> bufs(nbufs);
    for (int i = 0; i < nbufs; ++i) { bufs[i] = KvmBuf{}; bufs[i].p = (void*)(uintptr_t)(0x1000 + 64 * i); bufs[i].rows = bufs[i].cols = bufs[i].ld = 1; }
    const KvmSLayout l = kvm_static_layout(p);
    std::vector<std::vector<KvmSRec>> st(W);
    for (int w = 0; w < W; ++w) {
        st[w].resize(l.nrecs[w]);
        if (l.nrecs[w]) kvm_static_emit(p, l, w, tasks, args, bufs.data(), nbufs, op_of, st[w].data());
        res.records += st[w].size();
    }
    std::vector<int> head(W, 0), progress(W, 0);
    std::vector<long> start(nt, -1), done(nt, -1);
    std::map<int, std::set<int>> arrived;   // gang -> workers whose member is at the barrier
    std::set<int> seen_gangs;
    long clock = 0;
    size_t finished = 0;
    auto complete = [&](int w) {
        const KvmSRec& r = st[w][head[w]];
        if (!(r.flags & KVM_S_WAITONLY)) done[r.task] = clock;
        if (r.flags & KVM_S_PUBLISH) progress[w] = r.pos1;
        ++head[w]; ++finished;
    };
    for (bool moved = true; moved;) {
        moved = false;
        for (int w = 0; w < W; ++w) {
            if (head[w] >= (int)st[w].size()) continue;
            const KvmSRec& r = st[w][head[w]];
            bool ready = true;
            for (int k = 0; k < r.nwait; ++k) if (progress[r.waits[k].worker] < r.waits[k].need) ready = false;
            if (!ready) continue;
            const int g = (r.flags & KVM_S_WAITONLY) ? -1 : gang_of[r.task];
            if (g < 0) { start[r.task] = ++clock; complete(w); moved = true; continue; }
            if (arrived[g].insert(w).second) { start[r.task] = ++clock; moved = true; seen_gangs.insert(g); }
            if ((int)arrived[g].size() == members_of[r.task]) {   // the barrier opens: every member completes
                ++clock;
                for (int x : arrived[g]) complete(x);
                arrived.erase(g);
            }
        }
    }
    res.gangs = (int)seen_gangs.size();
    if (finished != res.records) { res.why = "deadlock: " + std::to_string(res.records - finished) + " of " + std::to_string(res.records) + " records never ran"; return res; }
    for (int u = 0; u < (int)edges.size(); ++u)
        for (int v : edges[u])
            if (!(done[u] >= 0 && start[v] >= 0 && done[u] < start[v])) {
                res.why = "edge " + std::to_string(u) + " -> " + std::to_string(v) + " broken (done " + std::to_string(done[u]) + ", start " + std::to_string(start[v]) + ")";
                return res;
            }
    res.ok = true;
    return res;
}

}  // namespace kvm_test_replay
