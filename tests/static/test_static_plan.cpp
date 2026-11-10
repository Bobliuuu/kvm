// CPU tests of the static-schedule compiler (include/kvm_static.h): build with g++ and run on a real bundle.
//   g++ -std=c++17 -O2 -Iinclude tests/static/test_static_plan.cpp -o /tmp/test_static_plan && /tmp/test_static_plan BUNDLE_DIR [WORKERS,...]
// The check replays the EMITTED RECORDS (not the plan) the way the device runs them, in a discrete-event simulation: every task runs exactly
// once, every dependency edge is honoured by the records' waits, no deadlock, and it prints the predicted step shape.
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <vector>
#include "kvm_static.h"

static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
template <class T> static std::vector<T> load(const std::string& p) { std::string r = slurp(p); assert(r.size() % sizeof(T) == 0); std::vector<T> v(r.size() / sizeof(T)); memcpy(v.data(), r.data(), r.size()); return v; }
static long jint(const std::string& j, const char* key) { std::smatch m; if (std::regex_search(j, m, std::regex(std::string("\"") + key + "\"\\s*:\\s*(-?\\d+)"))) return atol(m[1].str().c_str()); fprintf(stderr, "missing %s\n", key); exit(2); }

// The registry from the .def files (X(name, id, "sig") lines).
static std::map<int, KvmSOpInfo> g_ops; static std::map<int, std::string> g_names;
static void load_defs(const std::string& inc) {
    for (const char* f : {"kvm_ops.def", "kvm_ops_gpt.def", "kvm_ops_extra.def", "kvm_ops_comm.def", "kvm_ops_game.def"}) {
        std::ifstream in(inc + "/" + f); std::string line;
        std::regex re("^\\s*X\\(\\s*(\\w+)\\s*,\\s*(\\d+)\\s*,\\s*\"([^\"]*)\"");
        while (std::getline(in, line)) {
            std::smatch m; if (!std::regex_search(line, m, re)) continue;
            int id = std::stoi(m[2].str()); std::string sig = m[3].str(); KvmSOpInfo o; o.arity = 0; o.types = 0;
            std::stringstream ss(sig); std::string tok; int i = 0;
            while (std::getline(ss, tok, ',')) { auto c = tok.find(':'); std::string ty = tok.substr(c + 1); int t = ty == "buf" ? 0 : ty == "i32" ? 1 : ty == "f32" ? 2 : ty == "buf?" ? 3 : -1; assert(t >= 0); o.types |= (uint64_t)t << (3 * i++); o.arity++; }
            g_ops[id] = o; g_names[id] = m[1].str();
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s BUNDLE_DIR [workers,...] (run from the repo root)\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    std::vector<int> wlist = {1, 2, 4, 8, 16, 32, 64, 107};
    if (argc > 2) { wlist.clear(); std::stringstream s(argv[2]); std::string t; while (std::getline(s, t, ',')) wlist.push_back(std::stoi(t)); }
    load_defs("include");
    std::string meta = slurp(dir + "/metadata.json");
    const int nt = jint(meta, "task_count");
    auto tasks = load<KvmTaskDesc>(dir + "/tasks.bin"); auto succ = load<int32_t>(dir + "/successors.bin"); auto args = load<int32_t>(dir + "/args.bin");
    assert((int)tasks.size() == nt);
    KvmSOpLookup op_of = [](int id) { auto it = g_ops.find(id); return it == g_ops.end() ? KvmSOpInfo{} : it->second; };
    KvmSCostFn cost = [](int op, const int32_t*) { return op == 0 ? 0.1 : 5.0; };   // flat model for the structural checks
    int nbufs = 0; for (auto& t : tasks) { KvmSOpInfo o = op_of(t.op_id); for (int j = 0; j < t.args_words; ++j) { int ty = (o.types >> (3 * j)) & 7; if ((ty == 0 || ty == 3) && args[t.args_off + j] >= nbufs) nbufs = args[t.args_off + j] + 1; } }
    std::vector<KvmBuf> bufs(nbufs); for (int i = 0; i < nbufs; ++i) { bufs[i].p = (void*)(uintptr_t)(0x1000 + 64 * i); bufs[i].rows = bufs[i].cols = bufs[i].ld = 1; bufs[i].dtype = 0; }
    auto dag = kvm_static::build_dag(tasks, succ);
    long edges = 0; for (auto& e : dag) edges += (long)e.size();
    printf("bundle %s: %d tasks, %ld edges (successors + versions), %d buffers\n", dir.c_str(), nt, edges, nbufs);
    for (int W : wlist) {
        KvmSConfig cfg; cfg.workers = W;
        KvmSPlan p = kvm_static_plan(tasks, succ, args, cfg, op_of, cost);
        KvmSLayout l = kvm_static_layout(p);
        std::vector<std::vector<KvmSRec>> st(W);
        int nrec = 0;
        for (int w = 0; w < W; ++w) { st[w].resize(l.nrecs[w]); nrec += l.nrecs[w]; if (l.nrecs[w]) kvm_static_emit(p, l, w, tasks, args, bufs.data(), nbufs, op_of, st[w].data()); }
        // replay: each worker a sequence; a record starts when its worker is free and the watermarks it waits for have been published
        std::vector<int> pc(W, 0); std::vector<double> ft(W, 0.0); std::vector<int> progress(W, 0);
        struct Ev { double t; int w, val; bool operator>(const Ev& o) const { return t > o.t; } };
        std::vector<double> task_end(nt, -1), task_start(nt, -1); std::vector<int> runs(nt, 0);
        // per worker, the watermark times: vt[w][k] = time at which progress[w] reached k (from the records' publishes)
        std::vector<std::vector<double>> vt(W);
        for (int w = 0; w < W; ++w) vt[w].assign(st[w].size() + 2, 1e300);
        // iterate to a fixpoint: simple round-robin relaxation (programs are finite and acyclic by construction; a stall is a deadlock)
        bool progressed = true; int done = 0;
        for (int w = 0; w < W; ++w) vt[w][0] = 0;
        std::vector<int> cur(W, 0);
        while (progressed && done < nrec) {
            progressed = false;
            for (int w = 0; w < W; ++w) {
                while (cur[w] < (int)st[w].size()) {
                    const KvmSRec& r = st[w][cur[w]];
                    double t = ft[w]; bool ready = true;
                    for (int j = 0; j < r.nwait; ++j) { int nw = r.waits[j].worker, need = r.waits[j].need; if (vt[nw][need] > 1e200) { ready = false; break; } t = std::max(t, vt[nw][need]); }
                    if (!ready) break;
                    double c = (r.flags & KVM_S_WAITONLY) ? 0.0 : (r.op == 0 ? 0.1 : 5.0);
                    double end = t + c;
                    if (r.nwait) end += 1.5;   // hop model
                    if (!(r.flags & KVM_S_WAITONLY)) { if (runs[r.task]++) { printf("FAIL: task %d run twice\n", r.task); return 1; } task_start[r.task] = t; task_end[r.task] = end; }
                    ft[w] = end;
                    if (r.flags & KVM_S_PUBLISH) { for (int k = 0; k <= r.pos1; ++k) vt[w][k] = std::min(vt[w][k], end); }
                    assert(r.pos1 == cur[w] + 1);
                    cur[w]++; done++; progressed = true;
                }
            }
        }
        if (done != nrec) { printf("FAIL W=%d: DEADLOCK after %d of %d records\n", W, done, nrec); return 1; }
        double makespan = *std::max_element(ft.begin(), ft.end());
        for (int t = 0; t < nt; ++t) if (runs[t] != 1) { printf("FAIL: task %d ran %d times\n", t, runs[t]); return 1; }
        // honoured precedence, from the DAG itself (independent of the records' waits)
        int bad = 0; for (int u = 0; u < nt; ++u) for (int v : dag[u]) if (task_start[v] < task_end[u] - 1e-9) { if (!bad++) printf("FAIL W=%d: edge %d -> %d not honoured (start %.2f < end %.2f, workers %d %d)\n", W, u, v, task_start[v], task_end[u], p.worker_of[u], p.worker_of[v]); }
        if (bad) return 1;
        int used = 0; for (int w = 0; w < W; ++w) used += l.nrecs[w] > 0;
        int waitonly = nrec - nt;
        printf("W=%3d: used %3d workers, records %6d (wait-only %d), cross edges %5d of %5ld (%.0f%%), publishing tasks %d, plan makespan %.1f us, replay %.1f us, work %.1f us\n",
               W, used, nrec, waitonly, p.cross_edges, edges, 100.0 * p.cross_edges / std::max(1l, edges), p.publishing, p.makespan_us, makespan, p.total_work_us);
    }
    printf("PASS\n");
    return 0;
}
