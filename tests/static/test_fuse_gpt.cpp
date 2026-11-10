// CPU check of the GPT step matcher (include/kvm_fuse_gpt.h) on real bundles: prints the decision for every run, and on a matched run checks
// that its steps account for every task of the run exactly once, prints the step's shape and its parameter updates by buffer name, then
// breaks the bundle in ways the template must refuse (each mutation must leave the run unmatched, with its own reason).
//   g++ -std=c++17 -O2 -Iinclude tests/static/test_fuse_gpt.cpp -o /tmp/test_fuse_gpt && /tmp/test_fuse_gpt BUNDLE_DIR...   (from the repo root)
#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "bundle_io.h"
#include "kvm_fuse_gpt.h"

namespace {

KvmFuseGptOps gpt_ops(const bundle_io::Registry& reg) {
    const auto id = [&](const char* n) { auto it = reg.ids.find(n); return it == reg.ids.end() ? -1 : it->second; };
    KvmFuseGptOps o;
    o.select_window = id("select_window"); o.embedding_fwd = id("embedding_fwd"); o.layernorm_fwd = id("layernorm_fwd"); o.matmul_tile = id("matmul_tile");
    o.rope_fwd = id("rope_fwd"); o.attn_fwd = id("attn_fwd"); o.softmax_ce_fused = id("softmax_ce_fused"); o.softmax_ce_reduce = id("softmax_ce_reduce");
    o.layernorm_bwd_dx = id("layernorm_bwd_dx"); o.norm_bwd_params = id("norm_bwd_params"); o.colsum = id("colsum"); o.attn_bwd_delta = id("attn_bwd_delta");
    o.attn_bwd_dq = id("attn_bwd_dq"); o.attn_bwd_dkv = id("attn_bwd_dkv"); o.rope_bwd = id("rope_bwd"); o.embedding_bwd = id("embedding_bwd");
    o.gradnorm_partial = id("gradnorm_partial"); o.gradnorm_finish = id("gradnorm_finish"); o.counter_inc = id("counter_inc"); o.adamw = id("adamw"); o.noop = id("noop");
    return o;
}

struct Bundle { std::vector<KvmTaskDesc> tasks; std::vector<int32_t> succ, args; bundle_io::Buffers bufs; std::vector<std::string> names; };

// The first task of an op in the bundle (to mutate), or -1.
int first_of(const Bundle& b, int op) { for (size_t i = 0; i < b.tasks.size(); ++i) if (b.tasks[i].op_id == op) return (int)i; return -1; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s [--refuse] BUNDLE_DIR... (from the repo root; bundles after --refuse must match no GPT run)\n", argv[0]); return 2; }
    const bundle_io::Registry reg = bundle_io::load_registry("include");
    const KvmFuseGptOps ops = gpt_ops(reg);
    int failures = 0;
    bool refuse = false;
    for (int ai = 1; ai < argc; ++ai) {
        if (std::string(argv[ai]) == "--refuse") { refuse = true; continue; }
        const std::string dir = argv[ai];
        const std::string meta = bundle_io::slurp(dir + "/metadata.json");
        Bundle B{bundle_io::load<KvmTaskDesc>(dir + "/tasks.bin"), bundle_io::load<int32_t>(dir + "/successors.bin"), bundle_io::load<int32_t>(dir + "/args.bin"),
                 bundle_io::load_buffers(meta), bundle_io::load_task_names(meta)};
        const KvmFuseGptResult r = kvm_fuse_gpt_scan(B.tasks, B.succ, B.args, B.bufs.shape, ops);
        printf("%s: %zu tasks, %zu buffers -> matched runs %d, unchanged runs %d\n", dir.c_str(), B.tasks.size(), B.bufs.shape.size(), r.matched_runs, r.unchanged_runs);
        for (const auto& line : r.report) printf("  %s\n", line.c_str());
        if (refuse) {   // a bundle that is not a GPT training run: every run must stay as written
            if (r.matched_runs) { printf("  FAIL: %d runs matched in a bundle that must be refused\n", r.matched_runs); ++failures; }
            else printf("  refused, as it must be\n");
            continue;
        }
        if (!r.matched_runs) { printf("  FAIL: no GPT run matched\n"); ++failures; continue; }

        // 1. every task of a matched run is a data window, a replaced task of exactly one step, or a step's closing barrier
        const auto comps = kvm_fuse_components(B.tasks, B.succ);
        for (const KvmFuseGptRun& run : r.runs) {
            const std::vector<int>& comp = comps[run.component];
            std::vector<int> seen(B.tasks.size(), 0);
            // re-parse step by step to collect every step's tasks (the run keeps step 0's description)
            const kvm_fuse::GptParser parser(B.tasks, B.args, B.bufs.shape, ops);
            size_t pos = 0; std::string why;
            for (int st = 0; st < run.steps; ++st) {
                KvmGptStep s;
                if (!parser.parse(comp, pos, s, why)) { printf("  FAIL: step %d no longer parses: %s\n", st, why.c_str()); ++failures; break; }
                for (int t : s.replaced) ++seen[t];
                if (s.barrier_end >= 0) ++seen[s.barrier_end];
            }
            int windows = 0, twice = 0, missed = 0;   // the data windows belong to their steps too: the fused run reads the streams itself
            for (int t : comp) {
                windows += B.tasks[t].op_id == ops.select_window;
                if (seen[t] == 0) { if (missed++ < 3) printf("  FAIL: task %d (%s) belongs to no step\n", t, t < (int)B.names.size() ? B.names[t].c_str() : "?"); }
                else if (seen[t] > 1) ++twice;
            }
            if (missed || twice) { printf("  FAIL: %d tasks in no step, %d in two\n", missed, twice); ++failures; }
            else printf("  coverage: all %zu tasks of the run, each in exactly one step (%d of them data windows)\n", comp.size(), windows);
            printf("  streams: tokens from buffer %s (stride %d, offset %d), targets from %s (stride %d, offset %d); %zu loss buffers, one a step\n",
                   B.bufs.name[run.shape.ids_src].c_str(), run.shape.ids_stride, run.shape.ids_off, B.bufs.name[run.shape.tgt_src].c_str(), run.shape.tgt_stride,
                   run.shape.tgt_off, run.losses.size());

            // 2. the step's shape and its updates, by name (a reader checks them against the model)
            const KvmGptStep& s = run.shape;
            printf("  shape: L %d, d %d, heads %d x %d, MLP %d, V %d, B %d x T %d; LayerNorm eps %g, RoPE theta %g, attention scale %g\n", s.L, s.d, s.H, s.hd, s.F, s.V, s.B, s.T, s.ln_eps, s.theta, s.scale);
            printf("  AdamW lr %g, betas %g %g, eps %g; clip %g; %zu updates:\n", s.lr, s.beta1, s.beta2, s.adam_eps, s.max_norm, s.updates.size());
            for (const KvmGptUpdate& u : s.updates)
                printf("    %-24s grad %-6s m %-24s v %-24s wd %g\n", B.bufs.name[u.p].c_str(), B.bufs.name[u.g].c_str(), B.bufs.name[u.m].c_str(), B.bufs.name[u.v].c_str(), u.wd);
        }

        // 3. the rewrite: each matched run becomes one gang of G fgpt_step members; every other task is copied through; the static planner places the gang
        {
            const auto fgpt = reg.ids.find("fgpt_step");
            if (fgpt == reg.ids.end()) { printf("  FAIL: the registry has no fgpt_step\n"); ++failures; continue; }
            const KvmSOpLookup op_of = [&](int i) { auto it = reg.ops.find(i); return it == reg.ops.end() ? KvmSOpInfo{} : it->second; };
            KvmFuseGptOptions opt;
            const KvmFuseGptPlan P = kvm_fuse_gpt(B.tasks, B.succ, B.args, B.bufs.shape, ops, fgpt->second, opt);
            printf("  rewrite: %d fused, %d unchanged, %d tasks replaced by %d members\n", P.fused_runs, P.unchanged_runs, P.replaced_tasks, P.new_tasks);
            for (const auto& line : P.report) printf("    %s\n", line.c_str());
            // a matched run that is not fused must be refused for its fit alone (a run too large for a resident gang, like the 162M tape)
            int misrefused = 0;
            for (const auto& run : r.runs) {
                const std::string& line = P.report[run.component];
                if (line.find("fused as a gang") == std::string::npos && line.find("does not fit a resident gang") == std::string::npos) ++misrefused;
            }
            if (misrefused) { printf("  FAIL: %d matched runs refused for a reason other than their fit\n", misrefused); ++failures; }
            // kept tasks keep their op and arguments: the multiset of (op, args) of the non-member tasks is the original's minus the fused runs' tasks
            std::map<std::vector<int>, int> before, after;
            std::set<int> fused_comp;
            for (const auto& run : P.runs) fused_comp.insert(run.component);
            for (size_t ci = 0; ci < comps.size(); ++ci) {
                if (fused_comp.count((int)ci)) continue;
                for (int t : comps[ci]) { std::vector<int> k{B.tasks[t].op_id}; k.insert(k.end(), B.args.begin() + B.tasks[t].args_off, B.args.begin() + B.tasks[t].args_off + B.tasks[t].args_words); ++before[k]; }
            }
            std::map<int, std::vector<int>> members;   // gang -> members seen
            int bad_member = 0;
            for (size_t i = 0; i < P.tasks.size(); ++i) {
                const KvmTaskDesc& d = P.tasks[i];
                const int32_t* a = P.args.data() + d.args_off;
                if (d.op_id == fgpt->second) {
                    if (d.args_words != 4 || a[0] != a[1] || a[0] < 0 || a[0] >= (int)P.runs.size() || a[3] != P.gang[a[0]] || a[2] < 0 || a[2] >= a[3] || d.succ_cnt || d.indegree) ++bad_member;
                    members[a[1]].push_back(a[2]);
                    continue;
                }
                std::vector<int> k{d.op_id}; k.insert(k.end(), a, a + d.args_words); ++after[k];
            }
            for (auto& kv : members) { std::sort(kv.second.begin(), kv.second.end()); for (int m = 0; m < (int)kv.second.size(); ++m) bad_member += kv.second[m] != m; }
            if (before != after) { printf("  FAIL: the kept tasks changed\n"); ++failures; }
            // the static path takes a bundle's final versions from its publishers: the rewrite keeps every variable's last version
            std::map<int, int> vb, va;
            for (const auto& t : B.tasks) if (t.pub_var >= 0) vb[t.pub_var] = std::max(vb[t.pub_var], t.pub_ver);
            for (const auto& t : P.tasks) if (t.pub_var >= 0) va[t.pub_var] = std::max(va[t.pub_var], t.pub_ver);
            if (vb != va) { printf("  FAIL: the rewrite changed a final version\n"); ++failures; }
            else if (!vb.empty()) printf("  final versions kept: %zu variable(s), e.g. variable %d at %d\n", vb.size(), vb.begin()->first, vb.begin()->second);
            if (bad_member || (int)members.size() != P.fused_runs) { printf("  FAIL: %d malformed gang members, %zu gangs for %d runs\n", bad_member, members.size(), P.fused_runs); ++failures; }
            // the static planner places every task, and puts each gang's members on distinct workers
            for (int W : {8, 32, 108}) {
                if (P.fused_runs && W < *std::max_element(P.gang.begin(), P.gang.end())) continue;
                KvmSConfig cfg; cfg.workers = W;
                const KvmSCostFn cost = [&](int op, const int32_t*) { return op == fgpt->second ? 100.0 : op == 0 ? 0.1 : 3.0; };
                const KvmSPlan plan = kvm_static_plan(P.tasks, P.succ, P.args, cfg, op_of, cost);
                int placed = 0; for (const auto& prog : plan.prog) placed += (int)prog.size();
                std::map<int, std::set<int>> workers_of_gang;
                for (size_t i = 0; i < P.tasks.size(); ++i) if (P.tasks[i].op_id == fgpt->second) workers_of_gang[P.args[P.tasks[i].args_off + 1]].insert(plan.worker_of[i]);
                bool spread = true; for (const auto& kv : workers_of_gang) spread = spread && (int)kv.second.size() == P.gang[kv.first];
                printf("  W=%d: planned %d of %zu tasks, each gang on distinct workers: %s, predicted makespan %.1f us\n", W, placed, P.tasks.size(), spread ? "yes" : "NO", plan.makespan_us);
                if (placed != (int)P.tasks.size() || !spread) ++failures;
            }
            // refusals of the rewrite, on a run it fuses by default: a budget no gang fits, and a dumped activation the fused step would leave stale
            if (P.fused_runs) {
                KvmFuseGptOptions tight; tight.smem_floats = 1000;
                const KvmFuseGptPlan T = kvm_fuse_gpt(B.tasks, B.succ, B.args, B.bufs.shape, ops, fgpt->second, tight);
                KvmFuseGptOptions dump; if (!r.runs.empty()) dump.keep = {*r.runs[0].shape.interior.begin()};
                const KvmFuseGptPlan D = kvm_fuse_gpt(B.tasks, B.succ, B.args, B.bufs.shape, ops, fgpt->second, dump);
                const bool tight_ok = !T.fused_runs && T.report[r.runs[0].component].find("does not fit a resident gang") != std::string::npos;
                const bool dump_ok = !D.fused_runs && D.report[r.runs[0].component].find("stale") != std::string::npos;
                printf("  rewrite refusals: budget of 1000 floats %s; a dumped activation %s\n", tight_ok ? "refused" : "NOT REFUSED", dump_ok ? "refused" : "NOT REFUSED");
                if (!tight_ok || !dump_ok) ++failures;
            }
        }

        // 4. mutations the template must refuse, each with the reason it should give
        struct Mutation { const char* what; const char* want; std::function<bool(Bundle&)> apply; };
        const std::vector<Mutation> muts = {
            {"an AdamW update reads another parameter's gradient", "does not pair a parameter with its gradient", [&](Bundle& b) {
                int a0 = -1, a1 = -1;   // two updates of different parameters (a chunked update has several tasks a parameter)
                for (size_t i = 0; i < b.tasks.size() && a1 < 0; ++i) {
                    if (b.tasks[i].op_id != ops.adamw) continue;
                    if (a0 < 0) a0 = (int)i;
                    else if (b.args[b.tasks[i].args_off] != b.args[b.tasks[a0].args_off]) a1 = (int)i;
                }
                if (a0 < 0 || a1 < 0) return false;
                b.args[b.tasks[a0].args_off + 1] = b.args[b.tasks[a1].args_off + 1]; return true; }},
            {"a RoPE row block is missing (its rows go to a neighbour's range)", "RoPE does not cover", [&](Bundle& b) {
                const int t = first_of(b, ops.rope_fwd); if (t < 0) return false;
                b.args[b.tasks[t].args_off + 4] -= 1; return true; }},
            {"the q projection applies GELU", "q projection does not chain", [&](Bundle& b) {
                const int t = first_of(b, ops.rope_fwd); if (t < 1) return false;
                for (int i = t - 1; i >= 0 && b.tasks[i].op_id == ops.matmul_tile; --i) b.args[b.tasks[i].args_off + 11] |= kvm_fuse::G_GELU;
                return true; }},
            {"proj adds no residual", "attention output projection does not chain", [&](Bundle& b) {
                const int t = first_of(b, ops.attn_fwd); if (t < 0) return false;
                const int o = b.args[b.tasks[t].args_off + 3];      // the attention output: proj's tiles are the GEMM tiles that read it as their a operand
                int changed = 0;
                for (size_t i = 0; i < b.tasks.size(); ++i)
                    if (b.tasks[i].op_id == ops.matmul_tile && b.args[b.tasks[i].args_off] == o && b.args[b.tasks[i].args_off + 4] != -1) { b.args[b.tasks[i].args_off + 4] = -1; ++changed; }
                return changed > 0; }},
            {"attention reads k for v", "attention does not read the layer's q, k and v", [&](Bundle& b) {
                for (size_t i = 0; i < b.tasks.size(); ++i) if (b.tasks[i].op_id == ops.attn_fwd) b.args[b.tasks[i].args_off + 2] = b.args[b.tasks[i].args_off + 1];
                return first_of(b, ops.attn_fwd) >= 0; }},
            {"the grad-norm finish reads one partial too few", "does not read every partial", [&](Bundle& b) {
                const int t = first_of(b, ops.gradnorm_finish); if (t < 0) return false;
                b.args[b.tasks[t].args_off + 1] -= 1; return true; }},
            {"a LayerNorm backward overwrites the residual-stream gradient (acc 0)", "LayerNorm backward does not chain", [&](Bundle& b) {
                int seen_acc = -1;
                for (size_t i = 0; i < b.tasks.size(); ++i) if (b.tasks[i].op_id == ops.layernorm_bwd_dx && b.args[b.tasks[i].args_off + 9] == 1) { seen_acc = (int)i; break; }
                if (seen_acc < 0) return false;
                b.args[b.tasks[seen_acc].args_off + 9] = 0; return true; }},
        };
        for (const Mutation& m : muts) {
            Bundle b = B;
            if (!m.apply(b)) { printf("  mutation skipped (nothing to mutate): %s\n", m.what); continue; }
            const KvmFuseGptResult mr = kvm_fuse_gpt_scan(b.tasks, b.succ, b.args, b.bufs.shape, ops);
            std::string reasons;
            for (const auto& line : mr.report) if (line.find(": unchanged") != std::string::npos) reasons += line + " | ";
            const bool refused = mr.matched_runs < r.matched_runs;
            const bool right = reasons.find(m.want) != std::string::npos;
            printf("  mutation %-70s %s\n", m.what, refused && right ? "refused, with its reason" : refused ? "refused, but with another reason" : "NOT REFUSED");
            if (!refused || !right) { printf("    reasons: %s\n", reasons.substr(0, 400).c_str()); ++failures; }
        }
    }
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
