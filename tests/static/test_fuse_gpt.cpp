// CPU check of the GPT step matcher (include/kvm_fuse_gpt.h) on real bundles: prints the decision for every run, and on a matched run checks
// that its steps account for every task of the run exactly once, prints the step's shape and its parameter updates by buffer name, then
// breaks the bundle in ways the template must refuse (each mutation must leave the run unmatched, with its own reason).
//   g++ -std=c++17 -O2 -Iinclude tests/static/test_fuse_gpt.cpp -o /tmp/test_fuse_gpt && /tmp/test_fuse_gpt BUNDLE_DIR...   (from the repo root)
#include <cstdio>
#include <functional>
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
            int windows = 0, twice = 0, missed = 0;
            for (int t : comp) {
                if (B.tasks[t].op_id == ops.select_window) { ++windows; if (seen[t]) ++twice; continue; }
                if (seen[t] == 0) { if (missed++ < 3) printf("  FAIL: task %d (%s) belongs to no step\n", t, t < (int)B.names.size() ? B.names[t].c_str() : "?"); }
                else if (seen[t] > 1) ++twice;
            }
            if (missed || twice) { printf("  FAIL: %d tasks in no step, %d in two\n", missed, twice); ++failures; }
            else printf("  coverage: %zu tasks = %d data windows + the steps' %d tasks, each once\n", comp.size(), windows, (int)comp.size() - windows);

            // 2. the step's shape and its updates, by name (a reader checks them against the model)
            const KvmGptStep& s = run.shape;
            printf("  shape: L %d, d %d, heads %d x %d, MLP %d, V %d, B %d x T %d; LayerNorm eps %g, RoPE theta %g, attention scale %g\n", s.L, s.d, s.H, s.hd, s.F, s.V, s.B, s.T, s.ln_eps, s.theta, s.scale);
            printf("  AdamW lr %g, betas %g %g, eps %g; clip %g; %zu updates:\n", s.lr, s.beta1, s.beta2, s.adam_eps, s.max_norm, s.updates.size());
            for (const KvmGptUpdate& u : s.updates)
                printf("    %-24s grad %-6s m %-24s v %-24s wd %g\n", B.bufs.name[u.p].c_str(), B.bufs.name[u.g].c_str(), B.bufs.name[u.m].c_str(), B.bufs.name[u.v].c_str(), u.wd);
        }

        // 3. mutations the template must refuse, each with the reason it should give
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
