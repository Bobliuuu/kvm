// kvm_run: the generic host runner. It consumes a demo directory (MANIFEST.json + compiled bundles) and runs every workload in it on ONE
// persistent kernel, with no knowledge of models or games: names, shapes, version variables, admission order, scheduler classes and
// host resources all come from the manifest and from the operator registry (include/kvm_ops*.def).
//
//   nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off \
//        demo/kvm_run.cu doom/build/doom_dev.o -o kvm_run         (without the game pack: drop -rdc, -DKVM_WITH_GAME and the .o)
//   kvm_run DEMO_DIR --out OUT [--assets DIR] [--sched off|global|static|dynamic] [--observe REGEX] [--dump REGEX] ...
//
// Manifest semantics (frontend/kvmfe/demo.py writes it, FRONTEND.md describes it; kvmfe.run_manifest is the CPU reference):
//   buffers      allocated once, initialised from their owner workload, shared by name across bundles
//   admissions   start when: time >= at_us, every after_bundles entry has completed, and each `when` window holds for the LIVE version
//                of its variable. Several may be in flight; a bundle that publishes a variable starts only when the live version equals
//                the base it was compiled for and no other in-flight bundle publishes it.
//   classes      per scheduler class (one per workload `class`) weight / reservation / cap / EDF window -> KvmClassParams
//   requirements host resources of a workload (per-thread stack, resources[{kind,file}]); `kind` is looked up in kResources below.
// A bundle compiled for other versions is not recompiled when it only needs a version shift (see rebase()); otherwise --compile-cmd
// recompiles it. Everything is checked before the GPU is touched (parse, graph cycles, a serial simulation of the version flow).
#define main kvm_exec_cli_main
#include "kvm_exec.cu"
#undef main

#include <climits>
#include <condition_variable>
#include <regex>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace run {

struct ManifestError : std::runtime_error { using std::runtime_error::runtime_error; };
[[noreturn]] static void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void fail(const char* fmt, ...) {
  char b[1024]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
  throw ManifestError(b);
}

// ---------------------------------------------------------------- strict JSON accessors (a malformed manifest is an error, never a crash)
static const Json& member(const Json& o, const char* k, const std::string& where) {
  if (o.t != Json::O) fail("%s: expected an object", where.c_str());
  const Json* j = o.get(k);
  if (!j) fail("%s: missing key '%s'", where.c_str(), k);
  return *j;
}
static long long as_int(const Json& j, const std::string& where, long long lo = LLONG_MIN / 2, long long hi = LLONG_MAX / 2) {
  if (j.t != Json::N || j.n != std::floor(j.n) || !std::isfinite(j.n)) fail("%s: expected an integer", where.c_str());
  if (j.n < (double)lo || j.n > (double)hi) fail("%s: %.17g outside [%lld, %lld]", where.c_str(), j.n, lo, hi);
  return (long long)j.n;
}
static const std::string& as_str(const Json& j, const std::string& where) {
  if (j.t != Json::S || j.s.empty()) fail("%s: expected a non-empty string", where.c_str());
  return j.s;
}
static const std::vector<Json>& as_arr(const Json& j, const std::string& where) {
  if (j.t != Json::A) fail("%s: expected an array", where.c_str());
  return j.a;
}
static bool is_null(const Json* j) { return !j || j->t == Json::Z; }
// A relative path that stays inside the demo directory.
static const std::string& as_rel_path(const Json& j, const std::string& where) {
  const std::string& s = as_str(j, where);
  if (s[0] == '/' || s.find("..") != std::string::npos) fail("%s: '%s' must be a relative path without '..'", where.c_str(), s.c_str());
  return s;
}
using VerMap = std::map<std::string, long long>;
static VerMap as_versions(const Json* j, const std::string& where) {
  VerMap m;
  if (is_null(j)) return m;
  if (j->t != Json::O) fail("%s: expected an object of variable -> version", where.c_str());
  for (auto& kv : j->o) m[kv.first] = as_int(kv.second, where + "." + kv.first, 0, INT_MAX / 2);
  return m;
}

// ---------------------------------------------------------------- manifest model
struct Cond { std::string var; long long mn = 0, mx = LLONG_MAX; };
struct Workload {
  std::string name, file, cls;
  long long tasks = 0, edges = 0;
  int priority = 0;
  std::vector<std::string> variables;
  long long stack_bytes = 0;
  struct Res { std::string kind, file; };
  std::vector<Res> resources;
};
struct Admission {
  std::string name, workload, bundle_dir, compile_workload, compile_data_dir;
  std::vector<std::string> after;
  std::vector<Cond> when;
  long long at_us = -1;                    // -1: no time condition
  VerMap base, publishes, compile_base;
  int cls = 0;
};
struct BufDecl { std::string name; int rows, cols, dtype; std::string owner; };
struct Manifest {
  std::string root;
  std::vector<Workload> workloads; std::map<std::string, int> wl_idx;
  std::vector<Admission> adms; std::map<std::string, int> adm_idx;
  std::vector<BufDecl> buffers; std::map<std::string, int> buf_idx;
  std::vector<std::string> vars; std::map<std::string, int> var_idx;
  std::vector<std::string> classes; std::vector<KvmClassParams> class_params;
  int max_priority = 0;
  int var_id(const std::string& n, const std::string& where) const {
    auto it = var_idx.find(n); if (it == var_idx.end()) fail("%s: version variable '%s' is not declared by any workload", where.c_str(), n.c_str());
    return it->second;
  }
};

static int class_id(Manifest& m, const std::string& c) {
  for (size_t i = 0; i < m.classes.size(); ++i) if (m.classes[i] == c) return (int)i;
  if (m.classes.size() >= KVM_MAX_CLS) fail("more than %d scheduler classes (adding '%s')", KVM_MAX_CLS, c.c_str());
  m.classes.push_back(c);
  KvmClassParams p{1, 0, KVM_UNLIMITED, KVM_UNLIMITED, 0, 0, 0, KVM_UNLIMITED};
  m.class_params.push_back(p);
  return (int)m.classes.size() - 1;
}

static void apply_class_param(KvmClassParams& p, const std::string& key, long long v, const std::string& where) {
  if (key == "weight") { if (v < 1) fail("%s.weight must be >= 1", where.c_str()); p.weight = (int)v; }
  else if (key == "min_cta") { if (v < 0) fail("%s.min_cta must be >= 0", where.c_str()); p.min_cta = (int)v; }
  else if (key == "max_cta") { if (v < 0) fail("%s.max_cta must be >= 0", where.c_str()); p.max_cta = (int)v; }
  else if (key == "bw_cta") { if (v < 0) fail("%s.bw_cta must be >= 0", where.c_str()); p.bw_cta = (int)v; }
  else if (key == "urgent_us") { if (v < 0) fail("%s.urgent_us must be >= 0", where.c_str()); p.urgent_us = (int)v; }
  else fail("%s: unknown scheduler parameter '%s' (weight, min_cta, max_cta, bw_cta, urgent_us)", where.c_str(), key.c_str());
}

static Manifest parse_manifest(const std::string& root) {
  Manifest m; m.root = root;
  const std::string path = root + "/MANIFEST.json";
  Json j = JsonParser(read_file(path)).parse();
  if (as_int(member(j, "manifest_version", "manifest"), "manifest_version") != 1) fail("manifest_version must be 1");

  const Json& jw = member(j, "workloads", "manifest");
  if (jw.t != Json::O || jw.o.empty()) fail("workloads: expected a non-empty object");
  if (const Json* jc = j.get("classes")) {
    if (jc->t != Json::O) fail("classes: expected an object");
    for (auto& kv : jc->o) {
      const int c = class_id(m, kv.first);
      if (kv.second.t != Json::O) fail("classes.%s: expected an object", kv.first.c_str());
      for (auto& pk : kv.second.o) apply_class_param(m.class_params[c], pk.first, as_int(pk.second, "classes." + kv.first + "." + pk.first), "classes." + kv.first);
    }
  }
  for (auto& kv : jw.o) {
    const std::string where = "workloads." + kv.first;
    Workload w; w.name = kv.first;
    w.file = as_rel_path(member(kv.second, "file", where), where + ".file");
    w.tasks = as_int(member(kv.second, "tasks", where), where + ".tasks", 1, INT_MAX);
    w.edges = as_int(member(kv.second, "edges", where), where + ".edges", 0, INT_MAX);
    w.priority = (int)as_int(member(kv.second, "priority", where), where + ".priority", 0, 1 << 20);
    w.cls = kv.second.get("class") ? as_str(*kv.second.get("class"), where + ".class") : "default";
    m.max_priority = std::max(m.max_priority, w.priority);
    if (const Json* v = kv.second.get("variables")) for (auto& x : as_arr(*v, where + ".variables")) w.variables.push_back(as_str(x, where + ".variables[]"));
    if (const Json* rq = kv.second.get("requirements")) {
      w.stack_bytes = rq->get("stack_bytes") ? as_int(*rq->get("stack_bytes"), where + ".requirements.stack_bytes", 0, 1 << 20) : 0;
      if (const Json* rs = rq->get("resources")) for (auto& r : as_arr(*rs, where + ".requirements.resources"))
        w.resources.push_back({as_str(member(r, "kind", where + ".resources[]"), where + ".resource.kind"), as_rel_path(member(r, "file", where + ".resources[]"), where + ".resource.file")});
    }
    class_id(m, w.cls);
    for (auto& v : w.variables) if (!m.var_idx.count(v)) { m.var_idx[v] = (int)m.vars.size(); m.vars.push_back(v); }
    m.wl_idx[w.name] = (int)m.workloads.size(); m.workloads.push_back(w);
  }
  if ((int)m.vars.size() > 64) fail("%zu version variables; the executor holds 64", m.vars.size());

  const Json& jb = member(j, "buffers", "manifest");
  if (jb.t != Json::O) fail("buffers: expected an object");
  for (auto& kv : jb.o) {
    const std::string where = "buffers." + kv.first;
    BufDecl b; b.name = kv.first;
    b.rows = (int)as_int(member(kv.second, "rows", where), where + ".rows", 1, INT_MAX);
    b.cols = (int)as_int(member(kv.second, "cols", where), where + ".cols", 1, INT_MAX);
    const std::string& dt = as_str(member(kv.second, "dtype", where), where + ".dtype");
    if (dt != "f32" && dt != "i32") fail("%s.dtype: '%s' is not f32 or i32", where.c_str(), dt.c_str());
    b.dtype = dt == "i32";
    b.owner = as_str(member(kv.second, "owner", where), where + ".owner");
    if (!m.wl_idx.count(b.owner)) fail("%s.owner: unknown workload '%s'", where.c_str(), b.owner.c_str());
    if ((long long)b.rows * b.cols > (1ll << 31) - 1) fail("%s: %d x %d elements overflow", where.c_str(), b.rows, b.cols);
    m.buf_idx[b.name] = (int)m.buffers.size(); m.buffers.push_back(b);
  }

  const Json& ja = member(j, "admissions", "manifest");
  const auto& arr = as_arr(ja, "admissions");
  if (arr.empty()) fail("admissions: empty");
  for (size_t i = 0; i < arr.size(); ++i) {
    const Json& x = arr[i];
    const std::string where = "admissions[" + std::to_string(i) + "]";
    Admission a;
    a.name = as_str(member(x, "name", where), where + ".name");
    if (m.adm_idx.count(a.name)) fail("%s: duplicate admission name '%s'", where.c_str(), a.name.c_str());
    a.workload = as_str(member(x, "workload", where), where + ".workload");
    auto wi = m.wl_idx.find(a.workload);
    if (wi == m.wl_idx.end()) fail("%s (%s): unknown workload '%s'", where.c_str(), a.name.c_str(), a.workload.c_str());
    a.cls = class_id(m, m.workloads[wi->second].cls);
    const std::string w2 = "admission " + a.name;
    if (const Json* v = x.get("after_bundles")) for (auto& s : as_arr(*v, w2 + ".after_bundles")) a.after.push_back(as_str(s, w2 + ".after_bundles[]"));
    if (const Json* v = x.get("when")) for (auto& c : as_arr(*v, w2 + ".when")) {
      Cond cd; cd.var = as_str(member(c, "var", w2 + ".when[]"), w2 + ".when.var");
      cd.mn = as_int(member(c, "min", w2 + ".when[]"), w2 + ".when.min", 0, INT_MAX / 2);
      if (const Json* mx = c.get("max")) { cd.mx = as_int(*mx, w2 + ".when.max", 0, INT_MAX / 2); if (cd.mx < cd.mn) fail("%s: when %s has max %lld < min %lld (can never hold)", w2.c_str(), cd.var.c_str(), cd.mx, cd.mn); }
      a.when.push_back(cd);
    }
    if (const Json* t = x.get("at_us")) if (!is_null(t)) a.at_us = as_int(*t, w2 + ".at_us", 0, 1ll << 50);
    a.base = as_versions(x.get("base_versions"), w2 + ".base_versions");
    a.publishes = as_versions(x.get("publishes"), w2 + ".publishes");
    if (const Json* bd = x.get("bundle_dir")) if (!is_null(bd)) a.bundle_dir = as_rel_path(*bd, w2 + ".bundle_dir");
    if (const Json* cp = x.get("compile")) {
      a.compile_workload = as_rel_path(member(*cp, "workload", w2 + ".compile"), w2 + ".compile.workload");
      a.compile_data_dir = cp->get("data_dir") ? as_rel_path(*cp->get("data_dir"), w2 + ".compile.data_dir") : ".";
      a.compile_base = as_versions(cp->get("base_versions"), w2 + ".compile.base_versions");
    }
    const Workload& w = m.workloads[wi->second];
    auto known = [&](const std::string& v) { return std::find(w.variables.begin(), w.variables.end(), v) != w.variables.end(); };
    for (auto& kv : a.publishes) { m.var_id(kv.first, w2 + ".publishes"); if (!known(kv.first)) fail("%s publishes '%s', which its workload does not declare", w2.c_str(), kv.first.c_str()); }
    for (auto& kv : a.base) m.var_id(kv.first, w2 + ".base_versions");
    for (auto& c : a.when) m.var_id(c.var, w2 + ".when");
    for (auto& kv : a.publishes) if (!a.base.count(kv.first)) fail("%s publishes '%s' but has no base version for it", w2.c_str(), kv.first.c_str());
    if (a.bundle_dir.empty() && a.compile_workload.empty()) fail("%s has neither bundle_dir nor compile info", w2.c_str());
    m.adm_idx[a.name] = (int)m.adms.size(); m.adms.push_back(a);
  }
  for (auto& a : m.adms) for (auto& d : a.after) {
    if (!m.adm_idx.count(d)) fail("admission %s: after_bundles names unknown admission '%s'", a.name.c_str(), d.c_str());
    if (d == a.name) fail("admission %s waits for itself", a.name.c_str());
  }
  // every buffer owner must have an admission to initialise it from
  for (auto& b : m.buffers) if (std::none_of(m.adms.begin(), m.adms.end(), [&](const Admission& a) { return a.workload == b.owner; }))
    fail("buffer %s: owner workload '%s' has no admission", b.name.c_str(), b.owner.c_str());
  return m;
}

// ---------------------------------------------------------------- static checks: dependency cycles + a serial simulation of the version flow
static bool conds_hold(const Admission& a, const std::vector<long long>& live, const Manifest& m) {
  for (auto& c : a.when) { const long long v = live[m.var_idx.at(c.var)]; if (v < c.mn || v > c.mx) return false; }
  return true;
}
// Runs admissions one at a time in the first order the conditions allow (time conditions ignored), like kvmfe.run_manifest. Anything that
// cannot finish here (cyclic after_bundles, a `when` that no sequence reaches, a bundle compiled for versions the flow never has) is
// reported with the reason instead of surfacing as a hang on the device.
static std::vector<std::string> simulate(const Manifest& m) {
  std::vector<long long> live(m.vars.size(), 0);
  std::vector<char> done(m.adms.size(), 0);
  std::vector<std::string> order;
  size_t left = m.adms.size();
  while (left) {
    int pick = -1;
    for (size_t i = 0; i < m.adms.size() && pick < 0; ++i) {
      const Admission& a = m.adms[i];
      if (done[i]) continue;
      bool ok = conds_hold(a, live, m);
      for (auto& d : a.after) ok = ok && done[m.adm_idx.at(d)];
      for (auto& kv : a.publishes) ok = ok && live[m.var_idx.at(kv.first)] == a.base.at(kv.first);   // compiled for exactly this live version
      if (ok) pick = (int)i;
    }
    if (pick < 0) {
      std::string why;
      for (size_t i = 0; i < m.adms.size(); ++i) if (!done[i]) {
        const Admission& a = m.adms[i]; why += "\n  " + a.name + ":";
        for (auto& d : a.after) if (!done[m.adm_idx.at(d)]) why += " waits for bundle " + d + ";";
        for (auto& c : a.when) { const long long v = live[m.var_idx.at(c.var)]; if (v < c.mn || v > c.mx) why += " needs " + c.var + " in [" + std::to_string(c.mn) + "," + (c.mx == LLONG_MAX ? "inf" : std::to_string(c.mx)) + "] but it reaches only " + std::to_string(v) + ";"; }
        for (auto& kv : a.publishes) { const long long v = live[m.var_idx.at(kv.first)]; if (v != a.base.at(kv.first)) why += " compiled for " + kv.first + "=" + std::to_string(a.base.at(kv.first)) + " but the flow gives " + std::to_string(v) + ";"; }
      }
      fail("the manifest cannot complete (%zu of %zu admissions run in the serial simulation): %s", order.size(), m.adms.size(), why.c_str());
    }
    for (auto& kv : m.adms[pick].publishes) live[m.var_idx.at(kv.first)] += kv.second;
    done[pick] = 1; order.push_back(m.adms[pick].name); --left;
  }
  return order;
}

// ---------------------------------------------------------------- bundles: read, shift versions, remap to the global tables
struct RawBundle {
  KvmBundle b; Json meta; std::string dir;
  std::map<int, std::string> var_name;      // local id -> name
  VerMap var_base;                          // variable name -> base version the bundle was compiled for
  std::map<int, std::string> buf_name;      // local index -> name
};
static RawBundle read_bundle(const std::string& dir, bool with_tasks = true) {
  RawBundle r; r.dir = dir;
  r.meta = JsonParser(read_file(dir + "/metadata.json")).parse();
  if (as_int(member(r.meta, "abi_version", dir), dir + ".abi_version") != KVM_ABI_VERSION) fail("%s: abi_version != %d", dir.c_str(), KVM_ABI_VERSION);
  const long long nt = as_int(member(r.meta, "task_count", dir), dir + ".task_count", 1, INT_MAX), ne = as_int(member(r.meta, "edge_count", dir), dir + ".edge_count", 0, INT_MAX),
                  nw = as_int(member(r.meta, "args_words", dir), dir + ".args_words", 0, INT_MAX);
  if (with_tasks) {
    r.b.tasks = read_bin<KvmTaskDesc>(dir + "/tasks.bin", nt);
    r.b.succ = read_bin<int32_t>(dir + "/successors.bin", ne);
    r.b.args = read_bin<int32_t>(dir + "/args.bin", nw);
  }
  for (auto& v : as_arr(member(r.meta, "variables", dir), dir + ".variables")) {
    const std::string& nm = as_str(member(v, "name", dir), dir + ".variables.name");
    r.var_name[(int)as_int(member(v, "id", dir), dir + ".variables.id", 0, 1 << 20)] = nm;
    r.var_base[nm] = as_int(member(v, "base_version", dir), dir + ".variables.base_version", 0, INT_MAX / 2);
  }
  for (auto& b : as_arr(member(r.meta, "buffers", dir), dir + ".buffers")) r.buf_name[(int)as_int(member(b, "index", dir), dir + ".buffers.index", 0, 1 << 24)] = as_str(member(b, "name", dir), dir + ".buffers.name");
  return r;
}

// Version shift: a bundle compiled with base b for a variable it publishes is the same program with every wait/publish of that variable
// moved by (target base - b). Valid only when nothing in the bundle reads that variable by absolute version (those waits must not move),
// which metadata.reads states. Returns "" on success, else why it does not apply.
static std::string rebase(RawBundle& r, const VerMap& to) {
  const VerMap from = r.var_base;
  std::map<int, long long> by_id;
  for (auto& kv : r.var_name) {
    bool published = false;
    for (auto& t : r.b.tasks) if (t.pub_var == kv.first) { published = true; break; }
    if (!published) continue;
    auto f = from.find(kv.second), t = to.find(kv.second);
    if (f == from.end() || t == to.end()) return "no base version for published variable " + kv.second;
    if (t->second == f->second) continue;
    for (auto& rd : as_arr(member(r.meta, "reads", r.dir), r.dir + ".reads"))
      if (rd.get("var") && (int)rd.get("var")->n == kv.first) return "the bundle reads " + kv.second + " by absolute version";
    by_id[kv.first] = t->second - f->second;
  }
  for (auto& kv : by_id) r.var_base[r.var_name.at(kv.first)] = to.at(r.var_name.at(kv.first));
  for (auto& t : r.b.tasks) {
    auto p = by_id.find(t.pub_var);
    if (t.pub_var >= 0 && p != by_id.end()) { t.pub_ver += (int)p->second; if (t.pub_ver < 1) return "shifted publish version below 1"; }
    auto w = by_id.find(t.wait_var);
    if (t.wait_var >= 0 && w != by_id.end()) { t.wait_ver += (int)w->second; if (t.wait_ver < 0) return "shifted wait version below 0"; }
  }
  return "";
}

// Runs the compiler (argv-style, no shell) into a scratch directory, reads the result, removes it.
static RawBundle compile_bundle(const std::vector<std::string>& cmd, const Manifest& m, const Admission& a) {
  if (cmd.empty()) fail("admission %s has no usable precompiled bundle and no --compile-cmd was given; compile it with: compile_workload.py %s <out> --data-dir %s %s",
                        a.name.c_str(), a.compile_workload.c_str(), a.compile_data_dir.c_str(), "--base-versions var=N ...");
  char tmpl[] = "/tmp/kvm_run_compile.XXXXXX";
  if (!mkdtemp(tmpl)) fail("mkdtemp: %s", strerror(errno));
  const std::string out = std::string(tmpl) + "/b";
  std::vector<std::string> argv = cmd;
  argv.push_back(m.root + "/" + a.compile_workload); argv.push_back(out);
  argv.push_back("--data-dir"); argv.push_back(m.root + "/" + a.compile_data_dir);
  for (auto& kv : a.base) { argv.push_back("--base-versions"); argv.push_back(kv.first + "=" + std::to_string(kv.second)); }
  KLOG(2, "compiling %s on demand (%s...)", a.name.c_str(), cmd[0].c_str());
  pid_t pid = fork();
  if (pid < 0) fail("fork: %s", strerror(errno));
  if (pid == 0) {
    std::vector<char*> av; for (auto& s : argv) av.push_back((char*)s.c_str()); av.push_back(nullptr);
    execvp(av[0], av.data()); _exit(127);
  }
  int st = 0; waitpid(pid, &st, 0);
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { rmdir(tmpl); fail("compiler failed for %s (status %d)", a.name.c_str(), st); }
  RawBundle r = read_bundle(out);
  std::string rm = std::string("rm -rf '") + tmpl + "'"; if (system(rm.c_str())) KLOG(1, "could not remove %s", tmpl);
  return r;
}

static void remap(RawBundle& r, const Manifest& m, int cls, bool flat_priority, const std::string& what) {
  std::map<int, int> bmap, vmap;
  for (auto& kv : r.buf_name) { auto it = m.buf_idx.find(kv.second); if (it == m.buf_idx.end()) fail("%s: buffer '%s' is not in manifest.buffers", what.c_str(), kv.second.c_str()); bmap[kv.first] = it->second; }
  for (auto& kv : r.var_name) vmap[kv.first] = m.var_id(kv.second, what);
  for (size_t i = 0; i < r.b.tasks.size(); ++i) {
    KvmTaskDesc& t = r.b.tasks[i];
    const KvmOpInfo* op = kvm_op(t.op_id);
    if (!op) fail("%s: task %zu: op id %d is not in the operator registry this runner was built with", what.c_str(), i, t.op_id);
    if (t.args_words != op->arity || t.args_off < 0 || (long long)t.args_off + t.args_words > (long long)r.b.args.size()) fail("%s: task %zu: bad argument range", what.c_str(), i);
    for (int k = 0; k < t.args_words; ++k) {
      const int ty = (int)((op->types >> (3 * k)) & 7u);
      if (ty != KVM_A_BUF && ty != KVM_A_BUF_OPT) continue;
      int32_t& w = r.b.args[t.args_off + k];
      if (w == -1 && ty == KVM_A_BUF_OPT) continue;
      auto it = bmap.find(w);
      if (it == bmap.end()) fail("%s: task %zu (%s) arg %d: local buffer index %d is not in the bundle's buffer table", what.c_str(), i, op->name, k, w);
      w = it->second;
    }
    for (int32_t* v : {&t.wait_var, &t.pub_var}) {
      if (*v == KVM_NO_VAR) continue;
      auto it = vmap.find(*v);
      if (it == vmap.end()) fail("%s: task %zu (%s): local version variable %d is not declared in the bundle", what.c_str(), i, op->name, *v);
      *v = it->second;
    }
    t.workload_id = cls;
    t.priority = flat_priority || m.max_priority <= 0 ? 0 : (m.max_priority <= KVM_NBUCKETS - 1 ? t.priority : (int)((long long)t.priority * (KVM_NBUCKETS - 1) / m.max_priority));
  }
}

// ---------------------------------------------------------------- host resources (looked up by manifest `kind`)
struct HostResource { const char* kind; bool (*init)(const std::string& path, const KvmConfig& cfg); };
static const HostResource kResources[] = {
#ifdef KVM_WITH_GAME
    {"doom_wad", [](const std::string& path, const KvmConfig& cfg) { return kvm_doom_host_init(path.c_str(), (int)cfg.stack_bytes); }},
#endif
    {nullptr, nullptr}};

// ---------------------------------------------------------------- the run
struct Options {
  std::string root, out, assets, observe, dump, compile_cmd, sched = "dynamic";
  std::vector<std::string> class_params;
  bool flat_priority = false, check_only = false;
  int stall_s = 120, blocks = 0, live_cap = 0, max_tasks = 0, timeout_s = 120;
  std::vector<double> shares;
};

enum class St { PENDING, SUBMITTING, SUBMITTED, DONE, FAILED };
struct Runtime {
  std::atomic<int> st{(int)St::PENDING};
  std::vector<unsigned> wids; int tasks = 0; std::string how, error;
  std::thread th;
  double t_launch = -1, t_submitted = -1, t_done = -1;
  std::vector<int> live_launch, live_done;
  // for the post-run version check
  struct Read { int task; std::string policy; long long version; std::string var; };
  std::vector<Read> reads; std::vector<int> op_ids; bool streamed = false;
};

static double now_us(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); }

class Runner {
 public:
  Runner(Manifest m, Options o) : m_(std::move(m)), o_(std::move(o)) {}

  int run() {
    plan();
    if (o_.check_only) { check_bundles(); KLOG(2, "check-only: manifest, bundles and version flow are consistent"); return 0; }
    mkdir(o_.out.c_str(), 0755);
    init_host_resources();
    KvmConfig cfg = make_config();
    KvmExec ex(cfg);
    ex_ = &ex;
    allocate_buffers(ex);
    ex.start();
    t0_ = std::chrono::steady_clock::now();
    KLOG(2, "started: %zu admissions, %zu classes, %zu version variables, %d buffers", m_.adms.size(), m_.classes.size(), m_.vars.size(), ex.nbufs());
    bool ok = loop(ex);
    ex.close();   // no more admissions; a submitter still blocked on a dead device returns CLOSED
    for (auto& r : rt_) if (r.th.joinable()) r.th.join();
    const KvmResult& r = ex.finish();
    ok = ok && r.ok();
    int viol = r.ok() ? check_versions(ex, r) : 0;
    ok = ok && viol == 0;
    if (r.ok()) kvm_print_class_report(r);
    kvm_write_trace_csv(r, (o_.out + "/trace.csv").c_str());   // also after a device error: it shows what ran
    dump_buffers(ex);
    write_summary(ex, r, ok, viol);
    printf("%s: %d/%d tasks, %.3f ms, %zu admissions%s\n", ok ? "OK" : "FAIL", r.done, r.admitted, r.kernel_ms, m_.adms.size(), viol ? ", VERSION VIOLATIONS" : "");
    return ok ? 0 : 1;
  }

 private:
  Manifest m_; Options o_;
  KvmExec* ex_ = nullptr;
  std::chrono::steady_clock::time_point t0_;
  std::vector<Runtime> rt_;
  std::vector<std::string> sim_order_;
  std::map<std::string, std::unique_ptr<RawBundle>> templates_;  // workload -> last loaded bundle (original local ids, pre-remap), with its base versions
  std::mutex tmu_;
  std::vector<std::string> cmd_;
  std::string errors_;
  size_t max_observe_elems_ = 1 << 16;
  std::regex observe_re_, dump_re_;

  const Workload& wl(const Admission& a) const { return m_.workloads[m_.wl_idx.at(a.workload)]; }

  // ---- everything that can be decided without a GPU
  void plan() {
    sim_order_ = simulate(m_);
    KLOG(2, "manifest ok: %zu workloads, %zu admissions, %zu buffers, serial order %s ... %s", m_.workloads.size(), m_.adms.size(), m_.buffers.size(), sim_order_.front().c_str(), sim_order_.back().c_str());
    { std::stringstream ss(o_.compile_cmd); std::string t; while (ss >> t) cmd_.push_back(t); }
    try { if (!o_.observe.empty()) observe_re_ = std::regex(o_.observe); if (!o_.dump.empty()) dump_re_ = std::regex(o_.dump); }
    catch (const std::regex_error& e) { fail("bad regex: %s", e.what()); }
    // class parameter overrides: NAME.key=value
    for (auto& cp : o_.class_params) {
      const size_t dot = cp.find('.'), eq = cp.find('=');
      if (dot == std::string::npos || eq == std::string::npos || dot > eq) fail("--class-param '%s': expected CLASS.key=value", cp.c_str());
      const std::string c = cp.substr(0, dot);
      auto it = std::find(m_.classes.begin(), m_.classes.end(), c);
      if (it == m_.classes.end()) fail("--class-param '%s': no class '%s' (have: %s)", cp.c_str(), c.c_str(), join(m_.classes).c_str());
      char* e; const std::string v = cp.substr(eq + 1); long long x = strtoll(v.c_str(), &e, 10);
      if (v.empty() || *e) fail("--class-param '%s': value is not an integer", cp.c_str());
      apply_class_param(m_.class_params[it - m_.classes.begin()], cp.substr(dot + 1, eq - dot - 1), x, "--class-param " + c);
    }
    if (!o_.shares.empty() && o_.shares.size() != m_.classes.size()) fail("--shares has %zu entries for %zu classes", o_.shares.size(), m_.classes.size());
    // every precompiled bundle must at least exist and parse (so a missing directory fails now, not mid-run)
    for (auto& a : m_.adms) if (!a.bundle_dir.empty()) {
      struct stat sb; const std::string d = m_.root + "/" + a.bundle_dir;
      if (stat((d + "/metadata.json").c_str(), &sb) != 0) { KLOG(1, "admission %s: bundle_dir %s is missing; it will be derived or compiled", a.name.c_str(), a.bundle_dir.c_str()); }
    }
  }
  static std::string join(const std::vector<std::string>& v) { std::string s; for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + v[i]; return s; }

  // Every admission's bundle must load (or be derivable), map onto the manifest's tables and pass the executor's structural validation.
  void check_bundles() {
    int n = 0;
    for (auto& a : m_.adms) {
      std::string how;
      RawBundle r = obtain(a, &how);
      remap(r, m_, a.cls, o_.flat_priority, "admission " + a.name);
      const std::string why = r.b.validate((int)m_.buffers.size(), 64, 0);
      if (!why.empty()) fail("admission %s: %s", a.name.c_str(), why.c_str());
      ++n;
      KLOG(3, "bundle %-14s %6zu tasks (%s)", a.name.c_str(), r.b.tasks.size(), how.c_str());
    }
    KLOG(2, "check-only: %d bundles load, remap and validate", n);
  }

  void init_host_resources() {
    for (auto& w : m_.workloads) {
      if (std::none_of(m_.adms.begin(), m_.adms.end(), [&](const Admission& a) { return a.workload == w.name; })) continue;
      for (auto& r : w.resources) {
        const HostResource* h = kResources; while (h->kind && r.kind != h->kind) ++h;
        if (!h->kind) fail("workload %s needs host resource '%s', which this runner build does not provide (build with -DKVM_WITH_GAME for the game pack)", w.name.c_str(), r.kind.c_str());
        const std::string path = (o_.assets.empty() ? m_.root : o_.assets) + "/" + r.file;
        KvmConfig c; c.stack_bytes = stack_bytes();
        KLOG(2, "host resource %s for workload %s: %s (stack %zu B)", r.kind.c_str(), w.name.c_str(), path.c_str(), c.stack_bytes);
        if (!h->init(path, c)) fail("host resource %s (%s) failed to initialise", r.kind.c_str(), path.c_str());
      }
    }
  }
  size_t stack_bytes() const { long long s = 0; for (auto& w : m_.workloads) s = std::max(s, w.stack_bytes); return (size_t)s; }

  KvmConfig make_config() {
    KvmConfig cfg;
    long long maxw = 1, maxwords = 0, total = 0;
    for (auto& w : m_.workloads) { maxw = std::max(maxw, w.tasks); maxwords = std::max(maxwords, 28 * w.tasks + w.edges + 64); }
    for (auto& a : m_.adms) total += wl(a).tasks;
    auto pow2 = [](long long v, long long lo, long long hi) { long long p = lo; while (p < v && p < hi) p <<= 1; return p; };
    cfg.live_cap = o_.live_cap ? o_.live_cap : (int)pow2(maxw, 1 << 12, 1 << 16);
    cfg.ring_words = (int)std::min<long long>(pow2(2 * maxwords, 1 << 16, 1 << 22), 1 << 22);
    cfg.max_tasks = o_.max_tasks ? o_.max_tasks : (int)std::max<long long>(pow2(total + cfg.live_cap, 1 << 15, 1 << 21), 2 * cfg.live_cap);
    cfg.stack_bytes = stack_bytes();
    cfg.blocks = o_.blocks;
    cfg.timeout_us = (unsigned long long)o_.timeout_s * 1000000ull;
    if (o_.sched != "off") {
      if (o_.sched != "global" && o_.sched != "static" && o_.sched != "dynamic") fail("--sched %s: expected off|global|static|dynamic", o_.sched.c_str());
      cfg.sched.on = true;
      cfg.sched.mode = o_.sched == "global" ? KVM_SCHED_GLOBAL : o_.sched == "static" ? KVM_SCHED_STATIC : KVM_SCHED_DYNAMIC;
      cfg.sched.num_classes = (int)m_.classes.size();
      for (size_t c = 0; c < m_.classes.size(); ++c) cfg.sched.p.cls[c] = m_.class_params[c];
      if (cfg.sched.mode == KVM_SCHED_STATIC) {
        if (o_.shares.empty()) fail("--sched static needs --shares a,b,... (one per class: %s)", join(m_.classes).c_str());
        cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); kvm_static_split(cfg.sched.p, o_.shares, pr.multiProcessorCount);
      }
    }
    KLOG(2, "config: live_cap %d, slots %d, ring %d words, stack %zu B, sched %s, classes [%s]", cfg.live_cap, cfg.max_tasks, cfg.ring_words, cfg.stack_bytes, o_.sched.c_str(), join(m_.classes).c_str());
    for (size_t c = 0; c < m_.classes.size(); ++c) { const auto& p = m_.class_params[c]; KLOG(2, "  class %zu %-8s weight %d min_cta %d max_cta %d urgent_us %d", c, m_.classes[c].c_str(), p.weight, p.min_cta, p.max_cta, p.urgent_us); }
    return cfg;
  }

  // Buffers are fixed at launch: initialise each from the first bundle of its owner workload.
  void allocate_buffers(KvmExec& ex) {
    std::map<std::string, std::unique_ptr<RawBundle>> owner_meta;
    for (auto& b : m_.buffers) {
      auto& rb = owner_meta[b.owner];
      if (!rb) {
        const Admission& a = *std::find_if(m_.adms.begin(), m_.adms.end(), [&](const Admission& x) { return x.workload == b.owner; });
        rb = std::make_unique<RawBundle>(obtain(a, nullptr, false));
      }
      const Json* decl = nullptr;
      for (auto& jb : rb->meta.need("buffers").a) if (jb.get("name") && jb.get("name")->s == b.name) { decl = &jb; break; }
      if (!decl) fail("buffer %s: its owner workload %s has no such buffer in its bundle", b.name.c_str(), b.owner.c_str());
      const int rows = (int)decl->num("rows", 0), cols = (int)decl->num("cols", 0), dtype = (int)decl->num("dtype", 0);
      if (rows != b.rows || cols != b.cols || dtype != b.dtype) fail("buffer %s: manifest says %dx%d dtype %d, owner bundle %dx%d dtype %d", b.name.c_str(), b.rows, b.cols, b.dtype, rows, cols, dtype);
      const int idx = ex.add_buffer(rows, cols, kvm_parse_init(*decl, rb->dir, dtype), dtype);
      if (idx != m_.buf_idx.at(b.name)) throw std::logic_error("buffer index mismatch");
    }
  }

  // The raw (local-id) bundle of an admission: precompiled (shifted if compiled for other versions), else shifted from the last bundle of
  // the same workload, else compiled. `with_tasks=false` reads only the metadata (buffer table, inits).
  RawBundle obtain(const Admission& a, std::string* how, bool with_tasks = true) {
    std::string h;
    RawBundle r;
    bool have = false;
    auto try_shift = [&](RawBundle& t, const char* what) {
      if (!with_tasks) return true;
      bool needs = false;
      for (auto& kv : a.publishes) { auto b = t.var_base.find(kv.first); needs = needs || b == t.var_base.end() || b->second != a.base.at(kv.first); }
      if (!needs) return true;
      const std::string why = rebase(t, a.base);
      if (why.empty()) { h = std::string(what) + ", versions shifted"; return true; }
      KLOG(2, "admission %s: version shift of the %s bundle not applicable (%s)", a.name.c_str(), what, why.c_str());
      return false;
    };
    if (!a.bundle_dir.empty()) {
      struct stat sb;
      const std::string d = m_.root + "/" + a.bundle_dir;
      if (stat((d + "/metadata.json").c_str(), &sb) == 0) {
        RawBundle t = read_bundle(d, with_tasks);
        h = "precompiled";
        if (try_shift(t, "precompiled")) { r = std::move(t); have = true; }
      }
    }
    if (!have) {
      std::lock_guard<std::mutex> lk(tmu_);
      auto it = templates_.find(a.workload);
      if (it != templates_.end()) {
        RawBundle t = *it->second;
        if (try_shift(t, "previous")) { r = std::move(t); have = true; if (h.empty() || h == "precompiled") h = "shifted from a previous bundle of the workload"; }
      }
    }
    if (!have) { r = compile_bundle(cmd_, m_, a); h = "compiled on demand"; }
    if (how) *how = h;
    if (with_tasks) {
      std::lock_guard<std::mutex> lk(tmu_);
      if (!templates_.count(a.workload) || !a.publishes.empty()) templates_[a.workload] = std::make_unique<RawBundle>(r);
    }
    return r;
  }

  void load_and_submit(int ai, Runtime* rt) {
    const Admission& a = m_.adms[ai];
    try {
      std::string how;
      RawBundle r = obtain(a, &how);
      for (const Json& rd : r.meta.need("reads").a) {   // checked after the run against the trace
        const Json* g = rd.get("gate");
        Runtime::Read x; x.task = g && g->t == Json::N ? (int)g->n : (int)rd.num("task", -1);
        x.policy = rd.need("policy").s; x.version = (long long)rd.num("version", 0);
        auto vn = r.var_name.find((int)rd.num("var", -1));
        x.var = (vn == r.var_name.end() ? std::string("?") : vn->second) + " @ " + rd.need("name").s;
        rt->reads.push_back(x);
      }
      remap(r, m_, a.cls, o_.flat_priority, "admission " + a.name);
      for (auto& t : r.b.tasks) rt->op_ids.push_back(t.op_id);
      rt->how = how; rt->tasks = (int)r.b.tasks.size();
      std::string why = r.b.validate(ex_->nbufs(), 64, 0);
      if (!why.empty()) fail("admission %s: %s", a.name.c_str(), why.c_str());
      const bool stream = r.b.tasks.size() > (size_t)ex_->config().live_cap || r.b.words() > (size_t)ex_->config().ring_words / 2;
      rt->streamed = stream;
      KvmSubmit s;
      if (stream) s = ex_->submit_stream(r.b, 0, 600000, &rt->wids);
      else { unsigned w = 0; s = ex_->submit(r.b, 60000, true, &w); rt->wids.push_back(w); }
      if (s != KvmSubmit::OK) fail("admission %s: submit failed: %s (%s)", a.name.c_str(), kvm_submit_name(s), ex_->last_error().c_str());
      rt->t_submitted = now_us(t0_);
      rt->st = (int)St::SUBMITTED;
    } catch (const std::exception& e) {
      rt->error = e.what(); rt->st = (int)St::FAILED;
      KLOG(0, "admission %s failed: %s", a.name.c_str(), e.what());
    }
  }

  bool all_windows_done(const Runtime& r) const {
    for (unsigned w : r.wids) if (!ex_->window_done(w)) return false;
    return true;
  }

  std::string why_not(int ai, const std::vector<int>& live) const {
    const Admission& a = m_.adms[ai]; std::string s;
    if (a.at_us >= 0 && now_us(t0_) < (double)a.at_us) s += " time;";
    for (auto& d : a.after) if (rt_[m_.adm_idx.at(d)].st != (int)St::DONE) s += " after " + d + ";";
    for (auto& c : a.when) { const int v = live[m_.var_idx.at(c.var)]; if (v < c.mn || v > c.mx) s += " " + c.var + "=" + std::to_string(v) + " not in [" + std::to_string(c.mn) + "," + (c.mx == LLONG_MAX ? "inf" : std::to_string(c.mx)) + "];"; }
    for (auto& kv : a.publishes) { const int v = live[m_.var_idx.at(kv.first)]; if (v != a.base.at(kv.first)) s += " " + kv.first + " live " + std::to_string(v) + " vs compiled " + std::to_string(a.base.at(kv.first)) + ";"; }
    return s;
  }

  bool loop(KvmExec& ex) {
    rt_ = std::vector<Runtime>(m_.adms.size());
    size_t done = 0, failed = 0, inflight = 0;
    long long progress_mark = -1; auto progress_t = std::chrono::steady_clock::now();
    std::vector<int> live;
    bool ok = true;
    while (done < m_.adms.size()) {
      if (ex.device_error()) { KLOG(0, "device error %d: stopping admissions", ex.device_error()); ok = false; break; }
      live = ex.live_versions();
      // completions first, so that observed buffers are read before a successor that overwrites them is admitted
      for (size_t i = 0; i < m_.adms.size(); ++i) {
        Runtime& r = rt_[i];
        const int st = r.st.load();
        if (st == (int)St::SUBMITTED && all_windows_done(r)) {
          r.t_done = now_us(t0_); r.live_done = live; r.st = (int)St::DONE; ++done; --inflight;
          KLOG(2, "[%9.0f us] done      %-14s %6d tasks, versions %s", r.t_done, m_.adms[i].name.c_str(), r.tasks, vers(live).c_str());
          observe(ex, (int)i);
        } else if (st == (int)St::FAILED && r.th.joinable()) {
          r.th.join(); ++failed; ++done; --inflight; ok = false; errors_ += m_.adms[i].name + ": " + r.error + "\n";
        }
      }
      if (!ok) break;
      for (size_t i = 0; i < m_.adms.size(); ++i) {
        Runtime& r = rt_[i];
        if (r.st.load() != (int)St::PENDING) continue;
        const Admission& a = m_.adms[i];
        bool elig = a.at_us < 0 || now_us(t0_) >= (double)a.at_us;
        for (auto& d : a.after) elig = elig && rt_[m_.adm_idx.at(d)].st.load() == (int)St::DONE;
        for (auto& c : a.when) {
          const int v = live[m_.var_idx.at(c.var)];
          if (v > c.mx) { errors_ += a.name + ": version window passed (" + c.var + "=" + std::to_string(v) + " > max " + std::to_string(c.mx) + ")\n"; KLOG(0, "%s", errors_.c_str()); ok = false; }
          elig = elig && v >= c.mn && v <= c.mx;
        }
        for (auto& kv : a.publishes) {
          const int v = live[m_.var_idx.at(kv.first)], base = (int)a.base.at(kv.first);
          if (v > base) { errors_ += a.name + ": compiled for " + kv.first + "=" + std::to_string(base) + " but the live version is already " + std::to_string(v) + "\n"; KLOG(0, "%s", errors_.c_str()); ok = false; }
          elig = elig && v == base;
          for (size_t j = 0; j < m_.adms.size() && elig; ++j) if (j != i && rt_[j].st.load() != (int)St::PENDING && rt_[j].st.load() != (int)St::DONE && m_.adms[j].publishes.count(kv.first)) elig = false;   // one publisher of a variable in flight
        }
        if (!ok) break;
        if (!elig) continue;
        r.t_launch = now_us(t0_); r.live_launch = live; r.st = (int)St::SUBMITTING; ++inflight;
        KLOG(2, "[%9.0f us] admit     %-14s (class %s) versions %s", r.t_launch, a.name.c_str(), m_.classes[a.cls].c_str(), vers(live).c_str());
        r.th = std::thread([this, i, &r] { load_and_submit((int)i, &r); });
        live = ex.live_versions();
      }
      if (!ok) break;
      // progress / stall watchdog: the sum of live versions moves while anything runs; completions move `done`
      long long mark = (long long)done * 1000003ll; for (int v : live) mark += v;
      if (mark != progress_mark) { progress_mark = mark; progress_t = std::chrono::steady_clock::now(); }
      else if (std::chrono::steady_clock::now() - progress_t > std::chrono::seconds(o_.stall_s)) {
        std::string why;
        for (size_t i = 0; i < m_.adms.size(); ++i) if (rt_[i].st.load() != (int)St::DONE) why += "\n  " + m_.adms[i].name + (rt_[i].st.load() == (int)St::PENDING ? " pending:" + why_not((int)i, live) : " in flight");
        errors_ += "no progress for " + std::to_string(o_.stall_s) + " s:" + why + "\n"; KLOG(0, "%s", errors_.c_str()); ok = false; break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    (void)failed;
    return ok;
  }
  static std::string vers(const std::vector<int>& v) { std::string s = "["; for (size_t i = 0; i < v.size() && i < 8; ++i) s += (i ? "," : "") + std::to_string(v[i]); return s + "]"; }

  // Reads (small) buffers of a finished admission that match --observe: values that later admissions overwrite (e.g. per-window results).
  void observe(KvmExec& ex, int ai) {
    if (o_.observe.empty()) return;
    // the bundle's own buffer table
    std::vector<std::string> names;
    {
      std::lock_guard<std::mutex> lk(tmu_);
      auto it = templates_.find(m_.adms[ai].workload);
      if (it == templates_.end()) return;
      for (auto& kv : it->second->buf_name) names.push_back(kv.second);
    }
    FILE* f = fopen((o_.out + "/observations.jsonl").c_str(), "a");
    if (!f) { KLOG(1, "cannot open observations.jsonl"); return; }
    fprintf(f, "{\"adm\":\"%s\",\"t_done_us\":%.0f,\"buffers\":{", m_.adms[ai].name.c_str(), rt_[ai].t_done);
    bool first = true;
    for (auto& n : names) {
      if (!std::regex_search(n, observe_re_)) continue;
      const int idx = m_.buf_idx.at(n);
      const BufDecl& b = m_.buffers[idx];
      if ((size_t)b.rows * b.cols > max_observe_elems_) { KLOG(1, "observe: %s has %d elements, skipped", n.c_str(), b.rows * b.cols); continue; }
      std::vector<float> v = ex.read_buffer(idx);
      fprintf(f, "%s\"%s\":{\"dtype\":\"%s\",\"values\":[", first ? "" : ",", n.c_str(), b.dtype ? "i32" : "f32"); first = false;
      for (size_t i = 0; i < v.size(); ++i) { if (b.dtype) { int x; memcpy(&x, &v[i], 4); fprintf(f, "%s%d", i ? "," : "", x); } else fprintf(f, "%s%.9g", i ? "," : "", v[i]); }
      fprintf(f, "]}");
    }
    fprintf(f, "}}\n"); fclose(f);
  }

  void dump_buffers(KvmExec& ex) {
    if (o_.dump.empty()) return;
    const std::string dir = o_.out + "/buffers"; mkdir(dir.c_str(), 0755);
    FILE* idx = fopen((dir + "/index.json").c_str(), "w");
    if (!idx) { KLOG(1, "cannot write buffer index"); return; }
    fprintf(idx, "{"); bool first = true; int n = 0;
    for (size_t i = 0; i < m_.buffers.size(); ++i) {
      const BufDecl& b = m_.buffers[i];
      if (!std::regex_search(b.name, dump_re_)) continue;
      std::vector<float> v = ex.read_buffer((int)i);
      FILE* f = fopen((dir + "/" + b.name + ".bin").c_str(), "wb");
      if (!f) { KLOG(1, "cannot write %s", b.name.c_str()); continue; }
      fwrite(v.data(), 4, v.size(), f); fclose(f);
      fprintf(idx, "%s\"%s\":{\"rows\":%d,\"cols\":%d,\"dtype\":\"%s\"}", first ? "" : ",", b.name.c_str(), b.rows, b.cols, b.dtype ? "i32" : "f32"); first = false; ++n;
    }
    fprintf(idx, "}\n"); fclose(idx);
    KLOG(2, "dumped %d buffers to %s", n, dir.c_str());
  }

  // Version-tag check from the executor's own trace: every task that carries a version wait recorded the variable's version at start
  // and at end. exact: both equal the requested version (the weights did not move during the read); at_least: start >= version.
  int check_versions(KvmExec& ex, const KvmResult& r) {
    const std::vector<int> rec = ex.record_tasks();
    std::vector<long long> base(rec.size() + 1, 0);
    for (size_t i = 0; i < rec.size(); ++i) base[i + 1] = base[i] + rec[i];
    int viol = 0, checked = 0, exact = 0, advanced = 0, mapping_bad = 0, v0_skipped = 0;
    const bool recycled = base.back() > (long long)ex.config().max_tasks;
    if (recycled) { KLOG(1, "version check: %lld tasks > %d slots, slots were recycled; trace covers only the last occupants: skipped", base.back(), ex.config().max_tasks); version_checked_ = -1; return 0; }
    for (size_t ai = 0; ai < m_.adms.size(); ++ai) {
      const Runtime& rt = rt_[ai];
      if (rt.st.load() != (int)St::DONE || rt.reads.empty()) continue;
      if (rt.streamed || rt.wids.size() != 1) { KLOG(1, "version check: %s was streamed; skipped", m_.adms[ai].name.c_str()); continue; }
      const long long b0 = base[rt.wids[0]];
      for (size_t t = 0; t < rt.op_ids.size() && t < 4; ++t) if (r.tasks[b0 + t].op_id != rt.op_ids[t]) ++mapping_bad;
      for (auto& rd : rt.reads) {
        if (rd.task < 0 || rd.task >= (int)rt.op_ids.size()) continue;
        const KvmTrace& tr = r.trace[b0 + rd.task];
        if (rd.version == 0 && tr.ver0 < 0) { ++v0_skipped; continue; }   // version 0 is the initial state: the compiler emits no wait, so the trace has no tag
        if (r.tasks[b0 + rd.task].op_id != rt.op_ids[rd.task]) { ++mapping_bad; continue; }
        bool ok = tr.ver0 >= rd.version;
        if (rd.policy == "exact") { ok = tr.ver0 == rd.version && tr.ver1 == rd.version; ++exact; }
        else if (tr.ver1 > tr.ver0) ++advanced;
        ++checked;
        if (!ok) { ++viol; KLOG(0, "VERSION VIOLATION %s: %s wants %s v%lld, saw start %d end %d", m_.adms[ai].name.c_str(), rd.var.c_str(), rd.policy.c_str(), rd.version, tr.ver0, tr.ver1); }
      }
    }
    if (mapping_bad) { KLOG(0, "version check: %d slot/op mismatches: the trace-to-record mapping is not trustworthy", mapping_bad); viol += mapping_bad; }
    KLOG(2, "version check: %d version-carrying tasks (%d exact), %d violations, %d at_least readers saw the version advance while they ran; %d reads of version 0 carry no tag (ordered by the host barrier only)", checked, exact, viol, advanced, v0_skipped);
    version_checked_ = checked; version_exact_ = exact; version_advanced_ = advanced; version_viol_ = viol;
    return viol;
  }
  int version_checked_ = 0, version_exact_ = 0, version_advanced_ = 0, version_viol_ = 0;

  void write_summary(KvmExec& ex, const KvmResult& r, bool ok, int viol) {
    FILE* f = fopen((o_.out + "/run.json").c_str(), "w");
    if (!f) { KLOG(1, "cannot write run.json"); return; }
    fprintf(f, "{\"ok\":%s,\"device\":\"%s\",\"sms\":%d,\"grid\":%d,\"kernel_ms\":%.3f,\"sched\":\"%s\",\"flat_priority\":%s,\"tasks_admitted\":%d,\"tasks_done\":%d,\"slots\":%d,\"err_code\":%d,",
            ok ? "true" : "false", ex.device_name().c_str(), ex.sms(), r.grid, r.kernel_ms, o_.sched.c_str(), o_.flat_priority ? "true" : "false", r.admitted, r.done, ex.config().max_tasks, r.err_code);
    fprintf(f, "\"version_check\":{\"checked\":%d,\"exact\":%d,\"violations\":%d,\"at_least_advanced_during_read\":%d},", version_checked_, version_exact_, viol, version_advanced_);
    fprintf(f, "\"classes\":[");
    for (size_t c = 0; c < m_.classes.size(); ++c) fprintf(f, "%s\"%s\"", c ? "," : "", m_.classes[c].c_str());
    fprintf(f, "],\"errors\":\"");
    for (char ch : errors_) { if (ch == '"' || ch == '\\') fputc('\\', f); fputc(ch == '\n' ? ' ' : ch, f); }
    fprintf(f, "\",\"final_versions\":{");
    for (size_t v = 0; v < m_.vars.size(); ++v) fprintf(f, "%s\"%s\":%d", v ? "," : "", m_.vars[v].c_str(), r.version.empty() ? -1 : r.version[v]);
    fprintf(f, "},\"admissions\":[");
    const std::vector<int> rec = ex.record_tasks();
    for (size_t i = 0; i < m_.adms.size(); ++i) {
      const Runtime& rt = rt_[i]; const Admission& a = m_.adms[i];
      fprintf(f, "%s{\"name\":\"%s\",\"workload\":\"%s\",\"class\":\"%s\",\"state\":%d,\"tasks\":%d,\"how\":\"%s\",\"t_launch_us\":%.0f,\"t_submitted_us\":%.0f,\"t_done_us\":%.0f,\"wids\":[",
              i ? "," : "", a.name.c_str(), a.workload.c_str(), m_.classes[a.cls].c_str(), rt.st.load(), rt.tasks, rt.how.c_str(), rt.t_launch, rt.t_submitted, rt.t_done);
      for (size_t k = 0; k < rt.wids.size(); ++k) fprintf(f, "%s%u", k ? "," : "", rt.wids[k]);
      fprintf(f, "]}");
    }
    fprintf(f, "],\"record_tasks\":[");
    for (size_t k = 0; k < rec.size(); ++k) fprintf(f, "%s%d", k ? "," : "", rec[k]);
    fprintf(f, "]}\n"); fclose(f);
    KLOG(2, "wrote %s/run.json", o_.out.c_str());
  }
};

}  // namespace run

static void run_usage() {
  fprintf(stderr,
    "usage: kvm_run DEMO_DIR --out DIR [options]\n"
    "  DEMO_DIR            holds MANIFEST.json and the bundles it names\n"
    "  --out DIR           run.json, trace.csv, observations.jsonl, buffers/\n"
    "  --assets DIR        where manifest host resources (e.g. the game WAD) are looked up (default DEMO_DIR)\n"
    "  --sched MODE        off|global|static|dynamic (default dynamic: one class per manifest workload class)   --shares a,b,..  (static)\n"
    "  --class-param C.k=v override a manifest class parameter (weight, min_cta, max_cta, bw_cta, urgent_us)\n"
    "  --flat-priority     ignore task priorities (all 0): only the scheduler classes order work\n"
    "  --observe REGEX     after each admission completes, record its matching small buffers to observations.jsonl\n"
    "  --dump REGEX        at the end write matching buffers to DIR/buffers/NAME.bin (raw 4-byte elements) + index.json\n"
    "  --compile-cmd CMD   compiler to run for admissions with no usable bundle (e.g. 'python3 runtime/compile_workload.py')\n"
    "  --blocks N --live-cap N --max-tasks N --timeout-s S (device stall) --stall-s S (host: no version/completion progress)\n"
    "  --check-only        validate the manifest, the version flow and the bundles, then exit (no GPU needed)\n");
}

int main(int argc, char** argv) {
  run::Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto sval = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); run_usage(); exit(2); } return argv[++i]; };
    auto ival = [&](long long lo, long long hi) -> int {
      std::string s = sval(); char* e; errno = 0; long long v = strtoll(s.c_str(), &e, 10);
      if (s.empty() || *e || errno || v < lo || v > hi) { fprintf(stderr, "%s: integer in [%lld,%lld] expected, got '%s'\n", a.c_str(), lo, hi, s.c_str()); exit(2); }
      return (int)v; };
    if (a == "--out") o.out = sval();
    else if (a == "--assets") o.assets = sval();
    else if (a == "--sched") o.sched = sval();
    else if (a == "--shares") { std::stringstream ss(sval()); std::string t; while (std::getline(ss, t, ',')) { char* e; double v = strtod(t.c_str(), &e); if (t.empty() || *e || !(v > 0)) { fprintf(stderr, "--shares: positive numbers expected\n"); return 2; } o.shares.push_back(v); } }
    else if (a == "--class-param") o.class_params.push_back(sval());
    else if (a == "--flat-priority") o.flat_priority = true;
    else if (a == "--observe") o.observe = sval();
    else if (a == "--dump") o.dump = sval();
    else if (a == "--compile-cmd") o.compile_cmd = sval();
    else if (a == "--blocks") o.blocks = ival(0, 1 << 20);
    else if (a == "--live-cap") o.live_cap = ival(2, 1 << 24);
    else if (a == "--max-tasks") o.max_tasks = ival(2, 1 << 26);
    else if (a == "--timeout-s") o.timeout_s = ival(1, 86400);
    else if (a == "--stall-s") o.stall_s = ival(1, 86400);
    else if (a == "--check-only") o.check_only = true;
    else if (a == "--quiet") g_kvm_log = 0;
    else if (a[0] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); run_usage(); return 2; }
    else if (o.root.empty()) o.root = a;
    else { fprintf(stderr, "unexpected argument %s\n", a.c_str()); run_usage(); return 2; }
  }
  if (o.root.empty() || (o.out.empty() && !o.check_only)) { run_usage(); return 2; }
  try {
    run::Runner r(run::parse_manifest(o.root), o);
    return r.run();
  } catch (const run::ManifestError& e) {
    KLOG(0, "manifest/plan error: %s", e.what());
    return 3;
  } catch (const std::exception& e) {
    KLOG(0, "%s", e.what());
    return 1;
  }
}
