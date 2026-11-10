#pragma once
// The operator registry the executor compiles, for the static path's CPU tests (g++ -Iinclude): include/kvm_ops_list.inc expanded
// here at compile time, as kvm_exec.cuh expands it, so a test sees exactly the ids and signatures of the tree it was built from,
// wherever it runs. Reading the .def files at run time instead, by a list of names relative to the working directory, let a run
// from another directory load no operator and still pass, and let the list drift from the executor's (it had left out
// kvm_ops_fmlp.def).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include "kvm_static.h"

namespace kvm_test_registry {

struct Op { const char* name; int id; const char* sig; };
inline constexpr Op kOps[] = {
#define X(name, id, sig) {#name, id, sig},
#include "kvm_ops_list.inc"
#undef X
};

[[noreturn]] inline void fail(const std::string& why) { fprintf(stderr, "registry: %s\n", why.c_str()); exit(2); }

// The planner's view of a signature: its arity, each argument's 3-bit type (buf 0, i32 1, f32 2, buf? 3) and the gang arguments.
inline KvmSOpInfo info(const char* sig) {
    KvmSOpInfo o; o.arity = 0; o.types = 0;
    for (const char* p = sig; *p;) {
        const char* end = strchr(p, ','); if (!end) end = p + strlen(p);
        const char* colon = strchr(p, ':');
        if (!colon || colon > end) fail(std::string("malformed signature \"") + sig + "\"");
        const std::string ty(colon + 1, end);
        const int t = ty == "buf" ? 0 : ty == "i32" ? 1 : ty == "f32" ? 2 : ty == "buf?" ? 3 : -1;
        if (t < 0) fail("unknown argument type \"" + ty + "\" in \"" + sig + "\"");
        o.types |= (uint64_t)t << (3 * o.arity++);
        p = *end ? end + 1 : end;
    }
    kvm_static_gang_args(sig, o);
    return o;
}

struct Registry {
    std::map<int, KvmSOpInfo> ops; std::map<std::string, int> ids;
    int id(const char* name) const { const auto it = ids.find(name); return it == ids.end() ? -1 : it->second; }
    // The id of an operator the test cannot run without: a tree that lacks it fails the test here, never later and silently.
    int require(const char* name) const { const int i = id(name); if (i < 0) fail(std::string("no operator ") + name + " in kvm_ops_list.inc's registries"); return i; }
    KvmSOpLookup lookup() const { return [this](int i) { const auto it = ops.find(i); return it == ops.end() ? KvmSOpInfo{} : it->second; }; }
};

inline const Registry& registry() {
    static const Registry r = [] {
        Registry r;
        for (const Op& o : kOps) {
            if (!r.ids.emplace(o.name, o.id).second || r.ops.count(o.id)) fail(std::string("operator ") + o.name + " (" + std::to_string(o.id) + ") declared twice");
            r.ops[o.id] = info(o.sig);
        }
        if (r.ops.empty()) fail("kvm_ops_list.inc lists no operator");
        return r;
    }();
    return r;
}

}  // namespace kvm_test_registry
