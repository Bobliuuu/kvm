#pragma once
// Plain-C++ readers of a bundle directory (tasks.bin, successors.bin, args.bin, metadata.json) for the host-only checks under
// tests/static: no JSON library, no CUDA. The operator registry is registry.h's, compiled from the executor's own list.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include "kvm_fuse.h"

namespace bundle_io {

inline std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); }
    std::stringstream s; s << f.rdbuf(); return s.str();
}

template <class T> std::vector<T> load(const std::string& p) {
    const std::string r = slurp(p);
    if (r.size() % sizeof(T)) { fprintf(stderr, "%s: %zu bytes is not a whole number of %zu-byte records\n", p.c_str(), r.size(), sizeof(T)); exit(2); }
    std::vector<T> v(r.size() / sizeof(T));
    memcpy(v.data(), r.data(), r.size());
    return v;
}

// The buffer table from metadata.json: every {"name": ..., "index": N, "rows": R, "cols": C, "dtype": D} entry of "buffers".
struct Buffers { std::vector<KvmFuseBuf> shape; std::vector<std::string> name; };
inline Buffers load_buffers(const std::string& meta) {
    Buffers b;
    const size_t start = meta.find("\"buffers\"");
    assert(start != std::string::npos);
    const std::regex re("\"name\"\\s*:\\s*\"([^\"]*)\"\\s*,\\s*\"index\"\\s*:\\s*(\\d+)\\s*,\\s*\"rows\"\\s*:\\s*(\\d+)\\s*,\\s*\"cols\"\\s*:\\s*(\\d+)\\s*,\\s*\"dtype\"\\s*:\\s*(\\d+)");
    for (std::sregex_iterator it(meta.begin() + start, meta.end(), re), end; it != end; ++it) {
        const int idx = std::stoi((*it)[2].str());
        if (idx >= (int)b.shape.size()) { b.shape.resize(idx + 1); b.name.resize(idx + 1); }
        const int rows = std::stoi((*it)[3].str()), cols = std::stoi((*it)[4].str());
        b.shape[idx] = KvmFuseBuf{rows, cols, cols, std::stoi((*it)[5].str())};
        b.name[idx] = (*it)[1].str();
    }
    return b;
}

// metadata.json's "task_names" (diagnostics only: the passes never read names).
inline std::vector<std::string> load_task_names(const std::string& meta) {
    std::vector<std::string> v;
    const size_t k = meta.find("\"task_names\"");
    if (k == std::string::npos) return v;
    size_t i = meta.find('[', k);
    for (; i != std::string::npos && i < meta.size() && meta[i] != ']';) {
        const size_t q0 = meta.find('"', i), q1 = q0 == std::string::npos ? q0 : meta.find('"', q0 + 1);
        const size_t close = meta.find(']', i);
        if (q0 == std::string::npos || q1 == std::string::npos || q0 > close) break;
        v.push_back(meta.substr(q0 + 1, q1 - q0 - 1));
        i = q1 + 1;
    }
    return v;
}

}  // namespace bundle_io
