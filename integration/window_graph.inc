// Host-only window slicing. Sequential kernel launches create a global barrier
// between windows; all predecessor edges outside a window are already satisfied.
struct MvmGraphWindow {
  int first, count;
  std::vector<int> offsets, successors, indegree, initial;
};
static MvmGraphWindow mvm_slice_graph(const MvmReadyGraph& graph,
                                     int first, int count) {
  const int n = (int)graph.indegree.size();
  if (first < 0 || count <= 0 || first > n || count > n - first)
    throw std::invalid_argument("Magni VM window: invalid range");
  MvmGraphWindow out;
  out.first = first; out.count = count;
  out.offsets.push_back(0);
  out.indegree.resize(count, 0);
  for (int local = 0; local < count; ++local) {
    const int global = first + local;
    for (int e = graph.offsets[global]; e < graph.offsets[global + 1]; ++e) {
      int dst = graph.successors[e];
      if (dst < first)
        throw std::runtime_error("Magni VM window: backward edge violates tape-order windowing");
      if (dst < first + count) {
        out.successors.push_back(dst - first);
        ++out.indegree[dst - first];
      }
    }
    out.offsets.push_back((int)out.successors.size());
  }
  for (int i = 0; i < count; ++i)
    if (out.indegree[i] == 0) out.initial.push_back(i);
  return out;
}
