// Instantiate the one-step tape exactly as inst_at does on the device,
// then lower full-barrier waits into CSR predecessors/successors.
// Reject partial counter thresholds instead of silently changing semantics.
struct KvmReadyGraph {
  std::vector<int> offsets, successors, indegree, initial;
};
static KvmReadyGraph kvm_compile_ready_graph(
    const std::vector<Instr>& tape, int steps, int SC) {
  const int perStep = (int)tape.size();
  const int n = perStep * steps;
  std::vector<Instr> expanded(n);
  std::vector<std::vector<int>> producers((size_t)steps * SC);
  for (int s = 0; s < steps; ++s) for (int t = 0; t < perStep; ++t) {
    int id = s * perStep + t;
    Instr in = tape[t];
    if (in.xflags & 1) in.w0c = s ? in.w0c + (s - 1) * SC : -1;
    else if (in.w0c >= 0) in.w0c += s * SC;
    if (in.xflags & 2) in.w1c = s ? in.w1c + (s - 1) * SC : -1;
    else if (in.w1c >= 0) in.w1c += s * SC;
    if (in.s0 >= 0) in.s0 += s * SC;
    if (in.s1 >= 0) in.s1 += s * SC;
    expanded[id] = in;
    if (in.s0 >= 0) producers.at(in.s0).push_back(id);
    if (in.s1 >= 0) producers.at(in.s1).push_back(id);
  }
  std::vector<std::vector<int>> edges(n);
  std::vector<int> indegree(n, 0);
  for (int id = 0; id < n; ++id) {
    const Instr& in = expanded[id];
    std::vector<int> pred;
    auto append = [&](int counter, int threshold) {
      if (counter < 0) return;
      const auto& p = producers.at(counter);
      // Only full barriers have static DAG semantics independent of
      // producer completion order. 0 is always ready.
      if (threshold < 0 || threshold > (int)p.size() ||
          (threshold != 0 && threshold != (int)p.size()))
        throw std::runtime_error("KVM queue: unsupported partial/missing counter threshold");
      if (threshold) pred.insert(pred.end(), p.begin(), p.end());
    };
    append(in.w0c, in.w0v);
    append(in.w1c, in.w1v);
    std::sort(pred.begin(), pred.end());
    pred.erase(std::unique(pred.begin(), pred.end()), pred.end());
    for (int p : pred) {
      if (p == id) throw std::runtime_error("KVM queue: self dependency");
      edges[p].push_back(id);
      ++indegree[id];
    }
  }
  // Reject cyclic dependencies before launching a persistent GPU kernel.
  auto left = indegree;
  std::vector<int> work;
  for (int i = 0; i < n; ++i) if (!left[i]) work.push_back(i);
  const std::vector<int> initial = work;
  int visited = 0;
  while (!work.empty()) {
    int id = work.back(); work.pop_back(); ++visited;
    for (int next : edges[id]) if (--left[next] == 0) work.push_back(next);
  }
  if (visited != n) throw std::runtime_error("KVM queue: dependency cycle");
  KvmReadyGraph graph;
  graph.indegree = std::move(indegree);
  graph.initial = initial;
  graph.offsets.reserve(n + 1);
  graph.offsets.push_back(0);
  for (const auto& row : edges) {
    graph.successors.insert(graph.successors.end(), row.begin(), row.end());
    graph.offsets.push_back((int)graph.successors.size());
  }
  return graph;
}
