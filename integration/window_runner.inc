  // Reuses device buffers between windows. Sequential kernel launches enforce
  // the required cross-window barrier. This intentionally sacrifices overlap.
  auto run_queue_window = [&]() {
    const int window_tasks = perStep * 4;
    for (int first = 0; first < ntape; first += window_tasks) {
      KvmGraphWindow w = kvm_slice_graph(qgraph, first,
          std::min(window_tasks, ntape - first));
      CK(cudaMemcpy(q_offsets, w.offsets.data(), w.offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
      if (!w.successors.empty())
        CK(cudaMemcpy(q_successors, w.successors.data(), w.successors.size() * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(q_remaining, w.indegree.data(), w.count * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemset(q_ready, 0, w.count * sizeof(int)));
      if (!w.initial.empty()) {
        CK(cudaMemcpy(q_tasks, w.initial.data(), w.initial.size() * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<int> ones(w.initial.size(), 1);
        CK(cudaMemcpy(q_ready, ones.data(), ones.size() * sizeof(int), cudaMemcpyHostToDevice));
      }
      int zero = 0, count = (int)w.initial.size();
      CK(cudaMemcpy(q_head, &zero, sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(q_done, &zero, sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(q_tail, &count, sizeof(int), cudaMemcpyHostToDevice));
      ready_queue_kernel<<<queue_grid, NT, DSMEM>>>(dtapeP, w.count,
          q_offsets, q_successors, q_remaining, q_tasks, q_ready,
          q_head, q_tail, q_done, dloss, first);
      CK(cudaGetLastError());
      // The next iteration copies into the same buffers. Synchronous copies
      // and default-stream ordering ensure prior kernel has finished.
    }
  };
