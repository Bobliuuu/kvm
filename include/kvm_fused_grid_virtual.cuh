#pragma once
// The virtual grid member, device side (kvm_fused_grid.h: kvm_grid_layout_chunked): the grid step of P members, its arithmetic and every sum in
// its order, run by phys < P blocks, each owning a contiguous chunk of every role family, so R runs of one gang share one wave at phys = 108 / R.
// A block does each role's part exactly as grid_run's member of that role does it; its weights stay in the model and stream through one chunk
// load a group, stored back right after their own update; gelu'(Z) goes to L2 (d.gq). The grid step costs per phase, not per member's work
// (the grid bed: 4x a member's work, +14-16% a step); a member carried role by role cost a resident member's whole phase (the grid bed), so the
// chunk is loaded, computed and reduced as one contiguous range. Every cross-member read goes through L2 (cp.async.cg, ld.cg).
#include "kvm_fused_grid.cuh"

namespace kvm_fgrid {

// One step of physical member `pm` of a chunked virtual grid run (kvm_fused_grid.h: kvm_grid_layout_chunked; the arguments as grid_run's). It
// owns a contiguous chunk of every role family and does each role's part exactly as grid_run's member of that role does it: the same GEMMs on
// the same K ranges in the same order (on views into one contiguous load, not a block of its own), the same sums, the same updates. So a run's
// bits are the P-member run's at any phys. Loads are whole chunks: a group of up to G hidden blocks of one column is one A slice and one weight
// block, its GEMMs run back to back, and its reduce rows are one contiguous range.
template <class PH>
__device__ __forceinline__ void grid_run_chunked(int pm, const int32_t* ids, const int32_t* tgt, float* loss_out, float* sm, const int* abort, PH ph) {
  kvm_fmlp::cp_commit_wait();
  __syncthreads();
  const KvmGridDesc& d = *reinterpret_cast<const KvmGridDesc*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, phys = d.phys, T = T96, E = d.E, H = d.H, Hp = d.Hp, V = d.V, VP = d.VP, L = d.L;
  const KvmGridChunkShape c = kvm_grid_chunk_shape(d, d.G);
  const float lr = d.lr, inv_t = 1.f / (float)T;
  unsigned* const ctl = d.ctl;
  unsigned long long at = kvm_gang::cross_start(ctl, pm, phys);
  auto cross = [&]() { return kvm_gang::cross(ctl, pm, phys, at, abort); };
  int* sid = reinterpret_cast<int*>(sm + d.sints);
  int* stg = sid + T;
  float* const A = sm + d.sA;      // [T][c.pA]: a group's A slice, the head's rows of the top layer, or X0
  float* const W = sm + d.sS[0];   // a group's weight block [kc][c.pW], the head's [kh][c.pWh], or layer 0's [E][c.pW0]
  float* const B = sm + d.sB;      // [T][c.pB]
  float* const Bias = sm + d.sSb[0];
  float* const gq = d.gq + (size_t)pm * d.gqs;   // gelu'(Z): layer 0's [T][c.w0], then each hidden layer's [c.rows][c.wn]
  // this member's chunks
  const int nh = L > 1 ? d.ka * d.nb : 0;
  const int h0 = kvm_grid_chunk(nh, phys, pm), h1 = kvm_grid_chunk(nh, phys, pm + 1);
  const int q0 = kvm_grid_chunk(d.nb0, phys, pm), q1 = kvm_grid_chunk(d.nb0, phys, pm + 1);   // layer 0's column blocks
  const int s0 = kvm_grid_chunk(d.kah, phys, pm), s1 = kvm_grid_chunk(d.kah, phys, pm + 1);   // the head's row blocks
  const int e0 = kvm_grid_chunk(V, phys, pm), e1 = kvm_grid_chunk(V, phys, pm + 1);           // embedding rows
  const int cr0 = kvm_grid_chunk(T, phys, pm), cr1 = kvm_grid_chunk(T, phys, pm + 1);         // cross-entropy rows
  // each column run of the chunk's hidden blocks, in order: f(hb, a_lo, a_hi, rowoff), rowoff its first row in the gelu' slot
  auto runs = [&](auto&& f) {
    int ro = 0;
    for (int i = h0; i < h1;) {
      const int hb = i / d.ka, a_lo = i % d.ka;
      int n = 1;
      while (i + n < h1 && a_lo + n < d.ka) ++n;
      f(hb, a_lo, a_lo + n, ro);
      ro += kvm_grid_off(T, d.ka, a_lo + n, 1) - kvm_grid_off(T, d.ka, a_lo, 1);
      i += n;
    }
  };
  auto koff = [&](int a) { return kvm_grid_off(Hp, d.ka, a); };
  auto noff = [&](int b) { return kvm_grid_off(Hp, d.nb, b); };
  auto vmin = [](int a, int b) { return a < b ? a : b; };
  const int cc0 = kvm_grid_off(Hp, d.nb0, q0), wc0 = kvm_grid_off(Hp, d.nb0, q1) - cc0, vc0 = vmin(H - cc0, wc0);
  const int kh0 = kvm_grid_off(Hp, d.kah, s0), whk = kvm_grid_off(Hp, d.kah, s1) - kh0, vkh = vmin(H - kh0, whk);

  // ---- load: the step's tokens and targets; X0 and layer 0's chunk of W0 and b0
  for (int i = tid; i < T; i += NT) { sid[i] = ids[i]; stg[i] = tgt[i]; }
  __syncthreads();
  if (q1 > q0) {
    kvm_fmlp::for_cells(T, E >> 2, [&](int r, int c4) { kvm_fmlp::cp_async16(A + r * c.pA + 4 * c4, d.emb + (size_t)sid[r] * E + 4 * c4); });
    copy_rows_padded(W, c.pW0, d.w[0] + cc0, H, E, wc0, E, vc0);
    copy_rows_padded(Bias, wc0, d.b[0] + cc0, H, 1, wc0, 1, vc0);
  }
  kvm_fmlp::cp_commit_wait();
  __syncthreads();
  ph(0);
  // ---- forward, layer 0: the chunk's columns for every row (each column's chain is its block's in grid_run), gelu'(Z) to L2, H written
  if (q1 > q0) {
    gemm_tall<false, false>(A, c.pA, W, c.pW0, B, c.pB, T, wc0, E);
    __syncthreads();
    float* Y0 = d.Y[0];
    kvm_fmlp::for_cells(T, wc0, [&](int r, int cc) { float f, g; kvm_fmlp::gelu_fg(B[r * c.pB + cc] + Bias[cc], f, g); gq[r * c.w0 + cc] = g; Y0[(size_t)r * Hp + cc0 + cc] = f; });
  }
  ph(sub_f0(0));
  if (!cross()) return;
  ph(sub_f0(1));
  // ---- forward, hidden layers: each group's partials (one A slice, one weight block, a GEMM a block), then each run's rows summed
  for (int l = 1; l < L; ++l) {
    runs([&](int hb, int a_lo, int a_hi, int) {
      const int n0 = noff(hb), wn = noff(hb + 1) - n0, vn = vmin(H - n0, wn);
      for (int g = a_lo; g < a_hi; g += d.G) {
        const int ge = vmin(g + d.G, a_hi), k0 = koff(g), kw = koff(ge) - k0, vk = vmin(H - k0, kw);
        kvm_fsplit::copy_rows(A, c.pA, d.Y[l - 1] + k0, Hp, T, kw);
        copy_rows_padded(W, c.pW, d.w[l] + (size_t)k0 * H + n0, H, kw, wn, vk, vn);
        kvm_fmlp::cp_commit_wait();
        __syncthreads();
        for (int a = g; a < ge; ++a) {
          const int ka = koff(a) - k0, wk = koff(a + 1) - koff(a);
          kvm_fmlp::gemm<false, false, NRB, false, NRB>(A + ka, c.pA, W + (size_t)ka * c.pW, c.pW, d.part + (size_t)a * T * Hp + n0, Hp, T, wn, wk);
        }
        __syncthreads();   // A and W take the next group
      }
    });
    ph(sub_fh(l, 0));
    if (!cross()) return;
    ph(sub_fh(l, 1));
    runs([&](int hb, int a_lo, int a_hi, int rowoff) {
      const int n0 = noff(hb), wn = noff(hb + 1) - n0, vn = vmin(H - n0, wn);
      const int rr0 = kvm_grid_off(T, d.ka, a_lo, 1), nrow = kvm_grid_off(T, d.ka, a_hi, 1) - rr0;
      copy_rows_padded(Bias, wn, d.b[l] + n0, H, 1, wn, 1, vn);
      kvm_fmlp::cp_commit_wait();
      __syncthreads();
      float* gl = gq + (size_t)T * c.w0 + (size_t)(l - 1) * c.rows * c.wn + (size_t)rowoff * c.wn; float* Yl = d.Y[l];
      sum_cells(nrow, wn, (size_t)T * Hp, d.ka, [&](int r, int cc) { return d.part + (size_t)(rr0 + r) * Hp + n0 + cc; }, [&](int r, int cc, float z) {
        float f, g; kvm_fmlp::gelu_fg(z + Bias[cc], f, g); gl[r * c.wn + cc] = g; Yl[(size_t)(rr0 + r) * Hp + n0 + cc] = f; });
      __syncthreads();   // Bias takes the next run's
    });
    ph(sub_fh(l, 2));
    if (!cross()) return;
    ph(sub_fh(l, 3));
  }
  ph(1);
  // ---- head: the chunk's row blocks, one A slice and one weight block (both kept for the head's backward)
  if (s1 > s0) {
    kvm_fsplit::copy_rows(A, c.pA, d.Y[L - 1] + kh0, Hp, T, whk);
    copy_rows_padded(W, c.pWh, d.wh + (size_t)kh0 * V, V, whk, V, vkh, V);
    kvm_fmlp::cp_commit_wait();
    if (VP > V) kvm_fmlp::for_cells(whk, VP - V, [&](int r, int cc) { W[r * c.pWh + V + cc] = 0.f; });   // zero logits, zero gradients
    __syncthreads();
    for (int q = s0; q < s1; ++q) {
      const int kq = kvm_grid_off(Hp, d.kah, q) - kh0, wq = kvm_grid_off(Hp, d.kah, q + 1) - kvm_grid_off(Hp, d.kah, q);
      kvm_fmlp::gemm<false, false, NRB, false, NRB>(A + kq, c.pA, W + (size_t)kq * c.pWh, c.pWh, d.hpart + (size_t)q * T * VP, VP, T, VP, wq);
    }
  }
  if (!cross()) return;
  ph(2);
  // ---- cross-entropy: the chunk's rows, as grid_run takes a row
  for (int r = cr0; r < cr1; ++r) {
    float* const lrow = B;
    for (int v = tid; v < V; v += NT) lrow[v] = sum_cg(d.hpart + (size_t)r * VP + v, (size_t)T * VP, d.kah);
    __syncthreads();
    if (warp == 0) {
      float x[KVM_GRID_MAXN / 32];
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) { const int v = lane + 32 * j; x[j] = v < V ? lrow[v] : -INFINITY; mx = fmaxf(mx, x[j]); }
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      const int t = stg[r];
      float lt = 0.f;
#pragma unroll
      for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) if (lane + 32 * j == t) lt = x[j];
      for (int o = 16; o; o >>= 1) lt += __shfl_xor_sync(0xffffffffu, lt, o);
      float z = 0.f;
#pragma unroll
      for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) if (lane + 32 * j < V) { x[j] = expf(x[j] - mx); z += x[j]; }
      for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
#pragma unroll
      for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) {
        const int v = lane + 32 * j;
        if (v < VP) d.dlog[(size_t)r * VP + v] = v < V ? (x[j] / z - (v == t ? 1.f : 0.f)) * inv_t : 0.f;
      }
      if (lane == 0) d.rloss[r] = mx + logf(z) - lt;
    }
    __syncthreads();
  }
  if (!cross()) return;
  if (pm == 0 && tid == 0) { float sum = 0.f; for (int r = 0; r < T; ++r) sum += __ldcg(d.rloss + r); loss_out[0] = sum / (float)T; loss_out[1] = inv_t; }
  ph(3);
  // ---- backward, head: each row block's output-gradient partial from its block before the update, the updates, the chunk back to the model
  if (s1 > s0) {
    kvm_fsplit::copy_rows(B, c.pB, d.dlog, VP, T, VP);
    kvm_fmlp::cp_commit_wait();
    __syncthreads();
    for (int q = s0; q < s1; ++q) {
      const int kq = kvm_grid_off(Hp, d.kah, q) - kh0, wq = kvm_grid_off(Hp, d.kah, q + 1) - kvm_grid_off(Hp, d.kah, q);
      gemm_tall<false, true>(B, c.pB, W + (size_t)kq * c.pWh, c.pWh, d.dyp + kh0 + kq, Hp, T, wq, VP);   // dlogits Wh_block^T
    }
    __syncthreads();
    for (int q = s0; q < s1; ++q) {
      const int kq = kvm_grid_off(Hp, d.kah, q) - kh0, wq = kvm_grid_off(Hp, d.kah, q + 1) - kvm_grid_off(Hp, d.kah, q);
      kvm_fmlp::gemm_m<true, false, true>(A + kq, c.pA, B, c.pB, W + (size_t)kq * c.pWh, c.pWh, wq, VP, T, lr);   // Wh_block -= lr H^T dlogits
    }
    __syncthreads();
    store_rows(d.wh + (size_t)kh0 * V, V, W, c.pWh, vkh, V);
  }
  if (!cross()) return;
  ph(4);
  // ---- backward, hidden layers from the top: each run's dZ rows, then each group's input-gradient partials and its blocks' updates
  for (int l = L - 1; l >= 1; --l) {
    const int nparts = l == L - 1 ? 1 : d.nb, p = L - 1 - l;
    runs([&](int hb, int a_lo, int a_hi, int rowoff) {
      const int n0 = noff(hb), wn = noff(hb + 1) - n0;
      const int rr0 = kvm_grid_off(T, d.ka, a_lo, 1), nrow = kvm_grid_off(T, d.ka, a_hi, 1) - rr0;
      const float* gl = gq + (size_t)T * c.w0 + (size_t)(l - 1) * c.rows * c.wn + (size_t)rowoff * c.wn;
      sum_cells(nrow, wn, (size_t)T * Hp, nparts, [&](int r, int cc) { return d.dyp + (size_t)(rr0 + r) * Hp + n0 + cc; },
                [&](int r, int cc, float v) { d.dz[(size_t)(rr0 + r) * Hp + n0 + cc] = v * __ldcg(gl + r * c.wn + cc); });
    });
    ph(sub_bh(p, 0));
    if (!cross()) return;
    ph(sub_bh(p, 1));
    runs([&](int hb, int a_lo, int a_hi, int) {
      const int n0 = noff(hb), wn = noff(hb + 1) - n0, vn = vmin(H - n0, wn);
      kvm_fsplit::copy_rows(B, c.pB, d.dz + n0, Hp, T, wn);   // the column's dZ, every group of the run reads it
      if (a_lo == 0) copy_rows_padded(Bias, wn, d.b[l] + n0, H, 1, wn, 1, vn);
      for (int g = a_lo; g < a_hi; g += d.G) {
        const int ge = vmin(g + d.G, a_hi), k0 = koff(g), kw = koff(ge) - k0, vk = vmin(H - k0, kw);
        kvm_fsplit::copy_rows(A, c.pA, d.Y[l - 1] + k0, Hp, T, kw);
        copy_rows_padded(W, c.pW, d.w[l] + (size_t)k0 * H + n0, H, kw, wn, vk, vn);
        kvm_fmlp::cp_commit_wait();
        __syncthreads();
        for (int a = g; a < ge; ++a) {
          const int ka = koff(a) - k0, wk = koff(a + 1) - koff(a);
          kvm_fmlp::gemm<false, true, NRB, false, NRB>(B, c.pB, W + (size_t)ka * c.pW, c.pW, d.dyp + (size_t)hb * T * Hp + koff(a), Hp, T, wk, wn);   // dZ W_block^T
        }
        __syncthreads();
        for (int a = g; a < ge; ++a) {
          const int ka = koff(a) - k0, wk = koff(a + 1) - koff(a);
          kvm_fmlp::gemm_m<true, false, true>(A + ka, c.pA, B, c.pB, W + (size_t)ka * c.pW, c.pW, wk, wn, T, lr);   // W_block -= lr X^T dZ
        }
        __syncthreads();
        store_rows(d.w[l] + (size_t)k0 * H + n0, H, W, c.pW, vk, vn);
        __syncthreads();   // A and W take the next group
      }
      if (a_lo == 0) {   // the column's bias, by its first block's member, as grid_run stores it
        for (int cc = tid; cc < wn; cc += NT) { float s = 0.f; for (int r = 0; r < T; ++r) s += B[r * c.pB + cc]; Bias[cc] -= lr * s; }
        __syncthreads();
        store_rows(d.b[l] + n0, H, Bias, wn, 1, vn);
      }
      __syncthreads();   // B and Bias take the next run's
    });
    ph(sub_bh(p, 2));
    if (!cross()) return;
    ph(sub_bh(p, 3));
  }
  ph(5);
  // ---- backward, layer 0: the chunk's dZ columns, each column block's input-gradient partial and its update, the chunk back to the model
  if (q1 > q0) {
    const int nparts0 = L == 1 ? 1 : d.nb;
    sum_cells(T, wc0, (size_t)T * Hp, nparts0, [&](int r, int cc) { return d.dyp + (size_t)r * Hp + cc0 + cc; },
              [&](int r, int cc, float v) { B[r * c.pB + cc] = v * __ldcg(gq + r * c.w0 + cc); });
    kvm_fmlp::for_cells(T, E >> 2, [&](int r, int c4) { kvm_fmlp::cp_async16(A + r * c.pA + 4 * c4, d.emb + (size_t)sid[r] * E + 4 * c4); });   // X0 again: rows not yet updated
    copy_rows_padded(W, c.pW0, d.w[0] + cc0, H, E, wc0, E, vc0);
    copy_rows_padded(Bias, wc0, d.b[0] + cc0, H, 1, wc0, 1, vc0);
    kvm_fmlp::cp_commit_wait();
    __syncthreads();
    for (int q = q0; q < q1; ++q) {
      const int cq = kvm_grid_off(Hp, d.nb0, q) - cc0, wq = kvm_grid_off(Hp, d.nb0, q + 1) - kvm_grid_off(Hp, d.nb0, q);
      gemm_tall<false, true>(B + cq, c.pB, W + cq, c.pW0, d.dx0 + (size_t)q * T * E, E, T, E, wq);   // dZ W0_block^T
    }
    __syncthreads();
    for (int q = q0; q < q1; ++q) {
      const int cq = kvm_grid_off(Hp, d.nb0, q) - cc0, wq = kvm_grid_off(Hp, d.nb0, q + 1) - kvm_grid_off(Hp, d.nb0, q);
      gemm_tall<true, false, true>(A, c.pA, B + cq, c.pB, W + cq, c.pW0, E, wq, T, lr);   // W0_block -= lr X0^T dZ
    }
    for (int cc = tid; cc < wc0; cc += NT) { float s = 0.f; for (int r = 0; r < T; ++r) s += B[r * c.pB + cc]; Bias[cc] -= lr * s; }
    __syncthreads();
    store_rows(d.w[0] + cc0, H, W, c.pW0, E, vc0);
    store_rows(d.b[0] + cc0, H, Bias, wc0, 1, vc0);
  }
  if (!cross()) return;
  ph(6);
  // ---- the chunk's embedding rows, as grid_run updates a member's
  {
    int* const tok = reinterpret_cast<int*>(A);   // [T + 1]
    float* const ts = B;                          // [ntok][E]
    if (tid == 0) { int k = 0; for (int t = 0; t < T; ++t) if (sid[t] >= e0 && sid[t] < e1) tok[k++] = t; tok[T] = k; }
    __syncthreads();
    const int ntok = tok[T];
    sum_cells(ntok, E, (size_t)T * E, d.nb0, [&](int i, int cc) { return d.dx0 + (size_t)tok[i] * E + cc; }, [&](int i, int cc, float v) { ts[i * E + cc] = v; });
    __syncthreads();
    for (int v = e0 + warp; v < e1; v += NT / 32)
      for (int cc = lane; cc < E; cc += 32) {
        float g = 0.f;
        for (int i = 0; i < ntok; ++i) if (sid[tok[i]] == v) g += ts[i * E + cc];
        d.emb[(size_t)v * E + cc] -= lr * g;
      }
  }
  if (!cross()) return;
  ph(7);
}

}  // namespace kvm_fgrid
