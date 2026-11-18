// Communication operators (see mvm_ops_comm.def and include/mvm_dist.cuh for the protocol and the ordering argument).
// Included by mvm_exec.cuh after MvmCtx/ARG_* exist. Every op is block-cooperative (all 256 threads call it with the same
// arguments); validation failures are uniform across threads, so no thread leaves a barrier early.

__device__ __forceinline__ void mvm_put_body(const MvmCtx& c, MvmView s, MvmView d, int so, int dof, int rows, int cols, int sld, int dld) {
  if (!mvm_tile_ok(c, s, so, rows, cols, sld) || !mvm_tile_ok(c, d, dof, rows, cols, dld)) return;
  mvm_tile_copy(d.p + dof, dld, s.p + so, sld, rows, cols);
}
__device__ __forceinline__ void mvm_put_signal_body(const MvmCtx& c, MvmView s, MvmView d, MvmView f, int so, int dof, int rows, int cols,
                                                    int sld, int dld, int var, int ver) {
  if (!mvm_tile_ok(c, s, so, rows, cols, sld) || !mvm_tile_ok(c, d, dof, rows, cols, dld) || !mvm_flag_ok(c, f, var, ver)) return;
#ifdef MVM_DIST_BREAK_ORDER   // negative test: signal first, then a long delay, then the data (the consumer must see stale data)
  if (threadIdx.x == 0) { mvm_remote_publish((unsigned*)f.p, var, ver); unsigned long long e = mvm_now() + 200000ull; while (mvm_now() < e) {} }
  __syncthreads();
  mvm_tile_copy(d.p + dof, dld, s.p + so, sld, rows, cols);
#else
  mvm_tile_copy(d.p + dof, dld, s.p + so, sld, rows, cols);
  __syncthreads();                                   // every thread's stores precede thread 0's fence (header, cumulativity)
  if (threadIdx.x == 0) mvm_remote_publish((unsigned*)f.p, var, ver);
#endif
}
__device__ __forceinline__ MvmView mvm_words(MvmIView v) { return {(float*)v.p, v.rows, v.cols}; }   // i32 tile moved as raw 32-bit words

MVM_OP(put) {
  mvm_put_body(c, ARG_BUF(0), ARG_BUF(1), ARG_I32(2), ARG_I32(3), ARG_I32(4), ARG_I32(5), ARG_I32(6), ARG_I32(7));
}
MVM_OP(put_i32) {
  mvm_put_body(c, mvm_words(ARG_IBUF(0)), mvm_words(ARG_IBUF(1)), ARG_I32(2), ARG_I32(3), ARG_I32(4), ARG_I32(5), ARG_I32(6), ARG_I32(7));
}

MVM_OP(put_signal) {
  mvm_put_signal_body(c, ARG_BUF(0), ARG_BUF(1), ARG_BUF(8), ARG_I32(2), ARG_I32(3), ARG_I32(4), ARG_I32(5), ARG_I32(6), ARG_I32(7), ARG_I32(9), ARG_I32(10));
}
MVM_OP(put_signal_i32) {
  mvm_put_signal_body(c, mvm_words(ARG_IBUF(0)), mvm_words(ARG_IBUF(1)), ARG_BUF(8), ARG_I32(2), ARG_I32(3), ARG_I32(4), ARG_I32(5), ARG_I32(6),
                      ARG_I32(7), ARG_I32(9), ARG_I32(10));
}

MVM_OP(signal) {
  MvmView f = ARG_BUF(0); const int var = ARG_I32(1), ver = ARG_I32(2);
  if (!mvm_flag_ok(c, f, var, ver)) return;
  __syncthreads();
  if (threadIdx.x == 0) mvm_remote_publish((unsigned*)f.p, var, ver);
}

MVM_OP(reduce2) {
  MvmView x = ARG_BUF(0), y = ARG_BUF(1), o = ARG_BUF(2);
  const int xo = ARG_I32(3), yo = ARG_I32(4), oo = ARG_I32(5), n = ARG_I32(6);
  if (!mvm_tile_ok(c, x, xo, 1, n, n) || !mvm_tile_ok(c, y, yo, 1, n, n) || !mvm_tile_ok(c, o, oo, 1, n, n)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[oo + i] = x.p[xo + i] + y.p[yo + i];
}

MVM_OP(reduce_put_signal) {
  MvmView x = ARG_BUF(0), y = ARG_BUF(1), d = ARG_BUF(2), f = ARG_BUF(7);
  const int xo = ARG_I32(3), yo = ARG_I32(4), dof = ARG_I32(5), n = ARG_I32(6), var = ARG_I32(8), ver = ARG_I32(9);
  if (!mvm_tile_ok(c, x, xo, 1, n, n) || !mvm_tile_ok(c, y, yo, 1, n, n) || !mvm_tile_ok(c, d, dof, 1, n, n) || !mvm_flag_ok(c, f, var, ver)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) d.p[dof + i] = x.p[xo + i] + y.p[yo + i];
  __syncthreads();
  if (threadIdx.x == 0) mvm_remote_publish((unsigned*)f.p, var, ver);
}

// Inter-node send: payload into the next send-queue slot (mapped pinned host memory), then publish the slot.
__device__ __forceinline__ void mvm_send_proxy_body(const MvmCtx& c, MvmView s, MvmView q, int so, int rows, int cols, int sld, int rank, int dbuf,
                                                    int doff, int dld, int var, int ver) {
  if (!mvm_tile_ok(c, s, so, rows, cols, sld)) return;
  MvmSqHdr* H = (MvmSqHdr*)q.p;
  int* sm = (int*)c.smem;                            // sm[0] status, sm[1..2] ticket
  if (threadIdx.x == 0) {
    int st = 0;
    const unsigned long long bytes = (unsigned long long)rows * cols * 4ull;
    if ((size_t)q.rows * q.cols * 4 < sizeof(MvmSqHdr) || H->magic != MVM_SQ_MAGIC || H->slots == 0) st = MVM_ERR_DIST_BAD_QUEUE;
    else if (bytes > H->slot_bytes) st = MVM_ERR_DIST_MSG_TOO_BIG;
    else if (rank < 0 || dbuf < 0 || doff < 0 || dld < cols || (var >= 0 && ver < 1)) st = MVM_ERR_DIST_BAD_MSG;
    if (st) { c.fail(st, rows, cols); sm[0] = st; }
    else {
      const unsigned long long t = atomicAdd(H->tail_dev, 1ull);        // device memory: no PCIe atomics needed
      const unsigned long long t0 = mvm_now();
      while (t - ld_acq_sys64(&H->head) >= H->slots) {
        if (mvm_now() - t0 > H->timeout_ns) { st = MVM_ERR_DIST_SENDQ_TIMEOUT; break; }
        __nanosleep(256);
      }
      if (st) { c.fail(st, (int)t, (int)H->slots); }
      sm[0] = st; sm[1] = (int)(t & 0xffffffffu); sm[2] = (int)(t >> 32);
    }
  }
  __syncthreads();
  if (sm[0]) return;
  const unsigned long long t = (unsigned long long)(unsigned)sm[1] | ((unsigned long long)(unsigned)sm[2] << 32);
  char* slot = (char*)H + sizeof(MvmSqHdr) + (size_t)(t % H->slots) * (MVM_SQ_SLOT_HDR + (size_t)H->slot_bytes);
  mvm_tile_copy((float*)(slot + MVM_SQ_SLOT_HDR), cols, s.p + so, sld, rows, cols);
  __syncthreads();
  if (threadIdx.x == 0) {
    MvmSqSlot* h = (MvmSqSlot*)slot;
    h->dst_rank = rank; h->dst_buf = dbuf; h->dst_off = doff; h->dst_ld = dld; h->rows = rows; h->cols = cols;
    h->var = var; h->ver = ver; h->nbytes = rows * cols * 4;
    mvm_fence_sys();
    st_rel_sys64(&h->seq, t + 1);
  }
}

MVM_OP(send_proxy) {
  mvm_send_proxy_body(c, ARG_BUF(0), ARG_BUF(1), ARG_I32(2), ARG_I32(3), ARG_I32(4), ARG_I32(5), ARG_I32(6), ARG_I32(7), ARG_I32(8), ARG_I32(9),
                      ARG_I32(10), ARG_I32(11));
}
MVM_OP(send_proxy_i32) {
  mvm_send_proxy_body(c, mvm_words(ARG_IBUF(0)), ARG_BUF(1), ARG_I32(2), ARG_I32(3), ARG_I32(4), ARG_I32(5), ARG_I32(6), ARG_I32(7), ARG_I32(8),
                      ARG_I32(9), ARG_I32(10), ARG_I32(11));
}

__device__ __forceinline__ unsigned short mvm_f32_to_bf16(float x) {   // round to nearest even; NaN stays NaN
  unsigned u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40);
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
MVM_OP(pack_bf16) {
  MvmView s = ARG_BUF(0), d = ARG_BUF(1);
  const int so = ARG_I32(2), dof = ARG_I32(3), n = ARG_I32(4), words = (n + 1) / 2;
  if (!mvm_tile_ok(c, s, so, 1, n, n) || !mvm_tile_ok(c, d, dof, 1, words, words)) return;
  unsigned* out = (unsigned*)(d.p + dof);
  for (int i = threadIdx.x; i < words; i += blockDim.x) {
    const unsigned lo = mvm_f32_to_bf16(s.p[so + 2 * i]);
    const unsigned hi = 2 * i + 1 < n ? mvm_f32_to_bf16(s.p[so + 2 * i + 1]) : 0u;
    out[i] = lo | (hi << 16);
  }
}
MVM_OP(unpack_bf16) {
  MvmView s = ARG_BUF(0), d = ARG_BUF(1);
  const int so = ARG_I32(2), dof = ARG_I32(3), n = ARG_I32(4), acc = ARG_I32(5), words = (n + 1) / 2;
  if (!mvm_tile_ok(c, s, so, 1, words, words) || !mvm_tile_ok(c, d, dof, 1, n, n)) return;
  const unsigned* in = (const unsigned*)(s.p + so);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const unsigned w = in[i >> 1];
    const float v = __uint_as_float((i & 1 ? (w & 0xffff0000u) : (w << 16)));
    d.p[dof + i] = acc ? d.p[dof + i] + v : v;
  }
}
