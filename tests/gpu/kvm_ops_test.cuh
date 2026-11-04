// Implementations of tests/gpu/kvm_ops_test.def
KVM_OP(opt_add) {
  KvmView x = ARG_BUF(0), y = ARG_BUF_OPT(1), o = ARG_BUF(2); int n = ARG_I32(3);
  if (!c.fits(n, x, o)) return;
  if (y.p && !c.fits(n, y)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = x.p[i] + (y.p ? y.p[i] : 0.f);
}
KVM_OP(gather) {
  KvmIView idx = ARG_IBUF(0); KvmView s = ARG_BUF(1), o = ARG_BUF(2); int n = ARG_I32(3);
  if (idx.p == nullptr || !c.fits(n, s, o) || n > idx.rows * idx.cols) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int j = idx.p[i];
    if (j < 0 || j >= s.rows * s.cols) { c.fail(KVM_ERR_BOUNDS, j, i); break; }
    o.p[i] = s.p[j];
  }
}
KVM_OP(iota_i) {
  KvmIView o = ARG_IBUF(0); int n = ARG_I32(1);
  if (o.p == nullptr || n < 0 || n > o.rows * o.cols) { if (o.p) c.fail(KVM_ERR_BOUNDS, n); return; }
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = i;
}
KVM_OP(opt_i) {
  KvmIView x = ARG_IBUF_OPT(0); KvmView o = ARG_BUF(1); int n = ARG_I32(2);
  if (!c.fits(n, o)) return;
  if (x.p && n > x.rows * x.cols) { c.fail(KVM_ERR_BOUNDS, n); return; }
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = x.p ? (float)x.p[i] : -1.f;
}
