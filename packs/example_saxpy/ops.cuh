// Device code of packs/example_saxpy/ops.def: one KVM_OP per X line; block-cooperative, every thread of the block calls it.
KVM_OP(saxpy) {
  KvmView x = ARG_BUF(0), y = ARG_BUF(1), o = ARG_BUF(2); float alpha = ARG_F32(3); int n = ARG_I32(4);
  if (!c.fits(n, x, y, o)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = alpha * x.p[i] + y.p[i];
}
