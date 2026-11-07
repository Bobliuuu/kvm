// Per-operator cost metadata for the scheduler, keyed by operator NAME (kvm_ops.def and the optional op packs are owned by
// other agents; this table only adds a resource profile and a bytes-moved estimate next to them).
//   KC(name, RES, bytes_expr, flops_expr)
//     RES    COMPUTE | BW | LAT | MIXED      what the task is limited by
//     bytes  expression for the bytes one task moves to/from DRAM, in terms of U(k) = argument word k read as a non-negative
//            i32 (see the op's signature in kvm_ops.def; the index is the argument position)
//     flops  same, for arithmetic (informational; only bytes are used)
// An operator that is not listed gets KVM_COST_DEFAULT: MIXED, 0 bytes (never throttled as bandwidth-bound).
// A BW task that moves fewer than KVM_LAT_BYTES is demoted to LAT at run time (a 4 KiB copy is latency-bound, not a stream).
// Duplicate names fail to compile (duplicate specialization); a name that is not registered is caught by
// kvm_cost_names_registered() in the tests.
KC(noop,     LAT,     0, 0)
KC(copy,     BW,      8ull * U(2), 0)                                  // src:buf,dst:buf,n:i32          read n + write n floats
KC(add,      BW,      12ull * U(3), U(3))                              // a,b,out,n                      2 reads + 1 write
KC(scale,    BW,      8ull * U(3), U(3))                               // x,out,alpha,n
KC(relu,     BW,      8ull * U(2), U(2))                               // x,out,n
KC(sgd,      BW,      12ull * U(3), 2ull * U(3))                       // w,g,lr,n                       read w,g; write w
KC(predict,  BW,      8ull * U(3), U(3))                               // x,w,out,n
KC(mse_grad, BW,      12ull * U(5), 6ull * U(5))                       // pred,target,x,loss,grad,n
KC(matmul,   MIXED,   4ull * (U(3) * U(5) + U(5) * U(4) + U(3) * U(4)), 2ull * U(3) * U(4) * U(5))  // a,b,out,m,n,k
KC(fill,     BW,      4ull * U(2), 0)                                  // x,value,n
KC(spin,     LAT,     0, 0)                                            // us
// ---- GPT pack (include/kvm_ops_gpt.def). Only the operators that stream memory are listed (bytes from the element ranges); GEMM tiles,
// attention and the small reductions keep the default (MIXED, 0 bytes). Without these entries bw_cta caps could
// not see AdamW or the other sweeps at all.
KC(adamw,            BW, 28ull * (U(7) - U(6)), 10ull * (U(7) - U(6)))          // p,g,m,v read; p,m,v written: 7 words per element
KC(gradnorm_partial, BW, 4ull * (U(4) - U(3)), 2ull * (U(4) - U(3)))            // g read
KC(residual_add,     BW, 12ull * (U(4) - U(3)), U(4) - U(3))                    // a,b read; out written
KC(gelu_fwd,         BW, 8ull * (U(3) - U(2)), 8ull * (U(3) - U(2)))
KC(gelu_bwd,         BW, 12ull * (U(4) - U(3)), 10ull * (U(4) - U(3)))
KC(zero_buf,         BW, 4ull * (U(2) - U(1)), 0)
KC(embedding_fwd,    BW, 8ull * (U(5) - U(4)) * U(7), 0)                        // rows (r1-r0) of width d: gather + write
KC(softmax_ce_fwd,   BW, 4ull * (U(5) - U(4)) * U(6), 5ull * (U(5) - U(4)) * U(6))
KC(softmax_ce_bwd,   BW, 8ull * (U(6) - U(5)) * U(7), 6ull * (U(6) - U(5)) * U(7))
