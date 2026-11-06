// Per-operator cost metadata for the scheduler, keyed by operator NAME (kvm_ops.def and the optional op packs are owned by
// other agents; this table only adds a resource profile and a bytes-moved estimate next to them).
//   KC(name, RES, bytes_expr, flops_expr)
//     RES    COMPUTE | BW | LAT | MIXED      what the task is limited by
//     bytes  expression for the bytes one task moves to/from DRAM, in terms of U(k) = argument word k read as a non-negative
//            i32 (see the op's signature in kvm_ops.def; the index is the argument position)
//     flops  same, for arithmetic (informational; the arbiter only uses bytes)
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
// attention and the small reductions keep the default (MIXED, 0 bytes). Without these entries the bandwidth arbiter and bw_cta caps could
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
#ifndef KVM_COST_LEGACY   // -DKVM_COST_LEGACY rebuilds the table as it was before the audit (20 entries): the A/B baseline of tests/gpu/sched_bench.inc `mix`
// ---- Added by the arbiter audit (results/2026-11-04-claims-audit gap 5): every remaining operator, from its shapes ----
// Rule: a streaming sweep (norms, rope, softmax, gather, activations, optimizer, embedding, decode attention) is BW with the bytes it
// reads + writes in fp32; GEMM tiles and flash-attention tiles are COMPUTE (their tile-level traffic is mostly served by L2, see
// kvm_mm_bytes; the figure is reported in telemetry but a COMPUTE task is never counted against the bandwidth arbiter); the scalar,
// control and Doom ops are LAT (a few KiB at most, or serial game logic). A task that moves under KVM_LAT_BYTES is demoted to LAT at run time.
// ---- GPT pack remainder ----
KC(matmul_tile,      COMPUTE, kvm_mm_bytes(a, 6, 0, SI(3) >= 0, SI(4) >= 0, SI(5) >= 0), kvm_mm_flops(a, 6))       // a,b,out,bias?,res?,pre?,m0..flags
KC(embedding_bwd,    BW, 8ull * (U(6) - U(5)) * U(7), 0)                                                              // dtab slice read-modify-write (the dout gather is shared by all slices)
KC(layernorm_fwd,    BW, 8ull * (U(7) - U(6)) * U(8) + 8ull * (U(7) - U(6)), 8ull * (U(7) - U(6)) * U(8))            // x read, out written, mean+rstd written
KC(rmsnorm_fwd,      BW, 8ull * (U(5) - U(4)) * U(6) + 4ull * (U(5) - U(4)), 5ull * (U(5) - U(4)) * U(6))
KC(layernorm_bwd_dx, BW, 12ull * (U(7) - U(6)) * U(8), 10ull * (U(7) - U(6)) * U(8))                                   // x, dy read; dx written
KC(rmsnorm_bwd_dx,   BW, 12ull * (U(6) - U(5)) * U(7), 8ull * (U(6) - U(5)) * U(7))
KC(norm_bwd_params,  BW, 8ull * U(6) * (U(9) - U(8)), 4ull * U(6) * (U(9) - U(8)))                                    // x, dy columns [c0,c1) over R rows
KC(rope_fwd,         BW, 8ull * (U(4) - U(3)) * U(6) * U(7), 6ull * (U(4) - U(3)) * U(6) * U(7))
KC(rope_bwd,         BW, 8ull * (U(4) - U(3)) * U(6) * U(7), 6ull * (U(4) - U(3)) * U(6) * U(7))
KC(attn_fwd,         COMPUTE, kvm_attn_bytes(kvm_attn_tq(SI(7), SI(8)), kvm_attn_kv(SI(7), SI(8), true), U(11), 2, 2), 4ull * kvm_attn_tq(SI(7), SI(8)) * kvm_attn_kv(SI(7), SI(8), true) * U(11))
KC(attn_bwd_delta,   BW, 8ull * (U(4) - U(3)) * U(6) * U(7), 2ull * (U(4) - U(3)) * U(6) * U(7))
KC(attn_bwd_dq,      COMPUTE, kvm_attn_bytes(kvm_attn_tq(SI(9), SI(10)), kvm_attn_kv(SI(9), SI(10), true), U(13), 4, 2), 6ull * kvm_attn_tq(SI(9), SI(10)) * kvm_attn_kv(SI(9), SI(10), true) * U(13))
KC(attn_bwd_dkv,     COMPUTE, kvm_attn_bytes(kvm_attn_tq(SI(10), SI(11)), (unsigned long long)max(0, SI(11) - SI(10)), U(13), 4, 2), 8ull * kvm_attn_tq(SI(10), SI(11)) * (unsigned long long)max(0, SI(11) - SI(10)) * U(13))
KC(kv_append,        BW, 16ull * (U(6) - U(5)) * U(7) * U(8) * U(9), 0)                                                // k,v read; kc,vc written
KC(attn_decode,      BW, 8ull * U(10) * U(9) / 2 + 8ull * U(9), 4ull * U(10) * U(9) / 2)                              // one (batch, head): streams ~Tmax/2 cached K and V rows (lens is data dependent: half the capacity is a guess)
KC(softmax_ce_reduce, LAT, 8ull * U(3), 2ull * U(3))
KC(gradnorm_finish,  LAT, 4ull * U(1), 2ull * U(1))
KC(sample,           BW, 4ull * (U(4) - U(3)) * U(5), 3ull * (U(4) - U(3)) * U(5))
KC(argmax,           BW, 4ull * (U(3) - U(2)) * U(4), (U(3) - U(2)) * U(4))
KC(counter_inc,      LAT, 8ull * (U(2) - U(1)), 0)
KC(colsum,           BW, 4ull * U(2) * (U(4) - U(3)), U(2) * (U(4) - U(3)))
KC(select_window,    LAT, 8ull * (U(4) - U(3)), 0)
KC(gather_rows,      BW, 8ull * (U(4) - U(3)) * U(5), 0)
// ---- extra (model-zoo) pack ----
KC(act_fwd,          BW, 8ull * (U(4) - U(3)), 6ull * (U(4) - U(3)))
KC(act_bwd,          BW, 12ull * (U(5) - U(4)), 8ull * (U(5) - U(4)))
KC(silu_mul_fwd,     BW, 12ull * (U(4) - U(3)), 6ull * (U(4) - U(3)))
KC(silu_mul_bwd,     BW, 20ull * (U(6) - U(5)), 10ull * (U(6) - U(5)))
KC(softmax_rows_fwd, BW, 8ull * (U(3) - U(2)) * U(4), 5ull * (U(3) - U(2)) * U(4))
KC(softmax_rows_bwd, BW, 12ull * (U(4) - U(3)) * U(5), 4ull * (U(4) - U(3)) * U(5))
KC(gather_rows_scaled, BW, 8ull * (U(5) - U(4)) * U(6), (U(5) - U(4)) * U(6))
KC(scatter_add_rows, BW, 8ull * (U(6) - U(5)) * U(7), (U(6) - U(5)) * U(7))
KC(row_dot,          BW, 8ull * (U(5) - U(4)) * U(6), 2ull * (U(5) - U(4)) * U(6))
KC(moe_route,        BW, 8ull * U(4) * U(5), 2ull * U(4) * U(5))
KC(moe_gates_fwd,    BW, 4ull * U(4) * U(5) + 8ull * U(4) * U(6), U(4) * U(6))
KC(moe_gates_bwd,    BW, 8ull * U(5) * U(6), 2ull * U(5) * U(6))
KC(moe_balance_fwd,  BW, 4ull * U(3) * U(4), U(3) * U(4))
KC(moe_balance_bwd,  BW, 8ull * U(3) * U(4), 2ull * U(3) * U(4))
KC(attn_full_fwd,    COMPUTE, kvm_attn_bytes(kvm_attn_tq(SI(7), SI(8)), kvm_attn_kv(SI(7), SI(8), false), U(11), 2, 2), 4ull * kvm_attn_tq(SI(7), SI(8)) * kvm_attn_kv(SI(7), SI(8), false) * U(11))
KC(attn_full_bwd_dq, COMPUTE, kvm_attn_bytes(kvm_attn_tq(SI(9), SI(10)), kvm_attn_kv(SI(9), SI(10), false), U(13), 4, 2), 6ull * kvm_attn_tq(SI(9), SI(10)) * kvm_attn_kv(SI(9), SI(10), false) * U(13))
KC(attn_full_bwd_dkv, COMPUTE, kvm_attn_bytes(kvm_attn_tq(SI(10), SI(11)), (unsigned long long)SI(11), U(13), 4, 2), 8ull * kvm_attn_tq(SI(10), SI(11)) * U(11) * U(13))
KC(round_bf16,       BW, 8ull * (U(3) - U(2)), 2ull * (U(3) - U(2)))
KC(matmul_tile_bf16, COMPUTE, kvm_mm_bytes(a, 6, 0, SI(3) >= 0, SI(4) >= 0, SI(5) >= 0), kvm_mm_flops(a, 6))
KC(adamw_lr,         BW, 28ull * (U(7) - U(6)), 10ull * (U(7) - U(6)))
KC(lr_schedule,      LAT, 0, 0)
KC(scale_flat,       BW, 8ull * (U(4) - U(3)), U(4) - U(3))
KC(add_rowtile,      BW, 12ull * (U(4) - U(3)) * U(6), (U(4) - U(3)) * U(6))
KC(matmul_tile_dgelu, COMPUTE, kvm_mm_bytes(a, 4, 0, false, false, true), kvm_mm_flops(a, 4))
KC(matmul_tile_silu_mul, COMPUTE, kvm_mm_bytes(a, 5, 2, false, false, false), kvm_mm_flops(a, 5))
KC(matmul_tile_silu_mul_bwd, COMPUTE, kvm_mm_bytes(a, 6, 5, false, false, false), kvm_mm_flops(a, 6))
// ---- comm pack (fp32 payload; put/put_signal copy rows x cols) ----
KC(put,              BW, 8ull * U(4) * U(5), 0)
KC(put_signal,       BW, 8ull * U(4) * U(5), 0)
KC(signal,           LAT, 0, 0)
KC(reduce2,          BW, 12ull * U(6), U(6))
KC(reduce_put_signal, BW, 12ull * U(6), U(6))
KC(send_proxy,       BW, 8ull * U(3) * U(4), 0)
KC(pack_bf16,        BW, 6ull * U(4), 0)
KC(unpack_bf16,      BW, 6ull * U(4), U(4))
// ---- game pack: serial game logic and a small framebuffer blit; latency-bound by construction, never throttled as bandwidth ----
KC(doom_tick,        LAT, 0, 0)
KC(doom_present,     LAT, 0, 0)
KC(doom_status,      LAT, 0, 0)
KC(doom_tick_n,      LAT, 0, 0)
KC(doom_present_n,   LAT, 0, 0)
KC(doom_status_n,    LAT, 0, 0)
#endif  // KVM_COST_LEGACY
