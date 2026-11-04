#!/usr/bin/env python3
"""Back-of-the-envelope scaling model for a dense transformer with ZeRO-3 + pipeline + tensor parallelism. NOTHING here is measured:
it is arithmetic over stated assumptions, printed so DISTRIBUTED.md can be regenerated and challenged.

    python3 dist/scaling.py [--params 1e12] [--gpus 4096] [--tp 8] [--pp 16] [--microbatches 64] [--seq 4096] [--global-batch-seqs 4096]
                            [--layers 128] [--hidden 25600] [--peak-tflops 989] [--mfu 0.40] [--nic-gbs 50] [--nvlink-gbs 450]
                            [--wan-gbps 100] [--json]
"""
import argparse
import json


def model(a):
    N, G, TP, PP, M = a.params, a.gpus, a.tp, a.pp, a.microbatches
    DP = G // (TP * PP)
    assert TP * PP * DP == G, 'tp * pp must divide gpus'
    bytes_per_param = dict(weights_bf16=2, master_fp32=4, adam_m_v_fp32=8, grads_bf16=2)        # 16 B/param
    state_total = N * sum(bytes_per_param.values())
    shard = N / (TP * PP)                                  # parameters in one (tensor, pipeline) shard
    state_per_gpu = state_total / G                         # ZeRO-3 over DP on top of TP x PP: every state byte lives once
    layers_per_stage = a.layers / PP
    params_per_layer = N / a.layers
    gathered_layer = params_per_layer / TP * 2             # bf16 weights of one layer for one TP rank, after all-gather
    # activations: keep only each layer's input (full recompute), sequence-parallel over TP; 1F1B keeps <= PP microbatches in flight
    act_layer_input = a.seq * a.hidden * 2 / TP
    act_stored = layers_per_stage * min(PP, M) * act_layer_input
    act_recompute = 34 * a.seq * a.hidden / TP             # Korthikanti et al. per-layer activation bytes, one microbatch live at a time
    gathered_buffers = 2 * gathered_layer                  # current + prefetched layer
    mem = dict(model_state=state_per_gpu, gathered_weights=gathered_buffers, stored_activations=act_stored, recompute_workspace=act_recompute)
    # gathering once per step means the WHOLE stage's bf16 weights stay resident for the step; per-microbatch regather needs 2 layers
    mem_once = dict(mem, gathered_weights=shard * 2)
    # per step communication volume per GPU
    ag = shard * 2 * (DP - 1) / DP                         # one full all-gather of the stage's bf16 weights (receive side)
    zero = dict(allgather_fwd=ag, allgather_bwd=ag, reducescatter_grads=ag)                          # gathered once per step (weights kept across microbatches)
    zero_per_mb = dict(allgather_fwd=ag * M, allgather_bwd=ag * M, reducescatter_grads=ag)           # re-gathered for every microbatch
    tp_msgs_per_layer_per_mb = 10                          # 4 fwd + 4 bwd + 2 recompute AG/RS of s*b*h bf16 (sequence parallel)
    tp = layers_per_stage * M * tp_msgs_per_layer_per_mb * (a.seq * a.hidden * 2) * (TP - 1) / TP
    pp = 2 * M * (a.seq * a.hidden * 2 / TP)               # activation forward + gradient backward across a stage boundary
    tokens = a.global_batch_seqs * a.seq
    flops_per_gpu = 8 * N * tokens / G                     # 6N fwd+bwd, +2N recompute
    t_compute = flops_per_gpu / (a.peak_tflops * 1e12 * a.mfu)
    t_zero = sum(zero.values()) / (a.nic_gbs * 1e9)
    t_zero_mb = sum(zero_per_mb.values()) / (a.nic_gbs * 1e9)
    t_tp = tp / (a.nvlink_gbs * 1e9)
    t_pp = pp / (a.nic_gbs * 1e9)
    # pipeline across datacenters: the stage boundary crosses the WAN; pairs = TP x DP GPUs each carry their own slice
    wan_bytes_per_boundary = pp * TP * DP
    wan_gbps_needed = wan_bytes_per_boundary * 8 / t_compute / 1e9
    bubble = (PP - 1) / (M + PP - 1)
    return dict(
        dp=DP, tp=TP, pp=PP, microbatches=M, params=N, gpus=G,
        state_total_TB=state_total / 1e12, bytes_per_param=sum(bytes_per_param.values()),
        memory_GB={k: v / 1e9 for k, v in mem.items()}, memory_total_GB=sum(mem.values()) / 1e9,
        memory_if_gathered_once_GB=sum(mem_once.values()) / 1e9,
        comm_GB_per_gpu_per_step=dict(zero3_gather_once={k: v / 1e9 for k, v in zero.items()}, zero3_regather_per_microbatch_total=sum(zero_per_mb.values()) / 1e9,
                                      tensor_parallel=tp / 1e9, pipeline_p2p=pp / 1e9),
        step_seconds=dict(compute=t_compute, zero3_comm_once=t_zero, zero3_comm_per_microbatch=t_zero_mb, tp_comm=t_tp, pp_comm=t_pp),
        comm_fraction_of_compute=dict(zero3_once=t_zero / t_compute, zero3_per_microbatch=t_zero_mb / t_compute, tp=t_tp / t_compute, pp=t_pp / t_compute),
        pipeline_bubble_fraction=bubble,
        wan=dict(pipeline_boundary_GB_per_step=wan_bytes_per_boundary / 1e9, required_gbps=wan_gbps_needed, provisioned_gbps=a.wan_gbps,
                 hop_latency_vs_microbatch_time=0.015 / (t_compute / M)),
        required_dp_gbs_per_gpu_to_hide=sum(zero.values()) / t_compute / 1e9,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params', type=float, default=1e12)
    ap.add_argument('--gpus', type=int, default=4096)
    ap.add_argument('--tp', type=int, default=8)
    ap.add_argument('--pp', type=int, default=16)
    ap.add_argument('--microbatches', type=int, default=64)
    ap.add_argument('--seq', type=int, default=4096)
    ap.add_argument('--global-batch-seqs', type=int, default=4096)
    ap.add_argument('--layers', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=25600)
    ap.add_argument('--peak-tflops', type=float, default=989.0, help='dense bf16 per GPU (H100 SXM); NOT a measurement of anything in this repo')
    ap.add_argument('--mfu', type=float, default=0.40)
    ap.add_argument('--nic-gbs', type=float, default=50.0, help='GB/s per GPU to other nodes (one 400 Gb/s NIC per GPU)')
    ap.add_argument('--nvlink-gbs', type=float, default=450.0, help='GB/s per GPU inside a node')
    ap.add_argument('--wan-gbps', type=float, default=100.0)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    r = model(a)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    m = r['memory_GB']
    print(f"{r['params']:.3g} params, {r['gpus']} GPUs = TP {r['tp']} x PP {r['pp']} x DP {r['dp']}, {r['microbatches']} microbatches/step")
    print(f"state: {r['bytes_per_param']} B/param -> {r['state_total_TB']:.1f} TB total, {m['model_state']:.2f} GB per GPU with ZeRO-3 (every byte stored once)")
    print('per-GPU memory GB: ' + ', '.join(f'{k} {v:.2f}' for k, v in m.items()) + f'  => {r["memory_total_GB"]:.2f} GB (weights re-gathered per microbatch); {r["memory_if_gathered_once_GB"]:.2f} GB if the whole stage stays gathered for the step')
    c = r['comm_GB_per_gpu_per_step']
    print(f"per-step comm per GPU (GB): ZeRO-3 gather fwd {c['zero3_gather_once']['allgather_fwd']:.1f} + bwd {c['zero3_gather_once']['allgather_bwd']:.1f} + grad reduce-scatter "
          f"{c['zero3_gather_once']['reducescatter_grads']:.1f}; re-gather per microbatch instead: {c['zero3_regather_per_microbatch_total']:.0f}; tensor-parallel {c['tensor_parallel']:.0f}; pipeline p2p {c['pipeline_p2p']:.1f}")
    s, f = r['step_seconds'], r['comm_fraction_of_compute']
    print(f"step compute {s['compute']:.1f} s at MFU assumption; comm time/compute: ZeRO-3 {f['zero3_once']:.1%} (per-microbatch regather {f['zero3_per_microbatch']:.0%}), TP {f['tp']:.1%}, PP {f['pp']:.2%}")
    print(f"bandwidth that must be overlapped: ZeRO-3 {r['required_dp_gbs_per_gpu_to_hide']:.2f} GB/s per GPU (NIC provides {a.nic_gbs:.0f})")
    print(f"pipeline bubble {r['pipeline_bubble_fraction']:.1%}; WAN at a pipeline-stage boundary: {r['wan']['pipeline_boundary_GB_per_step']:.0f} GB/step = {r['wan']['required_gbps']:.0f} Gb/s needed "
          f"(have {a.wan_gbps:.0f}); 15 ms hop = {r['wan']['hop_latency_vs_microbatch_time']:.2%} of a microbatch")


if __name__ == '__main__':
    main()
