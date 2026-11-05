#!/usr/bin/env python3
"""Regenerates the tables in results/2026-11-04-scaleout-sim/ from dist/scaleout.py (cost model; nothing measured).
    python3 dist/scaleout_report.py [out.md]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scaleout as so   # noqa: E402
import resilience as rs   # noqa: E402

OUT = []


def p(s=''):
    OUT.append(s)
    print(s)


def row(*c): p('| ' + ' | '.join(str(x) for x in c) + ' |')


def hdr(*c):
    row(*c); row(*['---'] * len(c))


def go(model, par, topo, hw=None):
    return so.run(model, par, topo, hw or so.Hw())


def dense_par(G, tp=8, pp=16, M=64, seqs_per_dp=128, **kw):
    dp = G // (tp * pp)
    return so.Par(tp, pp, dp, M, seqs_per_dp * dp, lpt=2, **kw)


def main():
    D = so.dense_1t()
    p('## 1. Headline: dense 1T, TP8 x PP16 x DP, pipeline stages across 4 datacenters (4 stages each), ZeRO-3 over DP')
    p('Per DP rank: 64 microbatches of 2 sequences x 4096 tokens (weak scaling), 8 extra microbatches in flight per stage. WAN 100 Gb/s per datacenter pair per direction, 20 ms one way.')
    hdr('GPUs', 'step s', 'MFU', 'HFU', 'LB perfect-overlap s', 'LB +bubble s', 'actual / LB+bubble', 'exposed comm', 'other idle', 'comm hidden', 'GB/GPU')
    for G, wan in ((512, 100), (1024, 100), (2048, 100), (4096, 100), (4096, 400), (8192, 100), (8192, 400), (10240, 400)):
        topo = so.make_topology(G, wan, 20)
        r = go(D, dense_par(G, window_extra=8), topo)
        m = r.metrics
        row(f'{G} (WAN {wan} Gb/s)', f"{m['step_s']:.1f}", f"{m['mfu']:.1%}", f"{m['hfu']:.1%}", f"{m['lb_perfect_overlap_s']:.1f}", f"{m['lb_pipeline_s']:.1f}",
            f"{m['over_pipeline_lb']:.2f}", f"{m['exposed_comm_frac']:.1%}", f"{m['idle_frac']:.1%}", f"{m['comm_hidden_frac']:.0%}", f"{r.memory['max_gb']:.1f}")
    p()
    p('## 2. WAN sensitivity (4096 GPUs, dense 1T, M=64): step seconds')
    bws, lats = (50, 100, 200, 400, 800, 1600), (1, 10, 20, 50, 100)
    hdr('one-way latency \\ Gb/s per pair', *bws)
    for ms in lats:
        row(f'{ms} ms', *[f"{go(D, dense_par(4096), so.make_topology(4096, bw, ms)).metrics['step_s']:.1f}" for bw in bws])
    p()
    p('Same, with 8 extra microbatches allowed in flight per stage (`window_extra=8`, costs activation memory):')
    hdr('one-way latency \\ Gb/s per pair', *bws)
    for ms in lats:
        row(f'{ms} ms', *[f"{go(D, dense_par(4096, window_extra=8), so.make_topology(4096, bw, ms)).metrics['step_s']:.1f}" for bw in bws])
    p()
    p('## 3. Microbatches, window and activation compression across the WAN (4096 GPUs, 100 Gb/s, 20 ms)')
    hdr('variant', 'step s', 'MFU', 'bubble measured', 'bubble theory', 'GB/GPU')
    base_topo = so.make_topology(4096, 100, 20)
    for name, kw in (('M=16 (8 seq/mb)', dict(M=16)), ('M=32', dict(M=32)), ('M=64', dict(M=64)), ('M=128 (1 seq/mb)', dict(M=128)),
                     ('M=64, window +8', dict(window_extra=8)), ('M=64, window +8, fp8 activations on the wire (WHAT-IF)', dict(window_extra=8, act_bytes=1)),
                     ('M=64, window +16, fp8 activations (WHAT-IF)', dict(window_extra=16, act_bytes=1))):
        kw = dict(kw); M = kw.pop('M', 64)
        r = go(D, dense_par(4096, M=M, seqs_per_dp=128, **kw), base_topo)
        m = r.metrics
        row(name, f"{m['step_s']:.1f}", f"{m['mfu']:.1%}", f"{m['bubble_measured']:.1%}", f"{m['bubble_theory']:.1%}", f"{r.memory['max_gb']:.1f}")
    p()
    p('## 4. Overlap (4096 GPUs, WAN 400 Gb/s, 20 ms, window +8)')
    hdr('variant', 'step s', 'MFU', 'exposed comm', 'comm hidden', 'actual / LB+bubble')
    t4 = so.make_topology(4096, 400, 20)
    for name, kw, hw in (('as designed: comm on its own streams (tp_chunks=1)', dict(), so.Hw()),
                         ('tp_chunks=4 (needs row-tile ops: WHAT-IF)', dict(tp_chunks=4), so.Hw()),
                         ('comm SM tax 0%', dict(), so.Hw(comm_sm_tax=0.0)), ('comm SM tax 15%', dict(), so.Hw(comm_sm_tax=0.15)),
                         ('ZeRO-3 re-gather every microbatch', dict(dp_mode='zero3_mb'), so.Hw()),
                         ('ZeRO-1 (gather after optimizer)', dict(dp_mode='zero1'), so.Hw()),
                         ('no overlap at all (all comm serialized with compute)', dict(overlap=False), so.Hw())):
        r = go(D, dense_par(4096, window_extra=8, **kw), t4, hw)
        m = r.metrics
        row(name, f"{m['step_s']:.1f}", f"{m['mfu']:.1%}", f"{m['exposed_comm_frac']:.1%}", f"{m['comm_hidden_frac']:.0%}", f"{m['over_pipeline_lb']:.2f}")
    p()
    p('## 5. Placement: which axis crosses the WAN (4096 GPUs, 400 Gb/s, 20 ms)')
    hdr('placement', 'step s', 'MFU')
    for name, order in (('PP across datacenters (TP in node, DP in rack)', ('pp', 'dp', 'tp')), ('DP across datacenters (PP inside)', ('dp', 'pp', 'tp'))):
        r = go(D, dense_par(4096, window_extra=8, order=order), t4)
        row(name, f"{r.metrics['step_s']:.1f}", f"{r.metrics['mfu']:.1%}")
    p()
    p('## 6. MoE 1T (384 experts, top-8, 64 layers, hidden 7168), expert parallel = DP, a2a inside the rack')
    M_ = so.moe_1t()
    hdr('GPUs', 'TP x PP x DP', 'step s', 'MFU', 'a2a + TP comm exposed', 'comm hidden', 'GB/GPU')
    for G, tp, pp, sc in ((1024, 4, 16, 1.0), (4096, 4, 16, 1.0), (4096, 2, 16, 1.0), (8192, 4, 16, 1.0), (4096, 4, 16, 0.375), (8192, 4, 16, 0.375)):
        dp = G // (tp * pp)
        par = so.Par(tp, pp, dp, 60, 60 * dp * 4, ep=dp, lpt=2, window_extra=8, a2a_scale=sc)
        r = go(M_, par, so.make_topology(G, 400, 20))
        m = r.metrics
        row(G if sc == 1 else f'{G} (a2a bytes x{sc}, WHAT-IF)', f'{tp} x {pp} x {dp}', f"{m['step_s']:.1f}", f"{m['mfu']:.1%}", f"{m['exposed_comm_frac']:.1%}", f"{m['comm_hidden_frac']:.0%}", f"{r.memory['max_gb']:.1f}")
    p()
    p('## 7. Memory per GPU, headline configuration (GB; limit 80)')
    r = go(D, dense_par(4096, window_extra=8), t4)
    hdr('component', 'GB')
    for k, v in r.memory['parts_gb'].items():
        row(k, f'{v:.2f}')
    row('total (worst stage)', f"{r.memory['max_gb']:.1f}")
    p()
    p('## 8. Cross-check with a published configuration')
    p('Megatron-LM, 1T parameters, 3072 A100-80GB, sequence 2048, NVLink/NVSwitch in the node and 8 x 200 Gb/s HDR per node: reported 163 TFLOP/s per GPU = 52% of peak (arXiv:2104.04473, abstract and text).')
    p('Our model with TP8 x PP64 x DP6 (TP/PP from the model\'s recollection of that paper, NOT verified), batch 3072 sequences, A100 constants. Kernel efficiency is a free parameter:')
    hdr('kernel efficiency', 'step s', 'hardware TFLOP/s per GPU (recompute counted)', 'HFU')
    mm = so.ModelCfg('dense', 128, 25600, 2048, 51200, 4)
    tp_ = so.Topology(8, 32, 12, 1, nvlink_gbs=300, rack_gbs=25, dc_gbs=25)
    for eff in (0.6, 0.7, 0.8):
        r = go(mm, so.Par(8, 64, 6, 512, 3072, lpt=1), tp_, so.Hw(peak_tflops=312, kernel_eff=eff, hbm_gbs=2039))
        row(eff, f"{r.metrics['step_s']:.1f}", f"{r.metrics['tflops_per_gpu_hw']:.0f}", f"{r.metrics['hfu']:.1%}")
    p('This brackets 163 TFLOP/s; it shows the model is not off by a large factor in a regime with the same shape, not that it is calibrated.')
    p()
    p('## 9. Stragglers: work stealing across DP replicas (dynamic admission), 64 replicas x 64 bundles, bundle 100 us')
    hdr('slow replicas', 'static step us', 'stealing step us', 'gain')
    import random
    for k, f in ((0, 1.0), (1, 0.5), (1, 0.25), (4, 0.5), (8, 0.7)):
        sp = [1.0] * 64
        for i in range(k): sp[i * 7] = f
        a, _ = rs.step_time(64, 64 * 8, 100.0, sp, steal=False, jitter=0.1)
        b, _ = rs.step_time(64, 64 * 8, 100.0, sp, steal=True, claim_us=3.0, fetch_us=10.0, jitter=0.1)
        row(f'{k} at {f:.2f}x', f'{a:.0f}', f'{b:.0f}', f'{a / b:.2f}x')
    p()
    p('## 10. Failure goodput model (first order; per-GPU MTBF from 419 unexpected interruptions / 54 days / 16,384 GPUs reported for Llama 3, via secondary summaries)')
    mtbf = 54 * 24 * 16384 / 419
    hdr('GPUs', 'cluster MTBF h', 'checkpoint s', 'restart s', 'interval', 'goodput')
    for G in (4096, 10240, 16384):
        for ck, rst in ((30, 300), (5, 120)):
            g = rs.goodput(G, 90.0, mtbf, ck, rst)
            row(G, f"{g['cluster_mtbf_h']:.1f}", ck, rst, f"{g['interval_steps']} steps ({g['interval_s']/60:.0f} min)", f"{g['goodput']:.1%}")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
