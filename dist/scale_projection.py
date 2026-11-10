#!/usr/bin/env python3
"""MODELLED projection of a trillion-parameter step to N GPUs with NVLink / network links and KVM's measured hop costs.

EVERYTHING THIS SCRIPT PRINTS IS A COST-MODEL RESULT. No number here was measured on more than one GPU, on NVLink, or on a
network. The only measured inputs are the hop latencies below (one A100, executors sharing the device) and, when given, the
single-GPU throughput of the registry's matmul; every other constant is an assumption written next to its definition.

    python3 dist/scale_projection.py [--md FILE] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import scaleplan as sp   # noqa: E402

# ---- measured (one A100, shared box, indicative; results/2026-11-03-a100-exec and results/2026-11-03-dist-singlebox) --------
MEASURED_HOP_US = {
    'local dependent hop (measured 5.6 us)': 5.6,
    'cross-executor hop, median (measured 57-78 us; 57 used)': 57.0,
    'cross-executor hop, p90 (measured 203-298 us; 203 used)': 203.0,
}
# ---- assumptions: hardware presets (public peak specs; sustained efficiency is an ASSUMPTION) ------------------------------------
HW = {
    'A100-40GB (NVLink 300 GB/s/dir, 200 Gb/s NIC per GPU)': dict(peak=312.0, nvlink=300.0, nic=25.0, hbm=40.0),
    'H100-80GB (NVLink 450 GB/s/dir, 400 Gb/s NIC per GPU)': dict(peak=989.0, nvlink=450.0, nic=50.0, hbm=80.0),
}
IDEAL_HOP_US = 2.0          # assumption: NVLink put + remote flag, as in partition.Cluster defaults (intra 2 us)
NET_IDEAL_HOP_US = 15.0     # assumption: inter-node (partition.Cluster default inter 15 us)


def project(gpus, tp, pp, hop_us, hw, eff=0.5, overlap=0.0, model=None, microbatches=64):
    dp = gpus // (tp * pp)
    model = model or sp.Model.for_params(1e12)
    par = sp.Par(tp, pp, dp, zero=3, microbatches=microbatches)
    r = sp.transformer(model, par, peak_tflops=hw['peak'], eff=eff, hbm_gb=hw['hbm'])
    lps = model.layers // pp
    M = microbatches
    comm = r['comm_bytes_per_gpu_per_step']
    # hops on the critical path (latency-bound handoffs): ring steps of each collective, one per pipeline send
    tp_hops = M * lps * 6 * 2 * (tp - 1)
    pp_hops = 2 * (M + pp - 1)
    dp_hops = lps * 3 * (dp - 1)
    nic_hop = hop_us if hop_us is not None else NET_IDEAL_HOP_US
    nv_hop = hop_us if hop_us is not None else IDEAL_HOP_US
    t_tp = comm['tensor_parallel'] / (hw['nvlink'] * 1e9) + tp_hops * nv_hop * 1e-6
    t_pp = comm['pipeline'] / (hw['nic'] * 1e9) + pp_hops * nic_hop * 1e-6
    t_dp = comm['data_parallel'] / (hw['nic'] * 1e9) + dp_hops * nic_hop * 1e-6
    exposed = (t_tp + t_pp + t_dp) * (1.0 - overlap)
    lb = r['critical_path']['lower_bound_s']
    step = lb + exposed
    mfu = r['model_flops_per_step_whole_job'] / gpus / (step * hw['peak'] * 1e12)
    return dict(gpus=gpus, dp=dp, step_s=step, compute_lb_s=lb, exposed_comm_s=exposed, t_tp_s=t_tp, t_pp_s=t_pp, t_dp_s=t_dp,
                mfu=mfu, mem_gb=r['memory_bytes_per_gpu']['total'] / 1e9, fits=r['memory_bytes_per_gpu']['fits_hbm'],
                tp_hops=tp_hops, pp_hops=pp_hops, dp_hops=dp_hops)


def tables(tp=8, pp=16, eff=0.5):
    rows, md = [], []
    md.append('## MODELLED projection, dense 1T (not a measurement)\n')
    md.append(f'Model: {sp.Model.for_params(1e12).params()/1e12:.3f}T parameters, 128 layers, d={sp.Model.for_params(1e12).d}, seq 2048; TP{tp} x PP{pp} x DP, ZeRO-3 over DP, '
              f'64 microbatches of 1 sequence per replica, GEMM efficiency {eff:.0%} of peak (assumption), full recompute, 1F1B bubble.\n')
    for hw_name, hw in HW.items():
        for ov in (0.0, 0.9):
            md.append(f'\n### {hw_name}; {"no comm/compute overlap" if ov == 0 else "90% of communication overlapped"}\n')
            md.append('| GPUs | hop cost per handoff | step s | compute LB s | exposed comm s | MFU | GB/GPU | fits |')
            md.append('|---|---|---|---|---|---|---|---|')
            for n in (512, 1024, 2048, 4096, 8192):
                for label, h in [('assumed ideal (NVLink 2 us / net 15 us)', None)] + list(MEASURED_HOP_US.items()):
                    p = project(n, tp, pp, h, hw, eff, ov)
                    rows.append(dict(hw=hw_name, overlap=ov, hop=label, **p))
                    md.append(f"| {n} | {label} | {p['step_s']:.1f} | {p['compute_lb_s']:.1f} | {p['exposed_comm_s']:.2f} | {p['mfu']:.1%} | {p['mem_gb']:.1f} | {'yes' if p['fits'] else 'NO'} |")
    return rows, '\n'.join(md) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--md')
    ap.add_argument('--json')
    a = ap.parse_args()
    rows, md = tables()
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float) + '\n')
    print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
