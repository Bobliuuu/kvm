# Co-optimization: in-kernel population control (successive halving / retire rules), KVM vs PyTorch + CUDA graphs

Code: `bench/coopt/coopt_sh.cu` (KVM side), `bench/coopt/torch_sh.py` (oracle + baselines), `bench/coopt/modal_run.py` (Modal A100-40GB driver).
Model: 2-layer ReLU MLP regression (SGD), R runs with log-spaced learning rates, 300 steps, whole sweep = ONE kernel launch.
New in the VM: a `SEL` slot per step. At a rung boundary one block waits for all live runs, ranks them on device and clears `alive[r]`;
retired runs' later tiles are claimed, see the flag and skip (no waits, no work). Mode 0 = keep better half (successive halving),
mode 1 = data-dependent rule (retire diverged / converged-to-target (early stop) / worse than 3x best; survivor count is not known in advance).

Correctness oracle: fp32 (TF32 off) eager torch implementing the same rule. Checks per config: same retire decisions (`same_kill_rungs`), loss-curve
max relative error while a run is alive (<= 3.2e-4 in every config), survivors' weight L2 error (<= 5.1e-3). TF32 vs fp32.
Torch baselines: manual fwd/bwd with bmm, TF32, CUDA graphs; B1 = whole run in one graph with masking (no compute saved); B2 = static per-rung graphs
(shrinking, halving mode only); B3 = host-polled compaction with pre-cached graphs per (rung, n_alive), warm cache untimed; each also with torch.compile'd
elementwise (`_compiled`). Times are best of 7 (kernel/graph only, state reset excluded), ms, one A100-SXM4-40GB (Modal), shared-GPU noise ~1-3%.

See `c*/torch_result.json` for every number; `c*/kvm.log` for the VM run.

KNOWN INVALID: `B3_host_polled_compaction_cached_graphs_compiled` gives wrong losses (rel err ~1.0; cause not found, probably a torch.compile/capture
interaction) in configs 3 and 4; it is flagged `B3_INVALID_not_counted_compiled` and excluded from `kvm_ctrl_vs_best_torch_any`.
(Its 4.1-4.4 ms times must not be quoted.)
