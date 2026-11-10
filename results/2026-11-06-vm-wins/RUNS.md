# RUNS (bench/vm-wins)
| run | GPU | what | where |
|---|---|---|---|
| a100 final | A100-SXM4-40GB (Lambda, shared) | regime 1 matrix, 6 sizes x 6 variants, best of 5 | runs_a100tiling.txt |
| a100 default ring | same | same bundles, default ring/live_cap (streamed windows) | runs_default_streamed.txt |
| a100 hop micro | same | hop_bench, KVM_CHAIN build only | hop_chain.txt |
Scratch dir on the box: kvm-scratch-regimes-b (private). No Modal apps were launched.
