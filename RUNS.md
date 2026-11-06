# Run log

| when (UTC) | what | where | outputs |
|---|---|---|---|
| 2026-11-06 | perf/kernels: test_ops + bench mm/attn, HEAD (K edits) vs base fc0ced7 | A100 scratch dir kern-w1, job.sh | results/2026-11-06-a100-kernels/ |
| 2026-11-06 | perf/kernels: bench mm sweep KVM_GPT_BIG_NST=4 and KVM_RND_VELT | A100 scratch dir kern-w1, job2.sh | results/2026-11-06-a100-kernels/ |
| 2026-11-06 02:04 EST | Modal A100-40GB: test_ops + bench mm/attn, HEAD vs base fc0ced7 (tag a100-k1); command `BASE_DIR=<base> GPU=A100-40GB TAG=a100-k1 modal run --detach tests/gpu_ops/modal_bench.py` | Modal app ap-ptYcRgZnjCPqAruEPZRDF6, volume kvm-kern-results | results/2026-11-06-a100-kernels/ |
