# Landing branch suite results (A100, 2026-11-09)

Branch `landing/all-2026-11-09` = main + PRs #2, #3 (deduplicated), #4, #5, #1, #6 (results and bench only), #7. GPU: A100-SXM4-40GB, driver 580.105.08, CUDA 12.8, one `~/.gpu.lock` hold per script. The device code is the same after #6 and #7 as after #1 (PR #6's device changes were not taken, see the PR text), so the GPU suites ran once, on the tree after #1 (executor commit 82228ea).

| Suite | Command | Result |
|---|---|---|
| Executor | `KVM_HOST=<host> KVM_REMOTE=<dir> ./run_exec_tests.sh` | 8609 checks, 0 failed, 0 suites failed, PASS (`exec-suites.txt`) |
| `kvm.cu` | `./run_remote.sh --selftest`, `--runs 8 --steps 300` | selftest ok; run ends PASS, VM 1.50x best baseline, weights bitwise equal to all baselines (`kvm-cu.txt`) |
| Generic training | `KVM_HOST=<host> tests/generic/run.sh` | FAILED: reg_mlp, res_rms_mlp, cnn1d pass on both tilings (golden within 2e-4); encoder_reg fails on both tilings (worst `v.enc.k.b` rel L2 10.5, also `act53` 4.18); the manifest stage ran reg_mlp (PASS) and then lost the remote scratch before cnn1d (`generic.txt`) |
