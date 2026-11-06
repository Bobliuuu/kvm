# Runs log (feat/generality)

Box runs use a private scratch dir, nvcc builds outside the GPU lock, short lock holds for runs. Modal was not used (launch denied by the
permission layer; the coordinator's authorization is not a user approval).

| date | what | where | result |
|---|---|---|---|
| 2026-11-06 | python suites (frontend 93, runtime 150 tests incl. packs) | remote CPU | OK |
| 2026-11-06 | zoo vs CPU golden, cpu+a100 tilings, 3xTF32, A100, prebuilt executor of the integration tip (device code unchanged) | `results/2026-11-06-generality/generic-a100-run1.txt` | 17/18 PASS (all <= 1.3e-6 rel L2); encoder_reg failed on AdamW noise of the exactly-zero key-bias gradient -> case now has no key bias |
| 2026-11-06 | encoder_reg + wide_mlp x {cpu, a100, a100_big} (big GEMM tiles flag 32 on the GPU) | `generic-a100-run2-encoder-widemlp.txt` | 6/6 PASS (wide_mlp a100_big worst 3.5e-5) |
| 2026-11-06 | litmus weakpub/mp (serial build via `KVM_JOBS=1`) | A100 | pending (ptxas ~30 GB, box memory saturated by other agents) |
