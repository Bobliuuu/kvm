# Runs log (feat/generality)

Box runs use a private scratch dir, nvcc builds outside the GPU lock, short lock holds for runs. Modal was not used (launch denied by the
permission layer; the coordinator's authorization is not a user approval).

| date | what | where | result |
|---|---|---|---|
| 2026-11-06 | python suites (frontend 93, runtime 150 tests incl. packs) | remote CPU | OK |
| 2026-11-06 | `tests/generic/run_box.sh` zoo vs CPU golden on A100 | see results/2026-11-06-generality | pending (lock queue) |
| 2026-11-06 | litmus weakpub/mp (serial build via `KVM_JOBS=1`) | A100 | pending (ptxas ~30 GB, box memory saturated by other agents) |
