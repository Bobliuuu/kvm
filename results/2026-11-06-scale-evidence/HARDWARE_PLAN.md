# What real multi-GPU evidence needs (for the user to approve; nothing here has been launched)

Status of the question "does the NCCL-free design scale?": before this branch every distributed run was ranks-as-threads on one
GPU. Single containers on Modal give real NVLink between 2-8 GPUs; **no single container gives a network between nodes**, so
inter-node claims (send_proxy over a NIC, RDMA, WAN) stay unmeasured until two separate hosts exist.

| Box | What it can turn into real evidence | Tests (all already in the repo) | Estimated cost |
|---|---|---|---|
| 2x A100-40GB, NVLink (Modal `A100-40GB:2`, about $2.10/GPU/h list) | real peer-to-peer pointers across physical GPUs; NVLink bandwidth of `put`; true cross-GPU version-hop latency (replaces the 57-78 us same-GPU number); one-rank-per-GPU plans `tp_gpus zero_gpus pipe_p2p tp_ring zero_mixed`; the sharded AdamW stack (golden + 33M-param unsharded vs sharded) | `python dist/modal_scale.py` stages: `test_dist bw pingpong ops`, `test_dist plan`, `scale_gpu.py golden`, `scale_gpu.py pair` | about 25 min x 2 GPUs = $2-3 |
| 4x A100-40GB NVLink (`A100-40GB:4`) | same with 4 ranks on 4 GPUs: ZeRO gather/reduce-scatter over NVLink, pipeline of 4 stages, scaling 2 -> 4 of the measured step | same stages | about 30 min x 4 GPUs = $4-5 |
| 8x H100 NVLink/NVSwitch (`H100:8`, about $3.95/GPU/h list, sm_90 build) | 8-way tensor/ZeRO groups inside one node (the TP8 group of the 1T plan), NVSwitch hop and bandwidth at the width the projection assumes | same stages, `--gpus h100x8` | 45 min cap x 8 = about $24 |
| 2 hosts with a NIC (not available in one Modal container) | the only way to measure `send_proxy` over a real network and the `HAVE_IBVERBS` backend; remote hop across hosts | `dist/run_multigpu_tests.sh` with `KVM_NODES=a,b` (stage `multinode-tcp-plans`), then an RDMA build | not priced here |

Total for the three Modal rows: roughly $30-35, inside the $60 cap the coordinator quoted. Launch commands (detached, results to a
Modal Volume and the app logs): `modal run --detach dist/modal_scale.py::main --gpus a100x2|a100x4|h100x8`.
These were NOT run: the launch was denied by the permission classifier, so the user has to allow it (or run the commands).
