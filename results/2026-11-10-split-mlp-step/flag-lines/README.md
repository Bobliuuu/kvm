# The split step on flag lines: an A/B on an RTX 5070 Laptop

The split bed (`bench/fused_mlp/split_bed.cu`) on the counter barrier (the code of `5c56c90`, "old") against `kvm_gang::barrier_lines`
("new"):
- Both are scratch builds with `KVM_SPLIT_Q 2`, so 2 x 18 blocks fit the 5070's 36 SMs (run with `--slices 2`).
- Built with the panel's numerics flags (`--fmad=false -Xcompiler -ffp-contract=off`) for sm_120.
- 20 steps, the first 3 against float64.
- Two rounds, the arms in alternating order. Figures are (unverified) until the team's verifier names them.

| shape | shares | check lines, old against new | µs a step, old (rounds 1, 2) | new (rounds 1, 2) |
|---|---|---|---|---|
| h256-d4 (the panel's mlp256) | 18 | 5 of 5 identical | 55.60, 47.43 | 44.76, 46.17 |
| h256-d4 | 16 | 5 of 5 identical | 147.12, 45.78 | 43.28, 44.34 |
| h576-d1 | 18 | 5 of 5 identical | 17.64, 17.53 | 16.85, 16.86 |

The barrier changes no arithmetic, and every check line repeats digit for digit. The old arm's first round at depth 4 (55.6 and 147.1 µs)
is an outlier against every other run of either arm; its cause was not read. Round 2 has none, and there the flag lines save 2.7-3.8%.
Logs: `run_split_*.log`; the card: `gpu.txt`.
