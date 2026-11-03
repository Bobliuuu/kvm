# kvm on an RTX 5070 Laptop GPU (sm_120), 2026-11-03

`kvm.cu` at 1079f6e. NVIDIA GeForce RTX 5070 Laptop GPU (36 SMs, compute capability 12.0, 3,090 MHz maximum SM clock, 8 GB),
driver 610.74, under WSL2 with the display attached and clocks not locked; nvcc 12.9.86, host g++ 13.3.

```bash
nvcc -O3 -arch=sm_120 -std=c++17 -DKTILE=32 -DKKC=<KKC> -DKSTG=<KSTG> kvm.cu -o kvm
./kvm --selftest
./kvm --runs R --steps 300     # MLP B=128 D=128 H=256 O=64
```

The selftest passes all 16 cases (largest error 6.6e-3). All 30 runs end in `PASS`, and in every one the final weights are
bitwise identical across the four modes (`max |dW|` = 0).

The VM against the best baseline of the same run (the batched graph at 8 and 32 runs); n is the number of runs of that cell:

| KKC / KSTG | blocks per SM | 1 run | 8 runs | 32 runs |
|---|---|---|---|---|
| 64 / 4 (README flags) | 1 | 1.06x (n=4) | 0.81-0.84x (n=4) | 0.73-0.78x (n=4) |
| 64 / 2 | 2 | 1.12x | 0.95x | 0.90x |
| 32 / 3 | 3 | 0.87x | 0.94x | 0.96x |
| 32 / 2 | 4 | 0.88-0.99x (n=4) | 0.97-1.12x (n=4) | 0.91-0.97x (n=4) |

The repository README's A100 ratios (1.50x, 1.63x, 1.13x) were measured at acb3bf4 on another GPU and do not carry over: here
the VM never clearly beats the batched graph. A second resident block per SM helps (0.73-0.78x to 0.90x at 32 runs) and more do
not, which fits each instruction's claim, wait and signal running serially with its own work, overlapped only by other blocks.

Files: `env.txt`, `build.txt`, `selftest.txt`; `runs<R>.txt` (README flags) and `runs<R>_t32_k<KKC>_s<KSTG>.txt`, the first
pass; `rep_<config>_R<R>_<n>.txt`, the repeats (`sm120` is the README flags).
