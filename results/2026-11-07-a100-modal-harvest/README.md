# Modal harvest 2026-11-07 (kvm-suite volume, A100)
Raw logs from sub-agent Modal runs, recovered after coordinator session died.
Excluded: kvm_exec_main binary (22MB), bf16zoo bundles/dumps (>20MB), per-tensor grads dirs, trace npz.
Key: combined-bisect/ab_runs.txt (162M step, main 1070-1085 ms; +attn 1090; +bf16 1118; combined 1137 => regression is additive attn +1.8% and bf16 +4.4%), combined-grads (f32 exec grads rel-L2 vs fp64 9e-4 = tf32 level; bf16s 1.1e-2), adm-m162-1 (base 862 ms -> new 460 ms), tiles162-a, kvmcu-diag2.
