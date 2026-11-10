declare -a NAMES=([0]="mlp_f32" [1]="mlp_tf32" [2]="vit_f32" [3]="vit_tf32" [4]="llama_f32" [5]="llama_tf32" [6]="llama_fused" [7]="gqa_tied" [8]="moe_f32" [9]="moe_tf32" [10]="moe_fused" [11]="llama_accum" [12]="llama_remat" [13]="llama_bf16" [14]="mlp_bf16" [15]="gpt_fused" [16]="san_llama" [17]="san_gqa_fused" [18]="san_moe" [19]="san_moe_fused" [20]="san_vit" [21]="san_mlp" [22]="san_bf16" [23]="san_remat" [24]="llama_bf16s" [25]="mlp_bf16s" [26]="vit_bf16s" [27]="san_bf16s")
declare -- SANITIZE="0"
declare -- OUT="/tmp/zooout"
run_all () 
{ 
    for c in "${NAMES[@]}";
    do
        d="$OUT/bundles/$c";
        "$OUT/kvm_exec" --quiet --dump-buffers "$d/bundle" > "$d/dump.txt" 2> "$d/err.txt";
        echo "$c exit $?";
        for i in 1 2 3;
        do
            "$OUT/kvm_exec" --quiet "$d/bundle" | tail -n1;
        done > "$d/times.txt";
        if [ "$SANITIZE" = 1 ] || [[ "$c" == san_* ]]; then
            for tool in memcheck racecheck synccheck;
            do
                echo "== compute-sanitizer $tool $c";
                compute-sanitizer --tool $tool --error-exitcode 9 "$OUT/kvm_exec" --blocks 4 --quiet "$d/bundle" 2>&1 | tail -3;
                echo "sanitizer exit ${PIPESTATUS[0]}";
            done;
        fi;
    done
}
run_all
