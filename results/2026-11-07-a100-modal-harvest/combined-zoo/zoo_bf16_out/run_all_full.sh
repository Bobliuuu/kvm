declare -a NAMES=([0]="llama_bf16" [1]="mlp_bf16" [2]="vit_bf16" [3]="san_bf16" [4]="llama_bf16s" [5]="mlp_bf16s" [6]="vit_bf16s" [7]="san_bf16s")
declare -- SANITIZE="0"
declare -- OUT="/tmp/zooout_b"
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
