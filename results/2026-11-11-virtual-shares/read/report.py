"""Run each row's bundle once on each binary (ab_bench's launch flags for its pool) and read the fusion passes' reports from the log:
the MLP pass ('fuse: N run(s) fused, M unchanged; ...[; static kernel entry: E]': main's has no entry, and its one fused form is the
one-block slice) and the GPT pass ('fuse gpt: N run(s) fused'). A row whose two reports agree runs the same function on both binaries,
so its weights must match bit for bit; a row they disagree on is held to the panel's rule on its losses (2e-3)."""
import importlib.util, json, re, subprocess, sys
from pathlib import Path
run_dir, cand, ctrl, ab_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
spec = importlib.util.spec_from_file_location('ab_bench', ab_path); ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
MLP = re.compile(r'fuse: (\d+) run\(s\) fused, (\d+) unchanged;([^\n]*)')   # the rest of the line names the entry, when it has one
ENTRY = re.compile(r'static kernel entry: ([a-z-]+)')
GPT = re.compile(r'fuse gpt: (\d+) run\(s\) fused', re.M)
def report(exe, bundle, pool, log):
    meta = json.loads((Path(bundle) / 'metadata.json').read_text())
    flags = ab.exec_flags(meta['task_count']) if pool == 'mlp' else []
    p = subprocess.run([exe, *flags, '--warm-ms', '0', bundle], capture_output=True, text=True, timeout=1800)
    log.write_text(p.stdout + p.stderr)
    ok = p.returncode == 0 and re.search(r'^OK: (\d+)/\1 tasks', p.stdout, re.M) is not None
    m, g = MLP.search(p.stderr), GPT.search(p.stderr)
    mlp_fused = int(m.group(1)) if m else 0
    e = ENTRY.search(m.group(3)) if m else None
    entry = e.group(1) if e else ('one-block' if mlp_fused else 'generic')
    return {'ok': ok, 'rc': p.returncode, 'mlp_fused': mlp_fused, 'entry': entry, 'gpt_fused': int(g.group(1)) if g else 0}
out, lines = {}, []
for line in (run_dir / 'rows.txt').read_text().splitlines():
    row, pool, bundle = line.split()
    rc = {arm: report(exe, bundle, pool, run_dir / f'report_{row}_{arm}.log') for arm, exe in (('cand', cand), ('ctrl', ctrl))}
    same = all(rc[a]['ok'] for a in rc) and all(rc['cand'][k] == rc['ctrl'][k] for k in ('mlp_fused', 'entry', 'gpt_fused'))
    prefix, tol, bar = ('w.', 0, 0.99) if same else ('loss.', 2e-3, 1.00)
    out[row] = {**{f'{a}_{k}': v for a in rc for k, v in rc[a].items()}, 'gate_prefix': prefix, 'gate_tol': tol, 'bar': bar}
    lines.append(f'{row} {prefix} {tol} {bar}')
    fmt = lambda r: f"{'ok' if r['ok'] else 'RUN FAILED rc=%d' % r['rc']}: MLP {r['mlp_fused']} fused ({r['entry']}), GPT {r['gpt_fused']} fused"
    print(f"REPORT {row}: candidate {fmt(rc['cand'])}; main {fmt(rc['ctrl'])} -> gate {prefix} {'bitwise' if not tol else 'within %g' % tol}, bar x{bar:.2f}", flush=True)
(run_dir / 'reports.json').write_text(json.dumps(out, indent=1))
(run_dir / 'gates.txt').write_text('\n'.join(lines) + '\n')
