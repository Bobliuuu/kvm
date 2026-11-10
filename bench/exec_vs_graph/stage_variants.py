#!/usr/bin/env python3
"""Stage executor variants for modal_ab.py: git-archive each ref (or WORKTREE = the current working tree, uncommitted edits included)
into STAGE/<tree>/ (one tree per distinct ref; variants of the same ref share its build) and write STAGE/variants.json.

  python3 bench/exec_vs_graph/stage_variants.py STAGE main=origin/main port=WORKTREE nocont=WORKTREE:--no-cont
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = ("results", "assets", "doom", "docs", "dist", "demo/viz", ".claude")


def stage_tree(ref, d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    if ref == "WORKTREE":
        files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True, check=True).stdout.split("\n")
        for f in files:
            if not f or f.startswith(SKIP) or not (ROOT / f).is_file():
                continue
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, d / f)
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() + "+wt"
    arch = subprocess.run(["git", "-C", str(ROOT), "archive", ref], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(d)] + [f"--exclude={s}" for s in SKIP], input=arch, check=True)
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", ref], capture_output=True, text=True).stdout.strip()


def main():
    st = Path(sys.argv[1])
    st.mkdir(parents=True, exist_ok=True)
    out, trees = [], {}
    for spec in sys.argv[2:]:
        name, _, rest = spec.partition("=")
        ref, _, args = rest.partition(":")
        if ref not in trees:
            t = re.sub(r"[^A-Za-z0-9_.-]", "_", ref)
            trees[ref] = (t, stage_tree(ref, st / t))
        t, sha = trees[ref]
        out.append({"name": name, "ref": ref, "tree": t, "sha": sha, "args": args})
        print(name, ref, t, sha, args)
    (st / "variants.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
