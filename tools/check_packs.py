#!/usr/bin/env python3
"""check_packs.py PACK.def...: every operator of the built-in registries plus the given pack registries parses and has a unique
name and id (the same check the host compiler runs); prints the packs' operators."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import abi  # noqa: E402

try:
    ops = abi.load_ops(extra=tuple(sys.argv[1:]))
except abi.AbiError as e:
    sys.exit(f'check_packs: {e}')
core = abi.load_ops(extra=())
for n, o in ops.items():
    if n not in core:
        print(f'pack op {n} id {o.id} ({", ".join(a.name for a in o.args)})')
