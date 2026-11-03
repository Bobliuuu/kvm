"""The operator registry as the single source of truth: parses include/kvm_ops*.def (X(name, id, "arg:type,...") lines)."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from .ir import FrontendError

REPO = Path(__file__).resolve().parents[2]
_LINE = re.compile(r'^X\(\s*(\w+)\s*,\s*(\d+)\s*,\s*"([^"]*)"\s*\)')


def default_include_dir() -> Path:
    return Path(os.environ.get('KVM_INCLUDE', REPO / 'include'))


def read_defs(include_dir: str | Path | None = None) -> dict:
    """{op: (id, ((arg, type), ...))} over every kvm_ops*.def in the directory (kvm_ops.def first, then _gpt, _game)."""
    return _read(str(include_dir or default_include_dir()))


@lru_cache(maxsize=None)
def _read(directory: str) -> dict:
    d = Path(directory)
    files = sorted(d.glob('kvm_ops*.def'))
    if not files:
        raise FrontendError(f'no kvm_ops*.def in {d} (set KVM_INCLUDE or pass --defs-dir)')
    ops = {}
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.startswith('X('):
                continue
            m = _LINE.match(line)
            if not m:
                raise FrontendError(f'{f}:{n}: unparsable operator line {line!r}')
            name, oid, sig = m.group(1), int(m.group(2)), m.group(3)
            if name in ops:
                raise FrontendError(f'{f}:{n}: duplicate operator {name}')
            ops[name] = (oid, tuple(tuple(a.strip().split(':')) for a in sig.split(',') if a.strip()))
    return ops
