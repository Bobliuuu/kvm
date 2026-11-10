"""Single-sourced ABI: parses include/kvm_task_abi.h and every operator registry in include/.

Registries: the .def files include/kvm_ops_list.inc includes, read from it in its order, so a pack the executor compiles
is never invisible here; each is optional except the first (the core registry). Ids are sparse and globally unique. Signature
grammar: `name:type`, type in buf | buf? | i32 | f32; `buf?` is an optional buffer (word -1 = absent), `buf` is required.

Nothing about descriptor layout or the operator table is duplicated in Python, so the host
compiler, the CPU reference and the CUDA executor cannot drift apart.
"""
import logging
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger('kvm.abi')
INCLUDE_DIR = Path(__file__).resolve().parents[1] / 'include'
ARG_TYPES = ('buf', 'i32', 'f32')
ABSENT_BUF = -1                  # word of an absent optional buffer argument
OPS_LIST = 'kvm_ops_list.inc'   # the executor's one list of registry files (the X-macro expansion includes it)
_DEF_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"/]+\.def)"', re.M)


class AbiError(RuntimeError):
    """The header or operator registry is malformed (a repo bug, not a workload bug)."""


class OpArg(NamedTuple):
    name: str
    type: str
    optional: bool = False


@dataclass(frozen=True)
class OpSpec:
    name: str
    id: int
    args: tuple[OpArg, ...]


@dataclass(frozen=True)
class TaskAbi:
    version: int
    no_var: int
    fields: tuple[str, ...]   # KvmTaskDesc member names in declaration order
    layout: struct.Struct     # little-endian, one int32 per field

    @property
    def size(self) -> int:
        return self.layout.size


def _parse_registry(path: Path, ops: dict, ids: dict) -> None:
    line_re = re.compile(r'^X\(\s*(\w+)\s*,\s*(\d+)\s*,\s*"([^"]*)"\s*\)')
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.startswith('X('):
            continue
        m = line_re.match(line)
        if not m:
            raise AbiError(f'{path}:{lineno}: unparsable operator line: {line!r}')
        name, op_id, sig = m.group(1), int(m.group(2)), m.group(3)
        args = []
        for item in filter(None, (s.strip() for s in sig.split(','))):
            arg_name, _, arg_type = item.partition(':')
            optional = arg_type == 'buf?'
            arg_type = 'buf' if optional else arg_type
            if arg_type not in ARG_TYPES or not arg_name.isidentifier():
                raise AbiError(f'{path}:{lineno}: bad argument {item!r} (types: buf, buf?, i32, f32)')
            args.append(OpArg(arg_name, arg_type, optional))
        if len({a.name for a in args}) != len(args):
            raise AbiError(f'{path}:{lineno}: duplicate argument name in {name}')
        if name in ops or op_id in ids:
            raise AbiError(f'{path}:{lineno}: duplicate operator name or id: {name}={op_id} '
                           f'(already {ids.get(op_id) or name})')
        ops[name] = OpSpec(name, op_id, tuple(args))
        ids[op_id] = name


def registry_files(include_dir: Path = INCLUDE_DIR) -> tuple[str, ...]:
    """The registry files `include_dir`'s kvm_ops_list.inc includes, in its order, the first being the core registry.

    Every literal `#include "NAME.def"` counts whatever build condition guards it (KVM_LEAN, KVM_WITH_GAME, __has_include): this is
    the union over builds, as the operator ids are. The test hook `#include KVM_EXTRA_OPS_DEF` names no file and has no entry.
    """
    path = Path(include_dir) / OPS_LIST
    if not path.exists():
        raise AbiError(f'{path}: the executor\'s registry list not found')
    names = tuple(_DEF_INCLUDE.findall(path.read_text()))
    if not names:
        raise AbiError(f'{path}: includes no registry file')
    if len(set(names)) != len(names):
        raise AbiError(f'{path}: a registry file is included twice: {names}')
    return names


@lru_cache(maxsize=None)
def load_ops(include_dir: Path = INCLUDE_DIR) -> dict[str, OpSpec]:
    """All operators of every registry file kvm_ops_list.inc lists that is present in `include_dir`, in the list's order."""
    include_dir = Path(include_dir)
    ops: dict[str, OpSpec] = {}
    ids: dict[int, str] = {}
    names = registry_files(include_dir)
    for name in names:
        path = include_dir / name
        if path.exists():
            _parse_registry(path, ops, ids)
            log.debug('loaded %s (%d operators so far)', path.name, len(ops))
        elif name == names[0]:
            raise AbiError(f'{path}: core operator registry not found')
        else:
            log.debug('%s: listed by %s, absent here (an optional pack)', name, OPS_LIST)
    if not ops:
        raise AbiError(f'{include_dir}: no operators found')
    return ops


@lru_cache(maxsize=None)
def load_task_abi(path: Path = INCLUDE_DIR / 'kvm_task_abi.h') -> TaskAbi:
    text = Path(path).read_text()
    define = lambda key: int(re.search(rf'#define\s+{key}\s+\(?(-?\d+)\)?', text).group(1))
    body = re.search(r'struct\s+(?:alignas\(\d+\)\s+)?KvmTaskDesc\s*\{(.*?)\};', text, re.S)
    if not body:
        raise AbiError(f'{path}: KvmTaskDesc not found')
    fields = tuple(re.findall(r'^\s*int32_t\s+(\w+)\s*;', body.group(1), re.M))
    if len(fields) != len(re.findall(r'int32_t', body.group(1))):
        raise AbiError(f'{path}: KvmTaskDesc must consist of int32_t members only')
    layout = struct.Struct(f'<{len(fields)}i')
    declared = int(re.search(r'sizeof\(KvmTaskDesc\)\s*==\s*(\d+)', text).group(1))
    if layout.size != declared:
        raise AbiError(f'{path}: {len(fields)} int32 fields = {layout.size} bytes, header asserts {declared}')
    return TaskAbi(define('KVM_ABI_VERSION'), define('KVM_NO_VAR'), fields, layout)
