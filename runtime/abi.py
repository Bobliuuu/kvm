"""Single-sourced ABI: parses include/kvm_task_abi.h and include/kvm_ops.def.

Nothing about descriptor layout or the operator table is duplicated in Python, so the host
compiler, the CPU reference and the CUDA executor cannot drift apart.
"""
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

INCLUDE_DIR = Path(__file__).resolve().parents[1] / 'include'
ARG_TYPES = ('buf', 'i32', 'f32')


class AbiError(RuntimeError):
    """The header or operator registry is malformed (a repo bug, not a workload bug)."""


class OpArg(NamedTuple):
    name: str
    type: str


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


@lru_cache(maxsize=None)
def load_ops(path: Path = INCLUDE_DIR / 'kvm_ops.def') -> dict[str, OpSpec]:
    ops: dict[str, OpSpec] = {}
    line_re = re.compile(r'^X\(\s*(\w+)\s*,\s*(\d+)\s*,\s*"([^"]*)"\s*\)')
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.startswith('X('):
            continue
        m = line_re.match(line)
        if not m:
            raise AbiError(f'{path}:{lineno}: unparsable operator line: {line!r}')
        name, op_id, sig = m.group(1), int(m.group(2)), m.group(3)
        args = []
        for item in filter(None, (s.strip() for s in sig.split(','))):
            arg_name, _, arg_type = item.partition(':')
            if arg_type not in ARG_TYPES or not arg_name.isidentifier():
                raise AbiError(f'{path}:{lineno}: bad argument {item!r} (types: {ARG_TYPES})')
            args.append(OpArg(arg_name, arg_type))
        if len({a.name for a in args}) != len(args):
            raise AbiError(f'{path}:{lineno}: duplicate argument name in {name}')
        if name in ops or any(o.id == op_id for o in ops.values()):
            raise AbiError(f'{path}:{lineno}: duplicate operator name or id: {name}={op_id}')
        ops[name] = OpSpec(name, op_id, tuple(args))
    if not ops:
        raise AbiError(f'{path}: no operators found')
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
