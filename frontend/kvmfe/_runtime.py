"""Access to the compiler package's modules (runtime/*.py import each other flat, so runtime/ goes on sys.path)."""
import sys

from .defs import REPO

RUNTIME = REPO / 'runtime'
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import initgen  # noqa: E402,F401   (the compiler modules compile_workload / workload_vm are imported where needed)
