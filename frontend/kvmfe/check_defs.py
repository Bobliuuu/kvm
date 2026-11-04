"""python3 -m kvmfe.check_defs [--defs-dir include]: how the .def registry, the frontend's CPU references and its lowerings line up.

Exit status 1 if the frontend references an op the .def files do not define (the fatal drift); other findings are informational.
"""
from __future__ import annotations

import argparse
import sys

from . import game, lower, opset                     # noqa: F401  (importing registers the lowerings)
from .defs import read_defs


def main(argv=None):
    p = argparse.ArgumentParser(prog='kvmfe.check_defs', description=__doc__)
    p.add_argument('--defs-dir', default=None)
    a = p.parse_args(argv)
    defs = read_defs(a.defs_dir)
    unknown = sorted(n for n in opset.REF if n not in defs and not n.startswith('doom_'))
    no_ref = sorted(n for n in defs if n not in opset.REF or opset.REF[n][0] is None)
    print(f'{len(defs)} ops in the .def files; {len(defs) - len(no_ref)} have a frontend CPU reference')
    print('no CPU reference (not emitted, or opaque):', ', '.join(no_ref) or 'none')
    if unknown:
        print('ERROR: frontend references ops missing from the .def files:', ', '.join(unknown))
    return 1 if unknown else 0


if __name__ == '__main__':
    sys.exit(main())
