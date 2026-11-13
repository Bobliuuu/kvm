"""col_blocks: a split column reduction keeps every column's summation order (col_reduce in kvm_ops_gpt.cuh sums a column in strips of
NC columns, NC fixed by its task's width), and the default tiling's blocks are unchanged."""
import unittest

import _path  # noqa: F401
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling, blocks, col_blocks


def strip_width(width):
    return 8 if width <= 8 else 16 if width <= 16 else 32 if width <= 32 else 64


def per_column(ranges):
    return {c: strip_width(b - a) for a, b in ranges for c in range(a, b)}


class ColBlocks(unittest.TestCase):
    def test_every_column_keeps_its_strip(self):
        old, new = Tiling.preset('a100'), Tiling.preset('a100', col_sub=64)
        for width in range(1, 2200):
            ranges = col_blocks(width, new)
            self.assertEqual([c for a, b in ranges for c in range(a, b)], list(range(width)), width)   # a partition, in order
            self.assertEqual(per_column(ranges), per_column(blocks(width, old.cols)), width)

    def test_split_is_finer_where_it_can_be(self):
        new = Tiling.preset('a100', col_sub=64)
        self.assertEqual(len(col_blocks(768, new)), 12)
        self.assertEqual(col_blocks(72, new), [(0, 72)])              # the 8-column tail joins its neighbour
        self.assertEqual(col_blocks(260, new)[-1], (256, 260))        # the old 4-column block stays whole, as before

    def test_default_keeps_the_blocks(self):
        t = Tiling.preset('a100', col_sub=0)
        for width in range(1, 1100):
            self.assertEqual(col_blocks(width, t), blocks(width, t.cols))

    def test_col_sub_is_validated(self):
        for bad in (16, 32, -1, 64.0):
            with self.assertRaises(FrontendError):
                Tiling.preset('a100', col_sub=bad)


if __name__ == '__main__':
    unittest.main()
