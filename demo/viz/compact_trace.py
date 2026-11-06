#!/usr/bin/env python3
"""Shrink a full executor trace.csv (hundreds of MB at 162M) to the columns demo/viz/timeline.py draws, lossless for the figure.

  python3 demo/viz/compact_trace.py TRACE.csv OUT.npz

Keeps every task (no sampling): start_us and end_us (float32 microseconds, about 1 us resolution at 13 s, far below a pixel), smid (uint8) and the op class
index of timeline.CLASSES (uint8), and for Doom tasks the frame number (int16, -1 otherwise; slot order, 8 tasks per frame) which the video needs. Tasks that never ran (start < 0) are dropped, exactly as timeline.load_trace does for a CSV.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeline as tl  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    start, end, smid, cls, frame = [], [], [], [], []
    game_seen = 0                                            # Doom tasks come in slot order, 8 per frame: tick, 4 x present, status (+1 spare)
    with open(src, newline='') as fh:
        for r in csv.DictReader(fh):
            s, e = float(r['start_us']), float(r['end_us'])
            if int(r['workload']) == 2:
                f = game_seen // 8; game_seen += 1
            else:
                f = -1
            if s < 0 or e < 0:
                continue
            frame.append(f)
            start.append(s); end.append(e); smid.append(int(r['smid'])); cls.append(tl.classify(r['op'], int(r['workload'])))
    order = np.argsort(np.array(start), kind='stable')       # sorted by start: deltas compress far better
    np.savez_compressed(dst, start_us=np.array(start, dtype=np.float32)[order], end_us=np.array(end, dtype=np.float32)[order],
                        smid=np.array(smid, dtype=np.uint8)[order], cls=np.array(cls, dtype=np.uint8)[order],
                        frame=np.array(frame, dtype=np.int16)[order])
    print('wrote', dst, Path(dst).stat().st_size, 'bytes,', len(start), 'tasks', file=sys.stderr)


if __name__ == '__main__':
    main()
