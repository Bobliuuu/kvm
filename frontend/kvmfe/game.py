"""Game / frame workload builder for the real Doom port (ops doom_tick / doom_present / doom_status from include/kvm_ops_game.def).

Per frame f:   doom_tick(f) -> doom_status(f) -> doom_present(f, row block) x n -> barrier publishing `frame` version f+1
  * the engine is stateful and serial: tick f+1 depends on tick f (and on present f, which reads the engine's screen) through the
    state buffer, which every op declares as a region it reads/writes;
  * 35 Hz pacing is deadline metadata: frame f carries deadline_us = (f+1) * 1e6/35 relative to admission;
  * presents write frame slot f % 2 (double buffered); the barrier publishes the `frame` variable, so a host consumer reads frame
    f `exact` and the present two frames later waits for it, exactly like double-buffered weights;
  * the scheduler sees tasks, regions and versions only. If the .def does not define the ops, building fails with a clear error.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import log
from .emit import GraphPolicy, lower_program, to_workload
from .ir import Builder, FrontendError, Tensor
from .lower import TileTask, Tiling, blocks, lowers
from .opset import op as device_op

FRAME_VAR = 'frame'
SCREEN_W, SCREEN_H, WORDS_PER_ROW = 320, 200, 80          # 8-bit palette indices, 4 per word
PALETTE_WORD, FB_WORDS, ST_HEADER = 16000, 16256, 16     # doom_present fb layout; state header words, then 2 hash words/frame


@dataclass(frozen=True)
class GameConfig:
    frames: int = 8
    hz: int = 35
    priority: int = 100
    present_rows: int = 40       # screen rows per present task
    status: bool = True          # record the per-frame pixel hash (proof of identity against the CPU build)


@lowers('game_tick')
def _lower_tick(op, tc):
    st, inp = op.ins['state'], op.ins['input']
    yield TileTask('doom_tick', dict(frame=op.attrs['frame'], input=inp, state=st), [st.region(), inp.region()], [st.region()])


@lowers('game_status')
def _lower_status(op, tc):
    st = op.ins['state']
    yield TileTask('doom_status', dict(frame=op.attrs['frame'], state=st), [st.region()], [st.region()])


@lowers('game_present')
def _lower_present(op, tc):
    st, tok, fb = op.ins['state'], op.ins['tok'], op.outs['fb']
    for r0, r1 in blocks(SCREEN_H, op.attrs['present_rows']):
        writes = [fb.region(r0 * WORDS_PER_ROW, r1 * WORDS_PER_ROW, 0, 1)]
        if r0 == 0:
            writes.append(fb.region(PALETTE_WORD, FB_WORDS, 0, 1))      # the palette rides with the first block
        yield TileTask('doom_present', dict(fb=fb, row0=r0, rows=r1 - r0), [st.region(), tok.region()], writes)


def build_game(gc: GameConfig) -> Builder:
    if gc.frames < 1 or gc.hz < 1:
        raise FrontendError('frames and hz must be >= 1')
    for name in ('doom_tick', 'doom_present') + (('doom_status',) if gc.status else ()):
        device_op(name)                                               # fails clearly when the Doom defs are absent
    b = Builder('game')
    b.phase = 'game'
    state = Tensor('doom_state', 1, ST_HEADER + 2 * gc.frames, 'state', buf='doom_state', init=('fill', 0.0))
    inp = Tensor('doom_input', 1, gc.frames + 1, 'input', buf='doom_input', init=('fill', 0.0))   # no magic word: scripted player
    ftok = Tensor('frame_token', 1, 1, 'state', buf='frame_token', init=('fill', 0.0))
    b.tensors += [state, inp, ftok]
    for f in range(gc.frames):
        b.step = f
        fb = Tensor(f'doom_frame@{f}', FB_WORDS, 1, 'state', buf=f'doom_frame.{f % 2}', init=('fill', 0.0))
        b.tensors.append(fb)
        b.emit('game_tick', {'state': state, 'input': inp}, {'state': state}, frame=f)
        if gc.status:
            b.emit('game_status', {'state': state}, {'state': state}, frame=f)
        b.emit('game_present', {'state': state, 'tok': ftok}, {'fb': fb}, present_rows=gc.present_rows)
        b.barrier([fb, ftok], [ftok], publishes=FRAME_VAR)
    return b


def compile_game(gc: GameConfig, tiling: Tiling):
    b = build_game(gc)
    period = 1_000_000 // gc.hz
    low = lower_program(b.ops, tiling, 'shape', 0, prefix='game.act')
    spec = to_workload(low, {'game': GraphPolicy(gc.priority, lambda t: (t.op.step + 1) * period)})
    log.info('game: %d frames at %d Hz (period %d us): %d tasks, %d edges', gc.frames, gc.hz, period, len(low.tasks), low.edges)
    return low, spec
