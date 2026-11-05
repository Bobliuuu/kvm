"""Fault-tolerant multi-step driver for the CPU multi-device simulator (design + test harness; see DISTRIBUTED.md section 9).

Protocol (what is simulated):
  * A TRAINING STEP is the unit of recovery. Persistent state (weights, optimizer state) is published at a step boundary as
    version k; everything else (landing buffers, activations, inbox slots, version variables) is single-assignment per step and is
    rebuilt from scratch by re-running the step, so a fresh set of executors needs nothing but the version-k state.
  * Checkpoint = every rank's persistent shards at version k, held in memory by the rank itself AND by a buddy rank
    ((r+1) % P): one lost rank is always recoverable from its buddy; two adjacent lost ranks are not (CheckpointLost).
  * A rank is killed mid-step at an arbitrary scheduler action (between a data write and its signal, with messages in
    flight, ...). Survivors detect it (reliable.hpp: PEER_DOWN / DEAD; here: the exception), abort the step, a spare takes the
    dead slot, every rank reloads version k, the step is re-run. The result must equal the failure-free run BITWISE.
  * What is NOT simulated: the failure detector's timing, the spare allocation, the buddy copy's network cost, the checkpoint
    to durable storage, partial-step recovery (survivors keep finished work). Those are design only.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verify    # noqa: E402,F401  (sets up sys.path)
import sim       # noqa: E402
import partition as pt   # noqa: E402


class CheckpointLost(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, P, version, state):
        self.version, self.P = version, P
        # copies[holder][owner] = {buffer: values}
        self.copies = {h: {} for h in range(P)}
        for r in range(P):
            for h in (r, (r + 1) % P):
                self.copies[h][r] = {k: list(v) for k, v in state[r].items()}

    def lose(self, ranks):
        for r in ranks:
            self.copies[r] = {}

    def restore(self):
        out = {}
        for r in range(self.P):
            for h in (r, (r + 1) % self.P):
                if r in self.copies[h]:
                    out[r] = {k: list(v) for k, v in self.copies[h][r].items()}
                    break
            else:
                raise CheckpointLost(f'no surviving copy of rank {r} at version {self.version}')
        return out


def train(plan, state_names, steps, seed=0, kills=None, rng=None, lose_extra=()):
    """Run `steps` steps. kills: {step: (rank, action_index_or_None)} (None = random point). Returns (final state per rank,
    log). Every restart reloads the last checkpoint into a fresh executor set."""
    P = len(plan.shards)
    kills = dict(kills or {})
    results0, _ = None, None
    # version 0 = the workload's initial values
    dev0 = [sim.Device(s) for s in plan.shards]
    state = [{n: list(d.buf[n]) for n in state_names} for d in dev0]
    ck = Checkpoint(P, 0, state)
    log = []
    step = 0
    while step < steps:
        init = ck.restore()
        kill = None
        if step in kills:
            r, at = kills.pop(step)
            if at is None:
                # total actions of a clean run, to pick a point inside it
                sim.run(plan, seed, init=init)
                n = sim.run.last_steps
                at = rng.randrange(1, n)
            kill = (r, at)
        try:
            results, _ = sim.run(plan, seed + step, init=init, kill=kill)
        except sim.RankKilled as e:
            log.append(('killed', step, e.rank, e.at))
            ck.lose([e.rank] + list(lose_extra))      # the dead rank's memory (own copy + the buddy copy it held) is gone
            continue                                    # restore() raises CheckpointLost if no copy survived; else re-run the step
        state = [{n: list(results[r][n]) for n in state_names} for r in range(P)]
        step += 1
        ck = Checkpoint(P, step, state)
        log.append(('done', step))
    return state, log
