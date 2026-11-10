"""gpt_162m memory plan and task counts of one training step (CPU only, no GPU):
  python3 plan162.py REPO_ROOT bf16s,f32
"""
import sys, collections, logging
sys.path.insert(0, sys.argv[1] + '/frontend')
from kvmfe.nn import ModelConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig
from kvmfe.lower import Tiling
for prec in sys.argv[2].split(','):
    cfg = ModelConfig(name='gpt_162m', vocab=50304, d_model=768, layers=12, heads=12, seq=1024, batch=4, precision=prec)
    c = compile_training(cfg, TrainConfig(steps=1), Tiling.preset('a100'), reuse='shape', with_spec=False)
    st = c.low.plan.stats
    ops = collections.Counter(t.dev for t in c.low.tasks)
    print(prec, 'persistent %.3f GiB pool %.3f GiB total %.3f GiB' % (st['persistent_bytes']/2**30, st['pool_bytes']/2**30, (st['persistent_bytes']+st['pool_bytes'])/2**30),
          'tasks', len(c.low.tasks), 'round_bf16', ops.get('round_bf16', 0))
