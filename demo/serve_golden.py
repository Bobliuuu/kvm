#!/usr/bin/env python3
"""PyTorch golden for the demo's serving requests at any size (the CPU reference VM cannot run gpt_162m).

A serving request (frontend/kvmfe/serve.py) prefills the prompt, then greedily decodes `max_new` tokens with a KV cache on the
`weights` snapshot it read. Its result is the token buffers `<req>.cur1 .. <req>.cur<max_new>`. This module recomputes them with
PyTorch, without a KV cache and without any kvmfe op, from the model config and the weight values:

  * version 0 (an untrained snapshot): the weights are the executor's own counter-based initial values, rebuilt from the config
    (bench/vs_pytorch/torch_bench.py, self-checked against runtime/initgen.py), so nothing has to be dumped;
  * the last version: the weights the executor holds at the end of the run, read from `kvm_run --dump '^w\\.'` (RUN/buffers/w.*.bin).
    The decode is then checked against the weights the executor produced; how the executor trained is the training check's job.
  * a middle version is not checked: no snapshot of it survives, and a CPU re-training cannot reproduce it (AdamW amplifies the
    1xTF32 noise, results/2026-11-04-a100-gpt-golden).

The comparison is teacher-forced, one token at a time: the reference is fed prompt + the executor's own previous tokens (one causal
forward gives the logits of every step) and the executor's token t_i must be the float64 argmax, or be a numerically legitimate near
tie. A tie is legitimate when the float64 logit gap  max - logit[t_i]  is within 2 * E, where E is the largest logit deviation (over
the whole vocabulary at that position) of the same model with every matmul operand rounded to TF32 (10-bit mantissa, float32
accumulate): the arithmetic class of the executor's default tiles. A wrong weight version, a wrong KV offset or a wrong RoPE position
moves logits by far more than that. With 3xTF32 tiles (`--precise`) E is measured against that as well and the check is exact.

  serve_golden.py --model gpt_162m --prompt 43 --tokens 41803 33285 42236 [--weights-dir RUN/buffers] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench' / 'vs_pytorch'))
sys.path.insert(0, str(ROOT / 'frontend'))
sys.path.insert(0, str(ROOT / 'runtime'))
import initgen  # noqa: E402
import torch_bench as tb  # noqa: E402
from kvmfe.nn import GPT, ModelConfig  # noqa: E402
from kvmfe.train import param_init  # noqa: E402

SLACK = 2.0          # a tie is legitimate within SLACK * 2 * E (two logits, each off by at most E)


def init_weights(cfg: ModelConfig) -> dict[str, np.ndarray]:
    """The executor's initial parameter values (float32), rebuilt from the config alone."""
    tb.self_check_init()
    out = {}
    for n, s in GPT(cfg).specs.items():
        init = param_init(s, cfg.seed)
        if init[0] == 'fill':
            v = np.full(s.rows * s.cols, init[1], dtype=np.float32)
        else:
            draws, bias, coef = initgen.params('normal', np.float32(init[2]))
            v = tb.seeded_np(init[1], s.rows * s.cols, draws, bias, coef).astype(np.float32)
        out[n] = v.reshape(s.rows, s.cols)
    return out


def load_dumped(cfg: ModelConfig, weights_dir) -> dict[str, np.ndarray]:
    """`kvm_run --dump '^w\\.'` output: raw little-endian float32 files RUN/buffers/w.<param>.bin."""
    out = {}
    for n, s in GPT(cfg).specs.items():
        f = Path(weights_dir) / f'w.{n}.bin'
        if not f.exists():
            raise FileNotFoundError(f'{f}: dump the weights with kvm_run --dump "^w\\."')
        out[n] = np.fromfile(f, dtype='<f4').reshape(s.rows, s.cols)
    return out


def round_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to the nearest TF32 value (10 explicit mantissa bits, ties away like cvt.rna)."""
    u = x.contiguous().view(torch.int32)
    return ((u + 0x1000) & ~0x1FFF).view(torch.float32)


def logits(cfg: ModelConfig, w: dict[str, np.ndarray], ids, mode: str) -> torch.Tensor:
    """Logits [len(ids), vocab] of a causal forward over `ids`. mode: f64 (reference) | tf32 (matmul operands rounded to TF32, f32 accumulate)."""
    dt = torch.float64 if mode == 'f64' else torch.float32
    p = {n: torch.from_numpy(np.ascontiguousarray(v)).to(dt) for n, v in w.items()}
    d, H, T = cfg.d_model, cfg.heads, len(ids)
    hd = d // H
    mm = (lambda a, b: a @ b) if mode == 'f64' else (lambda a, b: round_tf32(a.float()) @ round_tf32(b.float()))
    vec = lambda n: p[n].reshape(-1)
    inv = cfg.rope_theta ** (-2.0 * torch.arange(hd // 2, dtype=torch.float64) / hd)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
    cos, sin = ang.cos().to(dt), ang.sin().to(dt)

    def rope(x):                                                       # x [H, T, hd]
        a, b = x[..., :hd // 2], x[..., hd // 2:]
        return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)

    def norm(x, pre):
        if cfg.norm == 'rmsnorm':
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * vec(f'{pre}.g')
        return F.layer_norm(x, (d,), vec(f'{pre}.g'), vec(f'{pre}.b'), 1e-5)

    def lin(x, pre):
        y = mm(x, p[f'{pre}.w'])
        return y + vec(f'{pre}.b') if cfg.bias else y

    x = p['gpt.wte'][torch.as_tensor(list(ids), dtype=torch.long)]
    for i in range(cfg.layers):
        n = f'gpt.h{i}'
        h = norm(x, f'{n}.n1')
        split = lambda t: t.view(T, H, hd).transpose(0, 1)
        q, k, v = rope(split(lin(h, f'{n}.attn.q'))), rope(split(lin(h, f'{n}.attn.k'))), split(lin(h, f'{n}.attn.v'))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(T, d)
        x = x + lin(att, f'{n}.attn.proj')
        h = norm(x, f'{n}.n2')
        x = x + lin(F.gelu(lin(h, f'{n}.mlp.fc1'), approximate='tanh'), f'{n}.mlp.fc2')
    return mm(norm(x, 'gpt.nf'), p['gpt.head.w'])


def check_request(cfg, w, prompt, tokens, precise=False):
    """Teacher-forced check of one request's generated `tokens`. Returns a dict with per-token verdicts and PASS/FAIL."""
    P, N = len(prompt), len(tokens)
    seq = list(prompt) + list(tokens[:-1])
    ref = logits(cfg, w, seq, 'f64')[P - 1:].double()              # row i: the logits that produce tokens[i]
    emu = logits(cfg, w, seq, 'tf32')[P - 1:].double()
    E = (ref - emu).abs().max(dim=1).values                        # per step, over the vocabulary
    rows, ok = [], True
    for i, t in enumerate(tokens):
        amax = int(ref[i].argmax())
        gap = float(ref[i].max() - ref[i][t])
        top2 = ref[i].topk(2).values
        tol = 0.0 if precise else SLACK * 2 * float(E[i])
        good = t == amax or gap <= max(tol, 1e-12)
        ok &= good
        rows.append(dict(step=i + 1, executor=int(t), ref_argmax=amax, exact=t == amax, gap=gap, top1_top2_margin=float(top2[0] - top2[1]), tf32_logit_dev=float(E[i]), tie_tol=tol,
                         ref_std=float(ref[i].std()), ok=good))
    return dict(ok=bool(ok), exact=sum(r['exact'] for r in rows), steps=N, tokens=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='preset name, JSON file or JSON string (kvmfe.nn.ModelConfig.load)')
    ap.add_argument('--prompt', type=int, nargs='+', required=True)
    ap.add_argument('--tokens', type=int, nargs='+', required=True, help='the executor\'s generated tokens (cur1 ..)')
    ap.add_argument('--weights-dir', help='RUN/buffers with w.*.bin (default: the initial weights, i.e. version 0)')
    ap.add_argument('--precise', action='store_true', help='the executor used 3xTF32 tiles: tokens must equal the float64 argmax')
    a = ap.parse_args(argv)
    cfg = ModelConfig.load(a.model)
    w = load_dumped(cfg, a.weights_dir) if a.weights_dir else init_weights(cfg)
    r = check_request(cfg, w, a.prompt, a.tokens, a.precise)
    print(json.dumps(r, indent=1))
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
