#!/usr/bin/env python3
"""Replay one recorded demo run as a video: Doom frames, the training loss and the executor timeline filling in, all on the run's own clock.

  python3 demo/viz/make_video.py --frames FRAMES.bin [--out assets/demo-tiny.mp4] [--gif assets/demo-tiny.gif]

Only recorded data is drawn:
  - Doom: FRAMES.bin = the device's own frames of the run (`doom_candidate/frames.bin` written by demo/check_demo.py; 700 x [64000 palette
    indices + 256 palette words]); the check proved them raw-identical to the CPU engine. Frame f appears when the last of its 8 tasks
    (tick, 4 x present, status) ended in trace.csv.gz. Frames completed faster than the output rate are skipped, none are invented.
  - Loss: observations.jsonl, `loss.<i>` of each training window, shown at the window's `t_done_us`.
  - Timeline: trace.csv.gz, same raster and colours as demo/viz/timeline.py, revealed up to the current time.
Playback is slowed down, piecewise: real [0, T1] ms plays in SEG_A_S seconds, the rest in SEG_B_S (the clock on screen states the factor).
Needs numpy, matplotlib, ffmpeg on PATH.
"""
import argparse
import gzip
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeline as tl  # noqa: E402

W, H, FB = 320, 200, 65024
SEG_A_S, SEG_B_S, FPS = 8.0, 12.0, 60
COLS = 1000
BG, FG, GRID = '#0d1117', '#e6edf3', '#30363d'


def load(trace_path, obs_path):
    if str(trace_path).endswith('.npz'):
        z = np.load(trace_path)
        fr, en = z['frame'].astype(int), z['end_us'].astype(float)
        frame_done = np.zeros(int(fr.max()) + 1)
        np.maximum.at(frame_done, fr[fr >= 0], en[fr >= 0])
        return load_obs(obs_path, frame_done)
    opener = gzip.open if str(trace_path).endswith('.gz') else open
    with opener(trace_path, 'rt', newline='') as fh:
        rows = list(csv.DictReader(fh))
    game = [r for r in rows if int(r['workload']) == 2]           # slot order = frame order, 8 tasks per frame
    game.sort(key=lambda r: int(r['slot']))
    assert len(game) % 8 == 0 and all(r['op'] == 'doom_tick' for r in game[::8]), 'unexpected Doom task layout'
    frame_done = np.array([max(float(r['end_us']) for r in game[i:i + 8]) for i in range(0, len(game), 8)])
    return load_obs(obs_path, frame_done)


def load_obs(obs_path, frame_done):
    frame_done = np.maximum.accumulate(frame_done)
    obs = [json.loads(l) for l in open(obs_path)]
    loss_pts, serve_t = [], []                                      # (t_done_us, step index, loss)
    for o in obs:
        if o['adm'].startswith('serve'):
            serve_t.append((o['t_done_us'], o['adm']))
        keys = sorted((k for k in o['buffers'] if k.startswith('loss.')), key=lambda k: int(k.split('.')[1]))
        base = int(o['adm'].split('_w')[1]) * len(keys) if keys else 0
        for j, k in enumerate(keys):
            loss_pts.append((o['t_done_us'], base + j + 1, o['buffers'][k]['values'][0]))
    return None, frame_done, np.array(loss_pts), serve_t


def video_time_to_real_us(v):
    if tl.FRAC_A >= 1.0:
        return END_US * v / (SEG_A_S + SEG_B_S)
    return tl.T1_US * v / SEG_A_S if v <= SEG_A_S else tl.T1_US + (v - SEG_A_S) / SEG_B_S * (END_US - tl.T1_US)


def palette_rgb(frames, f):
    fr = np.frombuffer(frames, dtype=np.uint8, count=FB, offset=f * FB)
    pal = fr[W * H:].reshape(256, 4)[:, [2, 1, 0]]
    return pal[fr[:W * H]].reshape(H, W, 3)


def main():
    global END_US
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', required=True)
    ap.add_argument('--trace', default=str(here / 'results/2026-11-06-demo-tiny/trace.csv.gz'))
    ap.add_argument('--obs', default=str(here / 'results/2026-11-06-demo-tiny/observations.jsonl'))
    ap.add_argument('--out', default=str(here / 'assets/demo-tiny.mp4'))
    ap.add_argument('--model', default='gpt_tiny')
    ap.add_argument('--run', default='results/2026-11-06-demo-tiny', help='result dir named in the captions')
    ap.add_argument('--footer', default='all checks pass: Doom 700/700 frames identical to the CPU engine; loss within 3.8e-6 of the CPU golden')
    ap.add_argument('--gif', default=None, help='also write a small GIF preview')
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, frame_done, loss_pts, serve_t = load(a.trace, a.obs)
    start, end, smid, cls = tl.load_trace(a.trace)
    END_US = float(end.max())
    tl.configure_axis(cls, end, END_US)
    sm_ids = sorted(set(smid.tolist()))
    busy = tl.rasterize(start, end, smid, cls, sm_ids, END_US, COLS)
    bg = np.array([int(BG[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    full = tl.compose(busy, bg)
    frames = open(a.frames, 'rb').read()
    assert len(frames) == FB * len(frame_done), 'frames.bin does not hold %d frames' % len(frame_done)
    col_t = tl.unwarp((np.arange(COLS) + 1) / COLS, END_US)         # real time at the right edge of each timeline column

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1, 1.15], left=0.06, right=0.985, top=0.9, bottom=0.2, hspace=0.42, wspace=0.18)
    ax_d, ax_l, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for ax in (ax_d, ax_l, ax_t):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    im_d = ax_d.imshow(palette_rgb(frames, 0), interpolation='nearest')
    ax_d.set_xticks([]); ax_d.set_yticks([])
    ttl = dict(color=FG, fontsize=9, loc='left')
    ax_d.set_title('Doom, device output (executor task doom_tick/present)', **ttl)
    nstep = int(loss_pts[:, 1].max()); lo, hi = float(loss_pts[:, 2].min()), float(loss_pts[:, 2].max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ax_l.set_xlim(0, nstep); ax_l.set_ylim(0 if lo > 0 and lo < 0.1 * hi else lo - pad, hi + pad)
    ax_l.set_xlabel('training step', color=FG, fontsize=8); ax_l.set_ylabel('loss', color=FG, fontsize=8)
    ax_l.set_title(a.model + ' loss; each point appears when its window completes', **ttl)
    ax_l.grid(color=GRID, lw=0.5)
    (ln,) = ax_l.plot([], [], color='#E69F00', lw=1.6)
    serve_lines = [ax_l.axvline(int(n[7:]), color='#D55E9A', lw=1, ls=':', visible=False) for _, n in serve_t]
    ax_l.text(0.98, 0.95, 'dotted: serve request answered', color='#D55E9A', fontsize=7.5, transform=ax_l.transAxes, ha='right', va='top')
    im_t = ax_t.imshow(np.tile(bg, (len(sm_ids), COLS, 1)), aspect='auto', interpolation='nearest', extent=(0, 1, len(sm_ids), 0))
    ticks_ms = tl.axis_ticks_ms(END_US)
    ax_t.set_xticks([float(tl.warp(t * 1000, END_US)) for t in ticks_ms]); ax_t.set_xticklabels([str(t) for t in ticks_ms])
    if tl.FRAC_A < 1.0:
        ax_t.axvline(tl.FRAC_A, color=FG, lw=0.8, ls=(0, (3, 3)))
    ax_t.set_xlabel('real time since first admission (ms)' + ('; scale changes at %.0f ms' % (tl.T1_US / 1000) if tl.FRAC_A < 1.0 else ', linear'), color=FG, fontsize=8)
    ax_t.set_ylabel('SM (106 workers)', color=FG, fontsize=8)
    ax_t.set_title('Executor timeline, one row per SM (all %s tasks, from the trace)' % format(len(start), ','), **ttl)
    cursor = ax_t.axvline(0, color='#ffffff', lw=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in tl.COLORS]
    leg = fig.legend(handles, [n.split(' (')[0] for n in tl.CLASSES], loc='lower center', ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.045))
    for t in leg.get_texts():
        t.set_color(FG)
    head = fig.text(0.06, 0.955, 'GPT training + inference + real Doom, one persistent kernel (A100)', color=FG, fontsize=11, weight='bold', va='center')
    clock = fig.text(0.985, 0.955, '', color='#9fb1c5', fontsize=8, ha='right', va='center')
    fig.text(0.06, 0.012, 'Recorded run: ' + a.run + ' (' + a.footer + '). '
             'Doom frames faster than 60/s are skipped; playback is slowed (factor on screen).\nNo frame, loss value or task in this video is simulated. Regenerate: demo/viz/make_video.py', color='#9fb1c5', fontsize=6.8, va='bottom')

    n_frames = int((SEG_A_S + SEG_B_S) * FPS)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', '1280x720', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '27', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    shown_frame = -1
    for i in range(n_frames):
        v = i / FPS
        t = video_time_to_real_us(v)
        f = max(0, int(np.searchsorted(frame_done, t, side='right')) - 1)
        if f != shown_frame:
            im_d.set_data(palette_rgb(frames, f)); shown_frame = f
        m = loss_pts[:, 0] <= t
        ln.set_data(loss_pts[m, 1], loss_pts[m, 2])
        for (td, _), ln_s in zip(serve_t, serve_lines):
            ln_s.set_visible(td <= t)
        k = int((col_t <= t).sum())
        cur = full.copy(); cur[:, k:, :] = bg
        im_t.set_data(cur)
        cursor.set_xdata([float(tl.warp(t, END_US))] * 2)
        if tl.FRAC_A >= 1.0:
            factor = (SEG_A_S + SEG_B_S) / (END_US / 1e6)
        else:
            factor = (SEG_A_S / (tl.T1_US / 1e6)) if t <= tl.T1_US else (SEG_B_S / ((END_US - tl.T1_US) / 1e6))
        clock.set_text('real time %7.1f ms | playback %.2gx slower | Doom frame %d/700' % (t / 1000, factor, f + 1))
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit('ffmpeg failed')
    print('wrote', a.out, Path(a.out).stat().st_size, 'bytes', file=sys.stderr)
    if a.gif:
        pal = Path(a.gif).with_suffix('.pal.png')
        vf = 'fps=10,scale=640:-1:flags=lanczos'
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', a.out, '-vf', vf + ',palettegen=max_colors=96', str(pal)])
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', a.out, '-i', str(pal), '-lavfi', vf + '[x];[x][1:v]paletteuse=dither=none', a.gif])
        pal.unlink()
        print('wrote', a.gif, Path(a.gif).stat().st_size, 'bytes', file=sys.stderr)


END_US = 0.0

if __name__ == '__main__':
    main()
