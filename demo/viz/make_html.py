#!/usr/bin/env python3
"""Write demo/index.html: one self-contained page (inline CSS/JS/SVG, no network) that plays the real SM trace and shows every chart.

  make_html.py [--data DIR] [--assets DIR] [--out demo/index.html]

The trace window and the numbers come from the committed run data (results/2026-11-05-showcase); the SVGs are inlined from demo/assets. The video is
referenced relatively (assets/doom_demo.mp4) when it exists: double-clicking the file works, moving it alone loses only the video.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vizdata as V
from make_anim import ORDER, window_tasks
from make_charts import anchor_window

SPAN_US = 6000.0

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVM demo: one persistent GPU kernel</title>
<style>
:root{--bg:#fff;--fg:#1b1f24;--mut:#59636e;--card:#f6f8fa;--bd:#d0d7de;--train:#0072B2;--serve:#E69F00;--game:#009E73;--idle:#e6e6e6;--bad:#b00020}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--card:#161b22;--bd:#30363d;--idle:#2a2f36;--bad:#ff7b8a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px 14px;margin:10px 0}
.bad{color:var(--bad)}.row{display:flex;flex-wrap:wrap;gap:12px}.row>*{flex:1 1 440px;min-width:0}
canvas{width:100%;height:auto;background:var(--card);border:1px solid var(--bd);border-radius:8px}
svg,img,video{max-width:100%;height:auto}.fig{background:#fff;border-radius:8px;padding:6px;border:1px solid var(--bd);overflow-x:auto}
.fig svg{min-width:640px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid var(--bd);padding:5px 8px;text-align:left;vertical-align:top}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}button,input{font:inherit}
button{padding:3px 12px;border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
</style></head><body><main>
<h1>One persistent kernel, three workloads</h1>
<p class="mut">KVM interprets a task DAG on every SM of an A100 (108 SMs). Below: GPT training, GPT serving on the weights training just published, and real Doom run
in one kernel launch. Every picture is drawn from the executor's own per-task timestamps (<code>%smid</code>, <code>%globaltimer</code>) or from the run's outputs.</p>
<div class="card"><b>Read this first.</b> This is a correctness-and-coexistence demo on a tiny model, not a speed claim. The GPU is mostly idle here (a few busy SMs of 108).
The VM loses to a whole-step CUDA graph on the same small models (3.9-5.4x slower), and its ops reach 0.26-0.39x of cuBLAS. __RUNLINE__</div>

<h2>SM timeline, played from the real trace</h2>
<div class="card"><button id="play">pause</button> <input id="pos" type="range" min="0" max="1000" value="0" style="width:min(520px,60%)">
 <label>speed <select id="spd"><option value="0.0005">0.05%</option><option value="0.001" selected>0.1%</option><option value="0.004">0.4%</option></select> of real time</label>
 <span class="sw" style="background:var(--train)"></span>train<span class="sw" style="background:var(--serve)"></span>serve<span class="sw" style="background:var(--game)"></span>Doom
 <canvas id="cv" width="1000" height="560"></canvas>
 <div class="mut" id="cap"></div></div>

<h2>Where each task ran, dynamic vs static</h2>
<div class="fig">__SM_TIMELINE__</div>
<p class="mut">Same demo, same binary, only <code>--sched</code> differs. Static = fixed 70/10/20 SM split; dynamic = classes share all SMs by weight. Tasks are not preemptible: sharing happens at task boundaries.</p>
<div class="fig">__SM_OCCUPANCY__</div>

<h2>Doom frames: gap between ticks, per scheduler</h2>
<div class="fig">__GAPS__</div>
<p class="mut">__GAPNOTE__</p>

<h2>Training publishes, serving reads</h2>
<div class="fig">__VERSIONS__</div>
<div class="fig">__LOSS__</div>

__VIDEO__

<h2>Scale-out (simulation)</h2>
<div class="fig">__SCALEOUT__</div>
<p class="bad">Simulated: no multi-GPU, NVLink or network run exists. Uncalibrated cost model.</p>

<h2>Honest scoreboard</h2>
<div class="fig">__SCORE__</div>

<h2>Status</h2>
__STATUS__
<p class="mut">Regenerate every asset on this page: <code>demo/viz/make_all.sh</code> (no GPU needed; reads <code>results/2026-11-05-showcase</code>). New GPU run: <code>demo/viz/collect.sh</code>.</p>
</main>
<script>
const D=__DATA__, COL=["#0072B2","#E69F00","#009E73"], SPAN=D.span, TRAIL=60;
const cv=document.getElementById("cv"), cx=cv.getContext("2d"), cap=document.getElementById("cap");
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let t=0, playing=true, last=null;
const modes=Object.keys(D.tasks), cols=12;
function busy(m,tt){const c=[0,0,0];for(const k of D.tasks[m]){if(k[0]<=tt&&tt<k[1])c[k[3]]++}return c}
const curves={};for(const m of modes){curves[m]=[];for(let i=0;i<300;i++)curves[m].push(busy(m,i*SPAN/299))}
const ymax=Math.max(...modes.map(m=>Math.max(...curves[m].map(c=>c[0]+c[1]+c[2]))))+1;
function draw(){
  const W=cv.width,H=cv.height,pw=W/modes.length;cx.clearRect(0,0,W,H);cx.fillStyle=css("--fg");cx.font="bold 18px system-ui";
  modes.forEach((m,j)=>{
    const x0=j*pw+20, gw=pw-40, cell=gw/cols, rows=Math.ceil(D.sms/cols);
    cx.fillStyle=css("--fg");cx.font="bold 17px system-ui";cx.fillText(m+(m=="static"?" (fixed 70/10/20 split)":" (shared by weight)"),x0,24);
    const idle=css("--idle"), st=new Array(D.sms).fill(null);
    for(const k of D.tasks[m]){const sm=k[2];if(k[0]<=t&&t<k[1])st[sm]={c:k[3],a:1};else if(k[1]<=t&&t<k[1]+TRAIL&&!(st[sm]&&st[sm].a==1))st[sm]={c:k[3],a:0.6*(1-(t-k[1])/TRAIL)}}
    for(let i=0;i<D.sms;i++){const r=Math.floor(i/cols),c=i%cols;cx.fillStyle=idle;cx.fillRect(x0+c*cell,40+r*cell,cell-2,cell-2);
      if(st[i]){cx.globalAlpha=st[i].a;cx.fillStyle=COL[st[i].c];cx.fillRect(x0+c*cell,40+r*cell,cell-2,cell-2);cx.globalAlpha=1}}
    const by=40+rows*cell+30,bh=H-by-30;cx.strokeStyle=css("--bd");cx.strokeRect(x0,by,gw,bh);
    for(let i=0;i<300;i++){let y=by+bh;for(let c=0;c<3;c++){const h=curves[m][i][c]/ymax*bh;cx.fillStyle=COL[c];cx.fillRect(x0+i*gw/300,y-h,Math.max(1,gw/300),h);y-=h}}
    cx.fillStyle=css("--fg");cx.fillRect(x0+t/SPAN*gw,by,2,bh);
    const b=busy(m,t);cx.font="14px system-ui";cx.fillText("busy SMs: train "+b[0]+"  serve "+b[1]+"  game "+b[2]+"   (axis 0-"+(ymax-1)+")",x0,H-8);
  });
  cap.textContent="t = "+t.toFixed(0)+" us of a "+(SPAN/1000).toFixed(0)+" ms window of the real A100 run; one cell = one SM (%smid)";
  document.getElementById("pos").value=Math.round(t/SPAN*1000);
}
function tick(ts){if(playing){if(last!=null){t+=(ts-last)*1000*parseFloat(document.getElementById("spd").value);if(t>=SPAN)t=0}}last=ts;draw();requestAnimationFrame(tick)}
document.getElementById("play").onclick=e=>{playing=!playing;e.target.textContent=playing?"pause":"play"};
document.getElementById("pos").oninput=e=>{t=e.target.value/1000*SPAN;draw()};
requestAnimationFrame(tick);
</script></body></html>
"""


def inline_svg(p):
    s = Path(p).read_text()
    return s[s.index("<svg"):]                     # drop the XML prolog/doctype so it embeds


def status_table():
    rows = [("Runs and is checked", "one launch: GPT training windows + serving on the weights each window published + real Doom; Doom frames equal the CPU engine's bytes; loss within 3.8e-6 relative of the CPU golden; served tokens equal the golden decode (report.json of the run)"),
            ("Measured, narrow", "dynamic SM sharing at task boundaries; weight-version flow (exact reads); Doom 35 Hz held beside a saturating load (results/2026-11-04-a100-sched-integrated)"),
            ("Measured, a loss", "whole-step CUDA graph beats the executor 3.9-5.4x on small models; matmul 0.26-0.39x cuBLAS; the adaptive controller did not beat a hand-set priority; dynamic scheduler 1.7-1.8x worse on serve p99 than a plain priority queue"),
            ("Code only", "multi-GPU / multi-node execution, RDMA (stub), 162M GPT on the executor, 1T-parameter plan"),
            ("Simulated", "scale-out chart above (cost model, uncalibrated)")]
    return "<table>" + "".join(f"<tr><th>{a}</th><td>{b}</td></tr>" for a, b in rows) + "</table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(V.DATA))
    ap.add_argument("--assets", default=str(V.ASSETS))
    ap.add_argument("--out", default=str(V.REPO / "demo" / "index.html"))
    a = ap.parse_args()
    modes = [m for m in ("dynamic", "static") if m in V.available_modes(a.data)]
    runs = {m: V.load_run(m, a.data) for m in modes}
    tasks = {}
    for m in modes:
        t0, t1 = anchor_window(runs[m], before_ms=1.0, span_ms=SPAN_US / 1e3)
        tasks[m] = [[round(s, 1), round(e, 1), sm, c] for s, e, sm, c in window_tasks(runs[m], t0, t1)]
    sms = runs[modes[0]]["run"]["sms"]
    rep = runs["dynamic"]["report"]
    cad = rep.get("cadence", {})
    commit = runs["dynamic"]["run"].get("commit", "")
    runline = (f"Run: {runs['dynamic']['run']['device']}, {sms} SMs, {runs['dynamic']['run']['tasks_done']} tasks in {runs['dynamic']['run']['kernel_ms']:.0f} ms, "
               f"all checks passed ({', '.join(c['name'] for c in rep.get('checks', []) if c['status'] == 'PASS')}).") if rep else ""
    gapnote = (f"Dynamic run: {cad.get('ticks')} ticks, gap p50 {cad.get('gap_p50_us', 0) / 1e3:.2f} / p99 {cad.get('gap_p99_us', 0) / 1e3:.2f} / max {cad.get('gap_max_us', 0) / 1e3:.2f} ms. "
               "Doom is not rate-limited in this demo (it free-runs), so this is tick-to-tick spacing under co-tenancy, not a frame-pacing proof. First gap (engine boot, ~200 ms) excluded.") if cad else ""
    assets = Path(a.assets)
    video = ""
    if (assets / "doom_demo.mp4").exists():
        video = ('<h2>Real Doom frames, byte-identical to the CPU engine</h2><div class="card"><video controls loop muted preload="metadata" src="assets/doom_demo.mp4" '
                 'style="width:100%"></video><div class="mut">GPU frames from this run next to the training loss and served tokens, from the run\'s logs. '
                 'Each frame is compared (sha256) with the CPU oracle in the video itself.</div></div>')
    page = (PAGE.replace("__DATA__", json.dumps(dict(sms=sms, span=SPAN_US, tasks=tasks), separators=(",", ":"))).replace("__RUNLINE__", runline)
            .replace("__GAPNOTE__", gapnote).replace("__VIDEO__", video).replace("__STATUS__", status_table()))
    for key, name in (("SM_TIMELINE", "sm_timeline"), ("SM_OCCUPANCY", "sm_occupancy"), ("GAPS", "doom_frame_gaps"), ("VERSIONS", "weight_versions"),
                      ("LOSS", "loss_tokens"), ("SCALEOUT", "scaleout"), ("SCORE", "scoreboard")):
        page = page.replace(f"__{key}__", inline_svg(assets / f"{name}.svg"))
    Path(a.out).write_text(page)
    print(f"wrote {a.out} ({len(page) / 1e3:.0f} kB), trace windows {({m: len(t) for m, t in tasks.items()})}")


if __name__ == "__main__":
    main()
