"""
Live progress dashboard for the running experiment chain.

Reads the log files the jobs are already writing and serves a status page
that refreshes itself. Nothing here touches the runs; it is read-only.

    .venv/bin/python progress_server.py        # then open http://localhost:8765
"""
import json
import os
import re
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "outputs_logs")
PORT = 8765

ARMS = ["A raw+random", "B norm+random", "C raw+group", "D norm+group"]
SEEDS_PER_ARM = 5
MODELS = [
    ("ResNet50", "resnet50_outputs/resnet50_results.csv"),
    ("InceptionV3", "inceptionv3_outputs/inceptionv3_results.csv"),
    ("MobileViT", "mobilevit_outputs/mobilevit_results.csv"),
    ("Swin-Tiny", "swin_outputs/swin_results.csv"),
    ("ScratchCNN", "cnn_scratch_outputs/cnn_scratch_results.csv"),
    ("ScratchViT", "vit_scratch_outputs/vit_scratch_results.csv"),
]


def read(path):
    try:
        with open(os.path.join(ROOT, path), "r", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def running_pids():
    """Any python process running one of our scripts."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,etime,pcpu,command"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    hits = []
    for line in out.splitlines():
        for tag in ("provenance_control.py", "feature_extraction_pipeline.py",
                    "cnn_scratch_train.py", "vit_scratch_train.py"):
            if tag in line and "grep" not in line:
                parts = line.split(None, 3)
                hits.append({"pid": parts[0], "elapsed": parts[1],
                             "cpu": parts[2], "script": tag})
                break
    return hits


def control_state():
    log = read("outputs_logs/provenance_control.log")
    arms = []
    cur = None
    for line in log.splitlines():
        m = re.match(r"\s*ARM (\S+ \S+)\s*\|", line)
        if m:
            cur = {"name": m.group(1), "seeds": [], "mean": None}
            arms.append(cur)
            continue
        m = re.match(r"\s*seed\s+(\d+): n_train=\s*(\d+) n_test=\s*(\d+) \| "
                     r"test_acc=([\d.]+) F1=([\d.]+) kappa=([-\d.]+)", line)
        if m and cur is not None:
            cur["seeds"].append({"seed": int(m.group(1)),
                                 "acc": float(m.group(4)),
                                 "kappa": float(m.group(6))})
            continue
        m = re.match(r"\s*MEAN test_acc=([\d.]+) \+/- ([\d.]+)", line)
        if m and cur is not None:
            cur["mean"] = {"acc": float(m.group(1)), "sd": float(m.group(2))}

    done = sum(len(a["seeds"]) for a in arms)
    started = None
    m = re.search(r"provenance_control at (.+?) ---", read("outputs_logs/rerun.log"))
    if m:
        try:
            started = datetime.strptime(m.group(1).strip(), "%a %b %d %H:%M:%S %Z %Y")
        except ValueError:
            started = None

    rate = eta = None
    if started and done:
        elapsed = (datetime.now() - started).total_seconds()
        rate = elapsed / done
        eta = rate * (ARMS.__len__() * SEEDS_PER_ARM - done)

    finished = "Control finished OK" in read("outputs_logs/rerun.log") or \
               "control OK" in read("outputs_logs/rerun.log")
    return {"arms": arms, "done": done, "total": len(ARMS) * SEEDS_PER_ARM,
            "rate_s": rate, "eta_s": eta, "finished": finished}


def trackb_state():
    log = read("outputs_logs/track_b.log")
    lines = [l for l in log.splitlines() if l.startswith("---") or "Track B" in l]
    recent = [l for l in lines if "Aug 26" in l or "Aug 27" in l]
    waiting = "Waiting for" in read("outputs_logs/track_b_nohup.log") and not recent
    started = any("--- xception at" in l for l in recent)
    done = any("Track B finished" in l for l in recent)
    return {"waiting": waiting, "started": started, "finished": done,
            "lines": recent[-6:]}


def models_state():
    import csv
    out = []
    for name, path in MODELS:
        txt = read(path)
        if not txt.strip():
            out.append({"name": name, "acc": None})
            continue
        rows = list(csv.DictReader(txt.splitlines()))
        avg = [r for r in rows if str(r.get("seed", "")).upper() == "AVG"]
        try:
            acc = float(avg[0]["test_acc"]) if avg else None
            kap = float(avg[0]["kappa"]) if avg else None
        except (ValueError, KeyError, IndexError):
            acc = kap = None
        out.append({"name": name, "acc": acc, "kappa": kap})
    return out


def snapshot():
    return {
        "now": datetime.now().strftime("%H:%M:%S"),
        "procs": running_pids(),
        "control": control_state(),
        "trackb": trackb_state(),
        "models": models_state(),
    }


def fmt_dur(s):
    if s is None:
        return "—"
    s = int(s)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60} min"
    return f"{s // 3600}h {(s % 3600) // 60}m"


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Experiment Progress</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap">
<style>
:root{--bg:#0E151C;--card:#172230;--line:#2A3A4C;--tx:#E4EBF1;--dim:#8FA0AF;
--ok:#3FBF7F;--run:#4C9AFF;--wait:#8A7CC8;--warn:#E8A33D;--bad:#E9605A;
--s:"IBM Plex Sans",system-ui,sans-serif;--m:"IBM Plex Mono",monospace;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font-family:var(--s);padding:26px}
h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;font-family:var(--m);margin-bottom:22px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ok);
margin-right:7px;animation:p 1.6s ease-in-out infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
@media(prefers-reduced-motion:reduce){.dot{animation:none}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--dim);
margin:0 0 13px;font-family:var(--m);font-weight:600}
.bar{height:7px;background:#0A1017;border-radius:4px;overflow:hidden;margin:10px 0 6px}
.bar i{display:block;height:100%;background:var(--run);transition:width .6s ease}
.big{font-family:var(--m);font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;
border-top:1px solid var(--line);font-size:13.5px}
.row:first-of-type{border-top:0}
.n{font-family:var(--m);font-variant-numeric:tabular-nums}
.pill{font-family:var(--m);font-size:10px;padding:2px 8px;border-radius:20px;
text-transform:uppercase;letter-spacing:.07em}
.p-ok{background:rgba(63,191,127,.16);color:var(--ok)}
.p-run{background:rgba(76,154,255,.16);color:var(--run)}
.p-wait{background:rgba(138,124,200,.16);color:var(--wait)}
.p-pend{background:rgba(143,160,175,.13);color:var(--dim)}
.arm{padding:9px 0;border-top:1px solid var(--line)}
.arm:first-child{border-top:0}
.arm .t{display:flex;justify-content:space-between;font-size:13.5px;margin-bottom:5px}
.seeds{display:flex;gap:4px}
.seeds b{width:100%;height:5px;border-radius:2px;background:#0A1017}
.seeds b.d{background:var(--ok)}
.hint{color:var(--dim);font-size:12px;margin-top:9px;line-height:1.5}
.mono{font-family:var(--m);font-size:11.5px;color:var(--dim);line-height:1.7}
</style></head><body>
<h1><span class="dot"></span>Experiment progress</h1>
<div class="sub" id="clock">connecting…</div>
<div class="grid">
  <div class="card"><h2>Control experiment</h2><div id="ctl"></div></div>
  <div class="card"><h2>Six-model benchmark</h2><div id="mdl"></div></div>
  <div class="card"><h2>Track B</h2><div id="tb"></div></div>
  <div class="card"><h2>Live processes</h2><div id="ps"></div></div>
</div>
<script>
const fd=s=>s==null?"\\u2014":s<90?s+"s":s<5400?Math.round(s/60)+" min":
  Math.floor(s/3600)+"h "+Math.round(s%3600/60)+"m";
async function tick(){
 let d;try{d=await (await fetch('/api',{cache:'no-store'})).json()}catch(e){
   document.getElementById('clock').textContent='server stopped';return}
 document.getElementById('clock').textContent=
   d.now+"  \\u00b7  auto-refreshing every 5s";

 const c=d.control,pct=Math.round(100*c.done/c.total);
 let h=`<div class="big">${c.done}<span style="color:var(--dim);font-size:15px">/${c.total} seeds</span></div>
 <div class="bar"><i style="width:${pct}%"></i></div>
 <div class="hint">${c.finished?'Finished.':'~'+fd(c.rate_s)+' per seed \\u00b7 about '+fd(c.eta_s)+' remaining'}</div>`;
 for(const a of c.arms){
   let sd='';for(let i=0;i<5;i++)sd+=`<b class="${i<a.seeds.length?'d':''}"></b>`;
   const mv=a.mean?(a.mean.acc*100).toFixed(2)+'% \\u00b1'+(a.mean.sd*100).toFixed(2):
     a.seeds.length?a.seeds.length+'/5':'';
   h+=`<div class="arm"><div class="t"><span>${a.name}</span>
   <span class="n" style="color:${a.mean?'var(--tx)':'var(--dim)'}">${mv}</span></div>
   <div class="seeds">${sd}</div></div>`;
 }
 document.getElementById('ctl').innerHTML=h;

 document.getElementById('mdl').innerHTML=d.models.map(m=>
  `<div class="row"><span>${m.name}</span>${m.acc!=null?
   `<span class="n">${(m.acc*100).toFixed(2)}% <span style="color:var(--dim)">\\u03ba ${m.kappa.toFixed(3)}</span></span>`
   :'<span class="pill p-pend">pending</span>'}</div>`).join('');

 const t=d.trackb;
 document.getElementById('tb').innerHTML=
  `<div class="row"><span>Status</span><span class="pill ${
    t.finished?'p-ok':t.started?'p-run':t.waiting?'p-wait':'p-pend'}">${
    t.finished?'done':t.started?'running':t.waiting?'queued':'pending'}</span></div>`+
  (t.waiting?'<div class="hint">Waiting for the control experiment to finish before starting, so the two do not compete for memory.</div>':'')+
  (t.lines.length?'<div class="mono">'+t.lines.map(l=>l.replace(/</g,'&lt;')).join('<br>')+'</div>':'');

 document.getElementById('ps').innerHTML=d.procs.length?d.procs.map(p=>
  `<div class="row"><span class="n" style="font-size:12px">${p.script}</span>
   <span class="n" style="color:var(--dim)">${p.elapsed} \\u00b7 ${p.cpu}% cpu</span></div>`).join('')
  :'<div class="hint">No training process running right now.</div>';
}
tick();setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Progress dashboard on http://localhost:{PORT}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
