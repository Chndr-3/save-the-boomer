"""Tiny local web UI: classify an SMS with all four models side by side.
Stdlib only. Run: python serve.py  then open http://localhost:8000
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import joblib

from train import clean

BUNDLE = joblib.load("models.joblib")
VEC = BUNDLE["vectorizer"]
MODELS = BUNDLE["models"]
TH = BUNDLE["fraud_threshold"]


def classify_all(text):
    Xv = VEC.transform([clean(text)])
    out = []
    for name, clf in MODELS.items():
        proba = clf.predict_proba(Xv)[0]
        prob = {c: float(p) for c, p in zip(clf.classes_, proba)}
        # same fraud-recall bias as train.py: flag fraud if it clears the threshold
        label = "fraud" if prob["fraud"] >= TH else max(
            (c for c in prob if c != "fraud"), key=lambda c: prob[c]
        )
        out.append({"model": name, "label": label, "prob": prob})
    return out


PAGE = """<!doctype html><meta charset=utf-8>
<title>Save the Boomer — 4-model SMS classifier</title>
<style>
 body{font:16px system-ui;max-width:820px;margin:40px auto;padding:0 16px}
 textarea{width:100%;box-sizing:border-box;padding:10px;font:inherit}
 button{margin-top:10px;padding:10px 18px;font:inherit;cursor:pointer}
 #out{margin-top:24px;display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .card{border:1px solid #ddd;border-radius:10px;padding:14px}
 .name{font-size:13px;color:#888;text-transform:uppercase;letter-spacing:.5px}
 .label{font-size:24px;font-weight:700;margin:2px 0 10px}
 .fraud{color:#c0392b}.promotion{color:#b7791f}.ham{color:#2e7d32}
 .bar{height:12px;border-radius:6px;background:#eee;margin:3px 0;overflow:hidden}
 .bar>span{display:block;height:100%;background:#999}
 .row{display:flex;justify-content:space-between;font-size:13px;color:#555}
</style>
<h1>🛡️ SMS classifier — 4 models</h1>
<textarea id=msg rows=4 placeholder="Paste an Indonesian SMS here..."></textarea><br>
<button onclick=go()>Classify with all models</button>
<div id=out></div>
<script>
async function go(){
  const text=document.getElementById('msg').value.trim();
  if(!text)return;
  const r=await fetch('/classify',{method:'POST',body:JSON.stringify({text})});
  const data=await r.json();
  document.getElementById('out').innerHTML=data.map(d=>{
    const bars=['fraud','promotion','ham'].map(c=>{
      const p=(d.prob[c]*100).toFixed(1);
      return `<div class=row><span>${c}</span><span>${p}%</span></div>
              <div class=bar><span style="width:${p}%"></span></div>`}).join('');
    return `<div class=card><div class=name>${d.model}</div>
            <div class="label ${d.label}">${d.label.toUpperCase()}</div>${bars}</div>`;
  }).join('');
}
</script>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/classify":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        text = json.loads(self.rfile.read(n))["text"]
        self._send(200, json.dumps(classify_all(text)), "application/json")

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    print("→ http://localhost:8000  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", 8000), H).serve_forever()
