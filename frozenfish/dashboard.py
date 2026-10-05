"""Dashboard: stdlib HTTP server exposing /status, /stats, /api/* and a
small built-in HTML console.  No third-party dependencies.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frozen Fish · Dashboard</title>
<style>
:root{--bg:#0a1a2b;--panel:#0e2236;--ink:#e8f1fa;--dim:#7d97ad;--ice:#0066ff;
--ok:#3ddc97;--bad:#ff5c7a;--mono:'JetBrains Mono',ui-monospace,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 Inter,system-ui,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:18px 28px;border-bottom:1px solid #14304a}
header h1{font-size:16px;margin:0;font-weight:600}
header .v{color:var(--dim);font-family:var(--mono);font-size:12px}
main{padding:24px 28px;max-width:1080px;margin:0 auto}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:28px 0 10px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #14304a;white-space:nowrap}
th{color:var(--dim);font-weight:500}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}
pre{background:var(--panel);border:1px solid #14304a;border-radius:10px;padding:14px;
overflow:auto;font-size:12px;color:#b8d2ea}
button{background:var(--ice);border:0;color:#fff;border-radius:8px;padding:7px 14px;
font-size:13px;cursor:pointer;margin:4px 6px 4px 0}
button.ghost{background:transparent;border:1px solid #1c3c5c;color:var(--ink)}
</style></head><body>
<header><h1>❄ Frozen Fish</h1><span class="v" id="v"></span></header>
<main>
<h2>Backends</h2><table id="backends"><thead><tr>
<th></th><th>name</th><th>kind</th><th>accel</th><th>util</th><th>warm</th><th>queue</th></tr>
</thead><tbody></tbody></table>
<h2>Context</h2><table id="ctx"><thead><tr><th>layer</th><th>value</th></tr></thead><tbody></tbody></table>
<h2>Workflows</h2><table id="wfs"><thead><tr><th>name</th><th>enabled</th><th>last status</th></tr>
</thead><tbody></tbody></table>
<h2>Raw status</h2><pre id="raw"></pre>
<p><button onclick="refresh()">Refresh</button><button class="ghost" onclick="probeAll()">Probe backends</button></p>
</main>
<script>
async function j(p){const r=await fetch(p);return r.json()}
async function refresh(){
 const s=await j('/status');document.getElementById('v').textContent=s.version;
 const bt=document.querySelector('#backends tbody');bt.innerHTML='';
 for(const b of s.backends){const tr=document.createElement('tr');
  tr.innerHTML=`<td><span class="dot ${b.healthy?'ok':'bad'}"></span></td><td>${b.name}</td>
   <td>${b.kind}</td><td>${b.accelerator}</td><td>${(b.utilization*100).toFixed(0)}%</td>
   <td>${b.warm_models.join(', ')||'-'}</td><td>${b.queue_depth}</td>`;bt.appendChild(tr)}
 const cs=s.context;const ct=document.querySelector('#ctx tbody');ct.innerHTML='';
 const rows=[['cag hit rate',cs.cag.hit_rate],['cag entries',cs.cag.entries],
  ['car partitions',JSON.stringify(cs.car.partitions)],['car docs',cs.car.total_docs],
  ['rag docs',cs.rag.docs],['source counts',JSON.stringify(cs.counts)]];
 for(const [k,v] of rows){const tr=document.createElement('tr');
  tr.innerHTML=`<td>${k}</td><td>${v}</td>`;ct.appendChild(tr)}
 const wt=document.querySelector('#wfs tbody');wt.innerHTML='';
 for(const n of s.workflows.loaded){const tr=document.createElement('tr');
  tr.innerHTML=`<td>${n}</td><td>${s.workflows.enabled.includes(n)?'yes':'no'}</td><td>-</td>`;
  wt.appendChild(tr)}
 document.getElementById('raw').textContent=JSON.stringify(s,null,2);
}
async function probeAll(){await j('/api/gpu/probe');refresh()}
refresh();
</script></body></html>
"""


class Dashboard:
    def __init__(self, orchestrator, host: str = "127.0.0.1", port: int = 8321,
                 auth_token: str | None = None):
        self.orch = orchestrator
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _handler(self):
        orch = self.orch
        token = self.auth_token

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # quiet
                pass

            def _auth(self) -> bool:
                if not token:
                    return True
                return self.headers.get("Authorization") == f"Bearer {token}"

            def _json(self, code: int, payload) -> None:
                body = json.dumps(payload, indent=2).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                if path == "/" or path == "/index.html":
                    if not self._auth():
                        self._json(401, {"error": "unauthorized"})
                        return
                    body = INDEX_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/status":
                    self._json(200, orch.status())
                elif path == "/stats":
                    self._json(200, orch.stats())
                elif path == "/api/gpu/probe":
                    self._json(200, {"probed": orch.gpus.probe_all()})
                elif path == "/api/mcp":
                    self._json(200, {e.name: e.info() for e in orch.mcp._entries.values()})
                elif path == "/api/health":
                    self._json(200, {"ok": orch.started})
                else:
                    self._json(404, {"error": "not found"})

        return H

    def start(self, blocking: bool = False) -> int:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        if blocking:
            try:
                self._httpd.serve_forever()
            except KeyboardInterrupt:
                self._httpd.server_close()
        else:
            self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            self._thread.start()
        return self._httpd.server_address[1]

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
