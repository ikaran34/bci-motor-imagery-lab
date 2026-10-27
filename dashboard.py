# dashboard.py
"""
Local motor-imagery BCI dashboard with user-selectable data input.
Run: python dashboard.py
Open: http://localhost:5000
"""
import os
import time
import pickle
import threading
import numpy as np
import mne
mne.set_log_level("ERROR")
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

BASE = Path(r"C:\Users\PC\bci_practical")
UPLOAD_DIR = BASE / "_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {".edf", ".bdf", ".fif"}

# --- Load trained model + list available EDFs ---
print(">>> loading model.pkl")
with open(BASE / "model.pkl", "rb") as f:
    payload = pickle.load(f)
pipe_rt = payload["pipe_rt"]
sfreq   = payload["sfreq"]
print(f">>> model loaded, sfreq = {sfreq} Hz")

def list_edfs():
    files = sorted([p.name for p in BASE.glob("*.edf") if p.is_file()])
    return files

# --- Flask + SocketIO ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "bci-dashboard"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024    # 100 MB upload cap
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# --- Global stream state ---
class StreamState:
    def __init__(self):
        self.stream = None          # np.ndarray (n_ch, n_samples)
        self.ch_names = None
        self.sfreq = sfreq
        self.montage_applied = False
        self.worker = None
        self.stop_event = threading.Event()
        self.user_id = None
        self.trace_channels = ["C3", "C4"]
        self.reverse_labels = False
        self.lock = threading.Lock()

state = StreamState()


def prepare_stream(edf_path: Path, trace_channels=None):
    """Load & filter an EDF, store on state."""
    print(f">>> preparing stream from {edf_path.name}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)
    raw.rename_channels(lambda x: x.strip(". "))
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    raw_filt = raw.copy().filter(8, 30, picks="eeg", verbose=False)
    data = raw_filt.get_data()

    with state.lock:
        state.stream = data
        state.ch_names = raw.ch_names
        state.sfreq = float(raw.info["sfreq"])
        state.trace_channels = trace_channels or state.trace_channels
    print(f">>> stream ready: {data.shape}, {state.sfreq} Hz")
    return {
        "n_channels": data.shape[0],
        "n_samples": data.shape[1],
        "duration_s": data.shape[1] / state.sfreq,
        "sfreq": state.sfreq,
        "ch_names": list(raw.ch_names),
    }


def stream_worker():
    """Emit one 'tick' every STEP/sfreq seconds (62.5 ms → 16 Hz)."""
    history, latencies = [], []
    idx = None
    WINDOW = int(0.500 * state.sfreq)
    STEP   = int(0.0625 * state.sfreq)
    period = STEP / state.sfreq

    while not state.stop_event.is_set():
        with state.lock:
            if state.stream is None:
                time.sleep(0.05); continue
            stream = state.stream
            if idx is None:
                idx = WINDOW
                history.clear(); latencies.clear()

        if idx >= stream.shape[1]:
            idx = WINDOW            # loop the stream
            history.clear()
            latencies.clear()

        chunk = stream[:, idx - WINDOW: idx][np.newaxis, ...]
        t_start = time.perf_counter()
        try:
            t0 = time.perf_counter()
            proba = pipe_rt.predict_proba(chunk)[0]
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            p_right = float(proba[1])
            raw_pred = int(p_right > 0.5)
        except Exception as e:
            print(">>> predict error:", e)
            p_right, raw_pred = 0.5, 0

        history.append(raw_pred)
        smoothed = int(round(np.mean(history[-3:]))) if len(history) >= 3 else raw_pred

        # Apply user-selected label orientation
        if state.reverse_labels:
            smoothed = 1 - smoothed
            p_right = 1.0 - p_right

        # Trace: pick 2 channels from user's selection
        chs = [state.ch_names.index(c) for c in state.trace_channels if c in state.ch_names]
        if len(chs) == 0:
            chs = [0, min(1, stream.shape[0]-1)]
        trace = stream[chs, idx - 40: idx].ravel().tolist()

        lat_arr = np.array(latencies[-200:]) if latencies else np.array([0.0])
        drops = int((np.array(latencies) > 15.0).sum()) if latencies else 0

        socketio.emit("tick", {
            "idx": idx - WINDOW,
            "pred": smoothed,
            "smoothed": len(history) >= 3,
            "p_right": p_right,
            "trace": trace,
            "trace_channels": state.trace_channels,
            "stats": {
                "n": len(latencies),
                "mean_ms": float(lat_arr.mean()),
                "p95_ms": float(np.percentile(lat_arr, 95)),
                "drops": drops,
            },
        })

        idx += STEP
        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.0, period - elapsed))


def start_stream():
    """Stop existing worker and start a fresh one."""
    with state.lock:
        if state.worker and state.worker.is_alive():
            state.stop_event.set()
            state.worker.join(timeout=1.0)
        state.stop_event = threading.Event()
        state.worker = threading.Thread(target=stream_worker, daemon=True)
        state.worker.start()
        print(">>> stream worker started")


# --- HTML template ---
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Motor Imagery BCI Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/socket.io@4.7.2/client-dist/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --blue: #58a6ff; --red: #f85149; --green: #3fb950; --amber: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', Roboto, sans-serif; font-size: 14px; }
  header {
    background: var(--panel); padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .status {
    padding: 4px 12px; border-radius: 12px;
    background: rgba(63,185,80,0.15); color: var(--green);
    font-size: 12px; font-weight: 600;
  }
  header .status.off { background: rgba(248,81,73,0.15); color: var(--red); }

  /* --- Input panel --- */
  .input-panel {
    background: var(--panel); padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: grid; grid-template-columns: 1fr 1fr 1fr 1fr auto;
    gap: 16px; align-items: end;
  }
  .field label {
    display: block; font-size: 11px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 4px; font-weight: 600;
  }
  .field select, .field input[type=text], .field input[type=file] {
    width: 100%; padding: 7px 10px;
    background: #0d1117; color: var(--text);
    border: 1px solid var(--border); border-radius: 5px;
    font-family: inherit; font-size: 13px;
  }
  .field input[type=file] { padding: 5px; }
  .field button {
    padding: 8px 20px; background: var(--blue);
    color: #0d1117; border: none; border-radius: 5px;
    font-weight: 600; cursor: pointer; font-size: 13px;
    transition: background 0.15s;
  }
  .field button:hover { background: #79b8ff; }
  .field button:disabled { background: #21262d; color: var(--muted); cursor: not-allowed; }
  .field.checkbox { display: flex; align-items: center; gap: 6px; }
  .field.checkbox input { width: auto; }

  .loading-text { color: var(--amber); font-size: 12px; margin-top: 4px; }
  .loading-text.error { color: var(--red); }

  .grid {
    display: grid; gap: 16px; padding: 16px;
    grid-template-columns: 2fr 1fr;
    grid-template-areas:
      "trace  pred"
      "trace  probs"
      "stats  stats";
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .card h2 {
    margin: 0 0 12px 0; font-size: 13px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px;
  }
  .trace  { grid-area: trace; height: 340px; }
  .pred   { grid-area: pred; text-align: center; padding: 24px; }
  .probs  { grid-area: probs; }
  .stats  { grid-area: stats; }
  .pred-label { font-size: 46px; font-weight: 700; margin: 16px 0 4px 0; transition: color 0.15s; }
  .pred-label.left  { color: var(--blue); }
  .pred-label.right { color: var(--red); }
  .pred-label.none  { color: var(--muted); }
  .pred-sub { color: var(--muted); font-size: 12px; }
  .prob-bar { height: 26px; background: #21262d; border-radius: 6px; overflow: hidden;
              margin: 8px 0; display: flex; }
  .prob-left  { background: var(--blue); transition: width 0.12s; }
  .prob-right { background: var(--red);  transition: width 0.12s; }
  .prob-legend { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); }
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; text-align: center; }
  .stat-value { font-size: 24px; font-weight: 700; color: var(--text); }
  .stat-label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
  .stat-value.warn { color: var(--amber); }
  .stat-value.ok { color: var(--green); }
  canvas { max-height: 100%; }
</style>
</head>
<body>

<header>
  <h1>🧠 Motor Imagery BCI · Live Dashboard</h1>
  <div class="status" id="status">● Connecting…</div>
</header>

<!-- Input panel -->
<div class="input-panel">
  <div class="field">
    <label>Data source (server)</label>
    <select id="file-select"><option>loading…</option></select>
  </div>
  <div class="field">
    <label>OR upload EDF</label>
    <input type="file" id="file-upload" accept=".edf,.bdf,.fif"/>
  </div>
  <div class="field">
    <label>Trace channels</label>
    <select id="channel-select" multiple size="1">
      <option value="C3">C3</option>
      <option value="C4">C4</option>
      <option value="Cz">Cz</option>
      <option value="Fz">Fz</option>
      <option value="Pz">Pz</option>
    </select>
  </div>
  <div class="field checkbox">
    <input type="checkbox" id="reverse-labels"/>
    <label for="reverse-labels" style="margin:0;text-transform:none;font-size:13px;color:var(--text);">
      Flip left↔right
    </label>
  </div>
  <div class="field">
    <button id="load-btn">▶ Load &amp; Stream</button>
    <div class="loading-text" id="load-status"></div>
  </div>
</div>

<div class="grid">
  <div class="card trace">
    <h2 id="trace-title">Live EEG (8–30 Hz filtered)</h2>
    <canvas id="trace"></canvas>
  </div>

  <div class="card pred">
    <h2>Prediction</h2>
    <div class="pred-label none" id="pred-label">—</div>
    <div class="pred-sub" id="pred-sub">waiting for data…</div>
    <div style="margin-top:24px">
      <h2>Class probability</h2>
      <div class="prob-bar">
        <div class="prob-left"  id="prob-left"  style="width:50%"></div>
        <div class="prob-right" id="prob-right" style="width:50%"></div>
      </div>
      <div class="prob-legend">
        <span>Left  <strong id="prob-left-pct">50%</strong></span>
        <span><strong id="prob-right-pct">50%</strong>  Right</span>
      </div>
    </div>
  </div>

  <div class="card stats">
    <h2>Streaming stats</h2>
    <div class="stats-grid">
      <div><div class="stat-value" id="s-decisions">0</div>
           <div class="stat-label">Decisions</div></div>
      <div><div class="stat-value ok" id="s-lat">0.00 ms</div>
           <div class="stat-label">Mean latency</div></div>
      <div><div class="stat-value ok" id="s-p95">0.00 ms</div>
           <div class="stat-label">P95 latency</div></div>
      <div><div class="stat-value ok" id="s-drops">0</div>
           <div class="stat-label">Drops &gt;15 ms</div></div>
    </div>
  </div>
</div>

<script>
const socket = io();

// --- Trace chart ---
const TRACE_LEN = 400;
const traceData = Array(TRACE_LEN).fill(0);
const traceChart = new Chart(
  document.getElementById('trace').getContext('2d'),
  {
    type: 'line',
    data: {
      labels: Array.from({length: TRACE_LEN}, (_, i) => i),
      datasets: [{ label: 'EEG', data: traceData,
        borderColor: '#58a6ff', borderWidth: 1.2, pointRadius: 0, tension: 0.2 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' } }
      }
    }
  }
);

socket.on('connect', () => {
  const s = document.getElementById('status');
  s.textContent = '● Connected'; s.classList.remove('off');
});
socket.on('disconnect', () => {
  const s = document.getElementById('status');
  s.textContent = '● Disconnected'; s.classList.add('off');
});

// --- Populate file list ---
async function loadFileList() {
  const r = await fetch('/files');
  const files = await r.json();
  const sel = document.getElementById('file-select');
  sel.innerHTML = '';
  files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f;
    sel.appendChild(opt);
  });
  if (files.length === 0) {
    sel.innerHTML = '<option disabled>no .edf files found</option>';
  }
}

// --- Load button handler ---
document.getElementById('load-btn').addEventListener('click', async () => {
  const btn = document.getElementById('load-btn');
  const statusEl = document.getElementById('load-status');
  statusEl.textContent = ''; statusEl.className = 'loading-text';

  const upload = document.getElementById('file-upload').files[0];
  const selectedFile = document.getElementById('file-select').value;
  const channels = Array.from(document.getElementById('channel-select').selectedOptions)
                        .map(o => o.value);
  const reverse = document.getElementById('reverse-labels').checked;

  btn.disabled = true; btn.textContent = '⏳ Loading…';

  try {
    let res;
    if (upload) {
      // upload path
      const fd = new FormData();
      fd.append('file', upload);
      fd.append('channels', JSON.stringify(channels));
      res = await fetch('/load', { method: 'POST', body: fd });
    } else {
      // server-path
      res = await fetch('/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: selectedFile, channels, reverse })
      });
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'unknown error');

    statusEl.textContent = `✓ Loaded ${data.info.duration_s.toFixed(1)} s, ` +
                           `${data.info.n_channels} ch @ ${data.info.sfreq} Hz`;
    document.getElementById('trace-title').textContent =
      `Live EEG — ${channels.join(' + ')} (8–30 Hz filtered)`;
  } catch (e) {
    statusEl.textContent = '✗ ' + e.message;
    statusEl.className = 'loading-text error';
  } finally {
    btn.disabled = false; btn.textContent = '▶ Load & Stream';
  }
});

// --- Live tick ---
socket.on('tick', (data) => {
  // Trace
  for (const v of data.trace) {
    traceData.shift();
    traceData.push(v);
  }
  traceChart.update('none');

  // Prediction
  const label = document.getElementById('pred-label');
  if (data.pred === 0) { label.textContent = 'LEFT';  label.className = 'pred-label left';  }
  else if (data.pred === 1) { label.textContent = 'RIGHT'; label.className = 'pred-label right'; }
  else { label.textContent = '—'; label.className = 'pred-label none'; }
  document.getElementById('pred-sub').textContent =
    (data.smoothed ? '3-sample smoothed' : 'raw') + ` · chunk ${data.idx}`;

  // Probabilities
  const pRight = Math.round(data.p_right * 100);
  const pLeft = 100 - pRight;
  document.getElementById('prob-left').style.width = pLeft + '%';
  document.getElementById('prob-right').style.width = pRight + '%';
  document.getElementById('prob-left-pct').textContent = pLeft + '%';
  document.getElementById('prob-right-pct').textContent = pRight + '%';

  // Stats
  document.getElementById('s-decisions').textContent = data.stats.n;
  document.getElementById('s-lat').textContent = data.stats.mean_ms.toFixed(2) + ' ms';
  document.getElementById('s-p95').textContent = data.stats.p95_ms.toFixed(2) + ' ms';
  const dropsEl = document.getElementById('s-drops');
  dropsEl.textContent = data.stats.drops;
  dropsEl.className = 'stat-value ' + (data.stats.drops > 0 ? 'warn' : 'ok');
});

loadFileList();
</script>
</body>
</html>
"""


# --- HTTP routes ---
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/files")
def files():
    return jsonify(list_edfs())


@app.route("/load", methods=["POST"])
def load():
    """Switch stream source. Accepts JSON {file, channels, reverse} or multipart upload."""
    try:
        channels = None
        reverse = False

        if request.content_type.startswith("multipart/form-data"):
            # File upload
            uploaded = request.files.get("file")
            if not uploaded or uploaded.filename == "":
                return jsonify(error="no file uploaded"), 400
            fn = secure_filename(uploaded.filename)
            if Path(fn).suffix.lower() not in ALLOWED_EXT:
                return jsonify(error=f"unsupported type: {fn}"), 400
            dest = UPLOAD_DIR / fn
            uploaded.save(str(dest))
            source_path = dest
            channels = request.form.get("channels")
            channels = json.loads(channels) if channels else None
        else:
            body = request.get_json(force=True, silent=True) or {}
            fname = body.get("file")
            if not fname:
                return jsonify(error="no 'file' specified"), 400
            source_path = BASE / fname
            if not source_path.is_file():
                return jsonify(error=f"not found: {fname}"), 404
            channels = body.get("channels")
            reverse = bool(body.get("reverse", False))

        if channels and len(channels) == 0:
            channels = None

        with state.lock:
            state.reverse_labels = reverse

        info = prepare_stream(source_path, channels)
        start_stream()
        return jsonify(ok=True, info=info)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/health")
def health():
    return jsonify(status="ok", sfreq=state.sfreq)


if __name__ == "__main__":
    import json
    # auto-load default file on boot so the dashboard isn't blank
    default = BASE / "S081R04.edf"
    if default.is_file():
        prepare_stream(default, ["C3", "C4"])
        start_stream()
    print(">>> dashboard running at http://localhost:5000")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)