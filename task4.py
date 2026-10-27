# --- Task 4: real-time streaming buffer, latency, ITR ---
import numpy as np
import mne
mne.set_log_level("ERROR")
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\bci_practical"

# --- 1. Train 500-ms streaming model on subject 81 R04 ---
print(">>> loading raw and building 500-ms training windows")
montage = mne.channels.make_standard_montage("standard_1005")
raw = mne.io.read_raw_edf(f"{BASE}\\S081R04.edf", preload=True)
raw.rename_channels(lambda x: x.strip(". "))
raw.set_montage(montage, match_case=False, on_missing="ignore")
raw_filt = raw.copy().filter(8, 30, picks="eeg", verbose=False)

events, event_id = mne.events_from_annotations(
    raw_filt, event_id={"T0": 1, "T1": 2, "T2": 3}, verbose=False,
)
keep = {k: v for k, v in event_id.items() if k in ("T1","T2")}
events_mi = events[np.isin(events[:, 2], list(keep.values()))]
event_id_mi = {"left": keep["T1"], "right": keep["T2"]}

ep_rt = mne.Epochs(raw_filt, events_mi, event_id_mi,
                   tmin=0.0, tmax=0.5, baseline=None,
                   preload=True, picks="eeg")
X_rt = ep_rt.get_data()
y_rt = (ep_rt.events[:, -1] == event_id_mi["right"]).astype(int)
print(">>> 500-ms training set:", X_rt.shape,
      "| label counts:", np.bincount(y_rt, minlength=2))

pipe_rt = Pipeline([
    ("csp",    CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("scaler", StandardScaler()),
    ("lda",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])
pipe_rt.fit(X_rt, y_rt)
acc_rt = accuracy_score(y_rt, pipe_rt.predict(X_rt))
print(f">>> training accuracy (500-ms windows): {acc_rt:.3f}")

# --- 2. Stream 500 ms / 62.5 ms step ---
sfreq = raw.info["sfreq"]
window_samples = int(0.500 * sfreq)      # 80
step_samples   = int(0.0625 * sfreq)     # 10

stream = raw_filt.get_data()
print(">>> stream:", stream.shape,
      "= ", stream.shape[1] / sfreq, "s")

latencies, predictions = [], []
history = []

for t in range(window_samples, stream.shape[1], step_samples):
    chunk = stream[:, t - window_samples : t][np.newaxis, ...]
    t0 = time.perf_counter()
    pred = int(pipe_rt.predict(chunk)[0])
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000.0)     # ms
    history.append(pred)
    smoothed = int(np.round(np.mean(history[-3:])))  # 3-sample majority
    predictions.append(smoothed)

latencies = np.array(latencies)
predictions = np.array(predictions)
print(f">>> chunks processed: {len(latencies)}")
print(f">>> latency mean {latencies.mean():.2f} ms | "
      f"median {np.median(latencies):.2f} | "
      f"P95 {np.percentile(latencies, 95):.2f} | "
      f"max {latencies.max():.2f}")
print(f">>> drops (>15 ms): {(latencies > 15).sum()}")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(latencies, lw=0.8)
ax.axhline(15, color="red", ls="--", label="15 ms budget")
ax.set_xlabel("chunk index"); ax.set_ylabel("latency (ms)")
ax.set_title("Real-time pipeline latency per 62.5 ms step")
ax.legend(); plt.tight_layout()
plt.savefig(f"{BASE}\\task4_latency.png", dpi=120)
plt.close(fig)
print(">>> saved task4_latency.png")

# --- 3. Wolpaw ITR using CV accuracy ---
from sklearn.model_selection import cross_val_score, StratifiedKFold

n_min = int(np.bincount(y_rt, minlength=2).min())      # smaller class count
n_folds = min(5, n_min)                                 # safe fold count
print(f">>> using {n_folds}-fold CV on RT windows "
      f"(smallest class has {n_min} trials)")

cv_rt = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
P = cross_val_score(pipe_rt, X_rt, y_rt, cv=cv_rt,
                    scoring="accuracy", n_jobs=1).mean()
print(f">>> CV accuracy on RT windows: {P:.3f}")

N = 2
if 0 < P < 1:
    B = (np.log2(N) + P*np.log2(P)
         + (1-P)*np.log2((1-P)/(N-1)))
else:
    B = 0.0
T = 0.5
ITR = B * (60 / T)
print(f">>> P = {P:.3f}  B = {B:.3f} bits/decision  "
      f"T = {T} s  ITR = {ITR:.2f} bits/min")
print(">>> DONE")