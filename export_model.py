# export_model.py
import mne
mne.set_log_level("ERROR")
import pickle
import numpy as np
from pathlib import Path
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(r"C:\Users\PC\bci_practical")

print(">>> loading epochs")
epochs = mne.read_epochs(BASE / "epochs_clean-epo.fif", preload=True)
y = epochs.events[:, -1]
y_bin = (y == 3).astype(int)
X = epochs.get_data()

print(">>> fitting CSP + LDA pipeline")
pipe = Pipeline([
    ("csp",    CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("scaler", StandardScaler()),
    ("lda",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])
pipe.fit(X, y_bin)

# Also fit a 500-ms (streaming-consistent) version on S081 R04
print(">>> building 500-ms RT pipeline")
raw = mne.io.read_raw_edf(BASE / "S081R04.edf", preload=True)
raw.rename_channels(lambda x: x.strip(". "))
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage, match_case=False, on_missing="ignore")
raw_filt = raw.copy().filter(8, 30, picks="eeg", verbose=False)
events, event_id = mne.events_from_annotations(
    raw_filt, event_id={"T0":1, "T1":2, "T2":3}, verbose=False,
)
keep = {k: v for k, v in event_id.items() if k in ("T1","T2")}
events_mi = events[np.isin(events[:,2], list(keep.values()))]
event_id_mi = {"left": keep["T1"], "right": keep["T2"]}
ep_rt = mne.Epochs(raw_filt, events_mi, event_id_mi,
                   tmin=0, tmax=0.5, baseline=None,
                   preload=True, picks="eeg")
X_rt = ep_rt.get_data()
y_rt = (ep_rt.events[:, -1] == event_id_mi["right"]).astype(int)

pipe_rt = Pipeline([
    ("csp",    CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("scaler", StandardScaler()),
    ("lda",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])
pipe_rt.fit(X_rt, y_rt)

# Save model + streaming data
with open(BASE / "model.pkl", "wb") as f:
    pickle.dump({
        "pipe_rt": pipe_rt,
        "channel_names": raw.ch_names,
        "sfreq": float(raw.info["sfreq"]),
        "stream": raw_filt.get_data(),         # (64, N)
    }, f)

print(">>> saved model.pkl  (~", (BASE/"model.pkl").stat().st_size // 1024, "KB)")