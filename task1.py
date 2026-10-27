# --- Task 1: load 9 EDFs, bandpass, ICA ocular rejection, epoching ---
import numpy as np
import mne
mne.set_log_level("WARNING")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\bci_practical"
RUNS = ["S081R04.edf", "S081R08.edf", "S081R12.edf"]
print(">>> starting")
montage = mne.channels.make_standard_montage("standard_1005")

raws = []
for fn in RUNS:
    r = mne.io.read_raw_edf(f"{BASE}\\{fn}", preload=True)
    r.rename_channels(lambda x: x.strip(". "))
    r.set_montage(montage, match_case=False, on_missing="ignore")
    r.annotations.description = np.array(
        [f"{d}_r{len(raws)}" for d in r.annotations.description]
    )
    print(f">>> {fn}: {r.info['nchan']} ch, {r.times[-1]:.1f} s")
    raws.append(r)

raw = mne.concatenate_raws(raws, verbose=False)
print(f">>> concatenated: {raw.times[-1]:.2f} s total")

# restore original T0/T1/T2 labels for the merged raw
raw.set_annotations(mne.Annotations(
    onset=raw.annotations.onset.copy(),
    duration=raw.annotations.duration.copy(),
    description=np.array(
        [d.split("_r")[0] for d in raw.annotations.description]
    ),
    orig_time=raw.annotations.orig_time,
))

# --- 1a. Bandpass 8-30 Hz (4th-order Butterworth) ---
raw_filt = raw.copy().filter(
    l_freq=8.0, h_freq=30.0,
    method="iir",
    iir_params=dict(order=4, ftype="butter", output="sos"),
    picks="eeg", verbose=False,
)
print(">>> bandpass 8-30 Hz done")

# --- 1b. ICA on 1-30 Hz copy ---
raw_ica = raw.copy().filter(l_freq=1.0, h_freq=30.0,
                            picks="eeg", verbose=False)
raw_ica.set_montage(montage, match_case=False, on_missing="ignore")
ica = mne.preprocessing.ICA(
    n_components=15, method="fastica",
    random_state=42, max_iter="auto",
)
ica.fit(raw_ica, picks="eeg")
ica.info.set_montage(montage, match_case=False, on_missing="ignore")
print(">>> ICA fitted:", ica.n_components_, "components")

fp_chs = [ch for ch in raw.ch_names if ch in ("Fp1", "Fp2")]
ica_src = ica.get_sources(raw_ica).get_data()
eog_src = raw_ica.copy().pick(fp_chs).get_data()
corrs = np.array([
    max(abs(np.corrcoef(ica_src[i], eog_src[j])[0, 1])
        for j in range(eog_src.shape[0]))
    for i in range(ica_src.shape[0])
])
eog_inds = np.where(corrs > 0.4)[0].tolist()
print(">>> ocular ICs to reject:", eog_inds,
      "| corr:", np.round(corrs[eog_inds], 2))
ica.exclude = eog_inds

raw_clean = raw_filt.copy()
ica.apply(raw_clean)
print(">>> ICA applied")

# SNR proxy: Fp variance reduction
fp_pre  = raw_filt.copy().pick(["Fp1", "Fp2"]).get_data()
fp_post = raw_clean.copy().pick(["Fp1", "Fp2"]).get_data()
snr_gain_db = 10 * np.log10(np.var(fp_pre) / np.var(fp_post))
print(f">>> Fp1/Fp2 variance reduction: {snr_gain_db:.2f} dB")

# --- 1c. Epoch 0-4 s ---
events, event_id = mne.events_from_annotations(
    raw_clean,
    event_id={"T0": 1, "T1": 2, "T2": 3},   # explicit mapping
    verbose=False,
)
print(">>> annotations:", event_id)
keep = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
events_mi = events[np.isin(events[:, 2], list(keep.values()))]
event_id_mi = {"left": keep["T1"], "right": keep["T2"]}
print(f">>> events: {len(events_mi)} total, "
      f"{np.sum(events_mi[:,2]==keep['T1'])} left, "
      f"{np.sum(events_mi[:,2]==keep['T2'])} right")

epochs = mne.Epochs(
    raw_clean, events_mi, event_id_mi,
    tmin=0.0, tmax=4.0, baseline=None,
    preload=True, picks="eeg",
)
print(">>> epochs shape:", epochs.get_data().shape)

from collections import Counter
y = epochs.events[:, -1]
print(">>> class balance:", Counter(y.tolist()))

epochs.save(f"{BASE}\\epochs_clean-epo.fif", overwrite=True)
print(">>> saved -> epochs_clean-epo.fif")

# --- 1d. Report figures ---
try:
    fig_ica = ica.plot_components(picks=range(15), show=False)
    fig_ica.suptitle("ICA components (rejected ocular marked)", fontsize=10)
    fig_ica.savefig(f"{BASE}\\task1_ica_components.png",
                    dpi=120, bbox_inches="tight")
    plt.close(fig_ica)
    print(">>> saved task1_ica_components.png")
except Exception as e:
    print(">>> skipped ICA figure:", e)

sfreq = int(raw_filt.info["sfreq"])
t0 = int(events_mi[0, 0])
pre_data,  _ = raw_filt.copy().pick(["Fp1","Fp2"])[:, t0:t0+4*sfreq]
post_data, _ = raw_clean.copy().pick(["Fp1","Fp2"])[:, t0:t0+4*sfreq]

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, data, title in zip(axes, [pre_data, post_data],
                           ["Before ICA", "After ICA"]):
    ax.plot(data[0], label="Fp1", lw=0.8)
    ax.plot(data[1], label="Fp2", lw=0.8)
    ax.set_title(title); ax.set_ylabel("uV")
    ax.legend(loc="upper right", fontsize=8)
axes[1].set_xlabel("samples @ 160 Hz")
fig.suptitle("Fp1/Fp2 segment - ICA effect on ocular artifact", fontsize=10)
plt.tight_layout()
plt.savefig(f"{BASE}\\task1_epochs_before_after.png", dpi=120)
plt.close(fig)
print(">>> saved task1_epochs_before_after.png")

print(">>> DONE")