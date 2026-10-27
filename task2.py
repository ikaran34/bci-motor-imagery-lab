# --- Task 2: CSP, PSD, log-var scatter ---
import numpy as np
import mne
mne.set_log_level("ERROR")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mne.decoding import CSP
from scipy.signal import welch
import warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\bci_practical"
print(">>> loading epochs")
epochs = mne.read_epochs(f"{BASE}\\epochs_clean-epo.fif", preload=True)
montage = mne.channels.make_standard_montage("standard_1005")
epochs.set_montage(montage, match_case=False, on_missing="ignore")

y = epochs.events[:, -1]
y_bin = (y == 3).astype(int)                     # 3 = right
print(">>> epochs:", epochs.get_data().shape,
      "| label counts:", np.bincount(y_bin, minlength=2))

X = epochs.get_data()

# --- 2a. CSP m=4 ---
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
X_csp = csp.fit_transform(X, y_bin)
print(">>> CSP feature shape:", X_csp.shape)

var_per_comp = np.exp(X_csp).mean(axis=0)
lam_ratio = float(var_per_comp[0] / var_per_comp[-1])
print(">>> mean variance per CSP component:", np.round(var_per_comp, 3))
print(f">>> lambda ratio (m=4): {lam_ratio:.2f}")

# --- 2b. Topographies ---
try:
    fig_p = csp.plot_patterns(epochs.info, ch_type="eeg",
                              units="Patterns (AU)", size=1.5)
    fig_p.suptitle("CSP patterns: first 2 / last 2", fontsize=10)
    fig_p.savefig(f"{BASE}\\task2_csp_patterns.png",
                  dpi=120, bbox_inches="tight")
    plt.close(fig_p)
    print(">>> saved task2_csp_patterns.png")
except Exception as e:
    print(">>> skipped patterns:", e)

try:
    fig_f = csp.plot_filters(epochs.info, ch_type="eeg",
                             units="Filters (AU)", size=1.5)
    fig_f.suptitle("CSP filters", fontsize=10)
    fig_f.savefig(f"{BASE}\\task2_csp_filters.png",
                  dpi=120, bbox_inches="tight")
    plt.close(fig_f)
    print(">>> saved task2_csp_filters.png")
except Exception as e:
    print(">>> skipped filters:", e)

# --- 2c. PSD + ERD at C3 / C4 ---
sfreq = epochs.info["sfreq"]
freqs, psd = welch(X, fs=sfreq, window="hamming",
                   nperseg=int(sfreq*1.0), noverlap=int(sfreq*0.5),
                   axis=-1)
c3 = epochs.ch_names.index("C3")
c4 = epochs.ch_names.index("C4")
alpha = (freqs >= 8) & (freqs <= 12)

psd_left  = psd[y_bin == 0].mean(axis=0)
psd_right = psd[y_bin == 1].mean(axis=0)
aL_c3, aR_c3 = psd_left[c3][alpha].mean(), psd_right[c3][alpha].mean()
aL_c4, aR_c4 = psd_left[c4][alpha].mean(), psd_right[c4][alpha].mean()
erd_c3 = (aR_c3 - aL_c3) / (aR_c3 + aL_c3) * 100
erd_c4 = (aR_c4 - aL_c4) / (aR_c4 + aL_c4) * 100
print(f">>> ERD C3 (right vs left): {erd_c3:+.1f}%")
print(f">>> ERD C4 (right vs left): {erd_c4:+.1f}%")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(freqs, 10*np.log10(psd_left[c3]),  label="C3 | left",  lw=1.5)
ax.plot(freqs, 10*np.log10(psd_right[c3]), label="C3 | right", lw=1.5)
ax.plot(freqs, 10*np.log10(psd_left[c4]),  label="C4 | left",  lw=1.5, ls="--")
ax.plot(freqs, 10*np.log10(psd_right[c4]), label="C4 | right", lw=1.5, ls="--")
ax.axvspan(8, 12, color="gold", alpha=0.15, label="alpha")
ax.axvspan(16, 24, color="skyblue", alpha=0.15, label="beta")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB)")
ax.set_title("Welch PSD - SMR channels per class")
ax.legend(fontsize=9); ax.set_xlim(5, 30)
plt.tight_layout()
plt.savefig(f"{BASE}\\task2_psd_c3_c4.png", dpi=120)
plt.close(fig)
print(">>> saved task2_psd_c3_c4.png")

# --- 2d. Log-var scatter ---
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(X_csp[y_bin==0,0], X_csp[y_bin==0,1],
           c="tab:blue", label="left", alpha=0.7, s=40)
ax.scatter(X_csp[y_bin==1,0], X_csp[y_bin==1,1],
           c="tab:red",  label="right", alpha=0.7, s=40)
ax.set_xlabel("CSP 1 (log-var)"); ax.set_ylabel("CSP 2 (log-var)")
ax.set_title("CSP log-variance separation"); ax.legend()
plt.tight_layout()
plt.savefig(f"{BASE}\\task2_csp_logvar_scatter.png", dpi=120)
plt.close(fig)
print(">>> saved task2_csp_logvar_scatter.png")
print(">>> DONE")