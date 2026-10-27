# --- Task 3: classifier benchmark, 10-fold CV ---
import numpy as np
import mne
mne.set_log_level("ERROR")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict, GridSearchCV)
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\bci_practical"
print(">>> loading epochs")
epochs = mne.read_epochs(f"{BASE}\\epochs_clean-epo.fif", preload=True)
y = epochs.events[:, -1]
y_bin = (y == 3).astype(int)
print(">>> sanity: labels should be ~50/50:", np.bincount(y_bin, minlength=2))
assert 15 < y_bin.sum() < 30, "class counts not 50/50 — check y_bin mapping"

X = epochs.get_data()
print(">>> X:", X.shape, "| label counts:", np.bincount(y_bin, minlength=2))
print(">>> first 10 y:", y[:10])
print(">>> first 10 y_bin:", y_bin[:10])

# Sanity check: correlation of labels with trial index (should be ~0)
idx_corr = np.corrcoef(np.arange(len(y_bin)), y_bin)[0, 1]
print(f">>> label/index correlation (should be ~0): {idx_corr:+.3f}")

pipe_lda = Pipeline([
    ("csp",    CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("scaler", StandardScaler()),
    ("lda",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])
pipe_svm = Pipeline([
    ("csp",    CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", probability=True, random_state=42)),
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_lda = cross_val_score(pipe_lda, X, y_bin, cv=cv,
                          scoring="accuracy", n_jobs=1)
print(f">>> LDA 10-fold: {acc_lda.mean()*100:.2f}% "
      f"± {acc_lda.std()*100:.2f}%")

param_grid = {"svm__C": [0.1, 1, 10],
              "svm__gamma": ["scale", "auto"]}
gs = GridSearchCV(pipe_svm, param_grid, cv=cv,
                  scoring="accuracy", n_jobs=1)
gs.fit(X, y_bin)
print(">>> SVM best params:", gs.best_params_)
print(f">>> SVM best CV acc: {gs.best_score_*100:.2f}%")

acc_svm = cross_val_score(gs.best_estimator_, X, y_bin, cv=cv,
                          scoring="accuracy", n_jobs=1)
print(f">>> SVM 10-fold (tuned): {acc_svm.mean()*100:.2f}% "
      f"± {acc_svm.std()*100:.2f}%")

y_prob_lda = cross_val_predict(pipe_lda, X, y_bin, cv=cv,
                               method="predict_proba", n_jobs=1)[:, 1]
y_prob_svm = cross_val_predict(gs.best_estimator_, X, y_bin, cv=cv,
                               method="predict_proba", n_jobs=1)[:, 1]
y_pred_lda = (y_prob_lda > 0.5).astype(int)
y_pred_svm = (y_prob_svm > 0.5).astype(int)

auc_lda = roc_auc_score(y_bin, y_prob_lda)
auc_svm = roc_auc_score(y_bin, y_prob_svm)
print(f">>> AUC LDA: {auc_lda:.3f}")
print(f">>> AUC SVM: {auc_svm:.3f}")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fpr_l, tpr_l, _ = roc_curve(y_bin, y_prob_lda)
fpr_s, tpr_s, _ = roc_curve(y_bin, y_prob_svm)
axes[0].plot(fpr_l, tpr_l, label=f"LDA (AUC={auc_lda:.3f})", lw=2)
axes[0].plot(fpr_s, tpr_s, label=f"SVM (AUC={auc_svm:.3f})", lw=2)
axes[0].plot([0,1],[0,1],"k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC curves — 10-fold CV")
axes[0].legend(loc="lower right")

ConfusionMatrixDisplay(
    confusion_matrix(y_bin, y_pred_lda),
    display_labels=["left","right"],
).plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"LDA (acc={accuracy_score(y_bin,y_pred_lda)*100:.1f}%)")

ConfusionMatrixDisplay(
    confusion_matrix(y_bin, y_pred_svm),
    display_labels=["left","right"],
).plot(ax=axes[2], colorbar=False)
axes[2].set_title(f"SVM (acc={accuracy_score(y_bin,y_pred_svm)*100:.1f}%)")

plt.tight_layout()
plt.savefig(f"{BASE}\\task3_benchmark.png", dpi=130)
plt.close(fig)
print(">>> saved task3_benchmark.png")

# Final fitted pipeline for task 4 reuse
pipe_lda.fit(X, y_bin)
print(">>> DONE")