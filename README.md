# BCI Motor Imagery Lab — End-to-End CSP / LDA / SVM Pipeline

A complete, runnable pipeline for a 2-class motor-imagery Brain–Computer Interface on the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/), with a live Flask + SocketIO dashboard.

## Pipeline
EEG Acquisition -> Preprocessing (8-30 Hz + ICA) -> Feature Extraction (CSP m=4) -> LDA/SVM Classifiers -> Real-time streaming.

## Results (Subject S081, 45 trials)
* **CSP λ1/λ4:** 3.30
* **ERD C3 / C4:** -1.0% / +1.4%
* **LDA 10-fold CV:** 61.50% ± 16.29% (AUC 0.711)
* **SVM 10-fold CV:** 64.50% ± 20.91% (AUC 0.729)
* **Real-time Latency:** 0.50 ms mean (well under the 15 ms budget)

## Quick Start
1. Install: `pip install mne scikit-learn numpy scipy matplotlib flask flask-socketio jupyter nbconvert`
2. Download data:
   ```powershell
   foreach ($r in "R04","R08","R12") {
       curl.exe -L -o "S081$r.edf" "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/S081/S081$r.edf"
   }





