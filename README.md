# ECG5000 Unsupervised Anomaly Detection (Based on Russo et al.)

This project replicates and extends the methodology from **Russo et al. (2025) — Unsupervised Anomaly Detection in ECG Signals Using Denoising Autoencoders: A Comparative Study**. It reframes ECG classification as an **unsupervised anomaly detection** task using autoencoders (linear, convolutional, LSTM).

---

## 📂 Repository Structure
```
ecq5000-tsad/
├─ README.md               # Project overview (this file)
├─ requirements.txt / pyproject.toml
├─ src/
│  ├─ data/
│  │  ├─ ecg5000.py       # Download, load, preprocess, normalize
│  │  └─ utils.py         # Windowing, seed utils
│  ├─ models/
│  │  ├─ ae_linear.py
│  │  ├─ ae_conv1d.py
│  │  └─ ae_lstm.py
│  ├─ train/
│  │  ├─ trainer.py       # Training loop, logging
│  │  ├─ loss_contractive.py
│  │  └─ noise.py         # Denoising functions
│  ├─ eval/
│  │  ├─ thresholding.py  # Decision threshold from MSE distributions
│  │  ├─ metrics.py       # Accuracy, ROC-AUC, PR-AUC, F1, etc.
│  │  └─ latent.py        # PCA plots, logistic regression on latent
│  ├─ config/
│  │  ├─ linear_dae.yaml
│  │  ├─ conv_dae.yaml
│  │  ├─ lstm_dae.yaml
│  │  ├─ linear_cae.yaml
│  │  └─ linear_mixed.yaml
│  └─ cli.py              # Entry point to run experiments
├─ experiments/
│  ├─ logs/               # TensorBoard or CSV logs
│  ├─ ckpts/              # Best model checkpoints
│  └─ figures/            # Loss curves, PCA, histograms
└─ notebooks/
   ├─ 00_eda.ipynb        # Dataset exploration
   └─ 01_quick_run.ipynb  # Sanity checks
```

---

## ⚙️ Pipeline Overview

### 1. Dataset & Preprocessing
- Dataset: **ECG5000**, 5000 ECG beats, 140 time steps each.
- Split:
  - Train: 60% of **normal (Class 1)** samples.
  - Validation: 20% of normal samples.
  - Test: 20% of normal + **all abnormal (Classes 2–5)**.
- Preprocessing:
  - Normalize to zero mean & unit variance (fit on train normals).
  - No augmentation except noise injection for denoising AEs.

### 2. Models
- **Linear AE**: FC(140→32→8 latent), mirror decoder. (Best model)
- **Conv1D AE**: 1D conv layers with kernel size 9, transposed conv decoder.
- **LSTM AE**: 3-layer uni-LSTM encoder/decoder, latent=8, repeated to reconstruct sequence.

Variants:
- **DAE (denoising)**: Add Gaussian noise (σ≈0.05) to inputs.
- **CAE (contractive)**: Add Jacobian penalty λ≈1e-4.
- **Mixed**: Both denoising + contractive.

### 3. Training
- Optimizer: **Adam (lr=0.01)**.
- Batch size: **64**.
- Epochs: ≤15 with **early stopping (patience=3)**.
- Loss: MSE + optional contractive term.
- Checkpoint: save best val loss.

### 4. Thresholding
- Compute reconstruction error (MSE).
- Threshold τ = 0.5 * [(μN+σN) + (μA−σA)], where μN = mean normal, μA = mean abnormal.
- Classify: MSE < τ → normal, else → anomaly.

### 5. Evaluation
- Paper reports Accuracy, but also compute:
  - **ROC-AUC, PR-AUC, F1, Precision, Recall, Confusion matrix**.
- Latent-space analysis:
  - PCA scatter plots of latent codes.
  - Logistic regression on latent representations (expect ~93% acc for linear DAE).

### 6. Expected Outcomes
- **Denoising Linear AE** ≈ 97.7% accuracy.
- LSTM and contractive models perform worse.
- Latent space: clean separation for DAE linear, poor for LSTM/contractive.

---

## 🚀 How to Run
```bash
# install deps
pip install -r requirements.txt

# run an experiment
python -m src.cli --config src/config/linear_dae.yaml \
  --data_root ./data --save_root ./experiments
```

Outputs will be saved in `experiments/`.

---

## ✅ Checklist for Reproduction
- [ ] Confirm dataset split & preprocessing.
- [ ] Train Linear DAE, match ~97.7% accuracy.
- [ ] Log reconstruction error histograms.
- [ ] Tune CAE λ and DAE noise σ.
- [ ] Compare linear, conv, lstm.
- [ ] Perform latent-space PCA + logistic regression.
- [ ] Generate full evaluation metrics.

---

## 📌 References
- Russo, S., Silvestri, P., Tibermacine, I. E. (2025). *Unsupervised Anomaly Detection in ECG Signals Using Denoising Autoencoders: A Comparative Study*. CEUR Workshop Proceedings.

