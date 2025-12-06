# ResNet-18 Noodle Classification

---

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-ResNet18-red" />
  <img src="https://img.shields.io/badge/Accuracy-95%2B-brightgreen" />
  <img src="https://img.shields.io/badge/Dataset-Noodles-blue" />
</p>

---

## 📌 Overview
Train a **from‑scratch ResNet‑18** that classifies three noodle types with **95%+ test accuracy**.

**Classes:** `spaghetti`, `ramen`, `udon`

---

## 📚 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Prediction](#-prediction)
- [Output Files](#-output-files)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [FAQ](#-faq)
- [Start Training](#-start-training)

---

## 📁 Project Structure
```
project/
│
├── resnet18_model.py        # Custom ResNet‑18 implementation
├── train.py                 # Training (auto-split 90/10)
├── predict.py               # Test prediction script
├── evaluate.py              # Model evaluation
│
├── run_train.bat            # Double-click to train
├── run_predict.bat          # Double-click to predict
├── run_evaluate.bat         # Double-click to evaluate
│
└── dataset2025/
    ├── train/               # 5,280 training images
    └── test/                # 4,179 test images
```

---

## 🔧 Installation
```bash
pip install -r requirements.txt
```
Required:
- torch, torchvision
- matplotlib, seaborn
- pandas, scikit-learn
- tqdm, pillow

---

## 🚀 Training
**Method A** — Double‑click:
```
run_train.bat
```
**Method B** — CLI:
```bash
python train.py
```

Training notes:
- Auto‑splits training set into 90% train / 10% val
- No need for manual folder setup
- 2–3 hours with CUDA GPU
- Saves: `best_resnet18_noodles.pth`
- Training curves: `training_history.png`

---

## 📊 Evaluation
**Method A:**
```
run_evaluate.bat
```
**Method B:**
```bash
python evaluate.py
```

Outputs:
- Overall & per‑class accuracy
- Precision / Recall / F1 report
- Confusion matrix → `confusion_matrix.png`

---

## 🔍 Prediction
**Method A:**
```
run_predict.bat
```
**Method B:**
```bash
python predict.py
```

Output: `submission.csv`
```
ID,Target
0,2
1,0
2,1
```
Target mapping:
- `0 = spaghetti`
- `1 = ramen`
- `2 = udon`

---

## 🧾 Output Files
### After Training
- `best_resnet18_noodles.pth`
- `training_history.png`

### After Evaluation
- `confusion_matrix.png`

### After Prediction
- `submission.csv`

---

## ⚙️ Hyperparameter Tuning
Inside `train.py` → `main()`:
```python
LEARNING_RATE = 0.1
BATCH_SIZE = 32
NUM_EPOCHS = 100
patience = 15
```
Try:
- LR → 0.05 / 0.15
- Batch Size → 16 / 64
- Epochs → 150+

Data augmentation can be modified in `get_data_transforms()`.

---

## ❓ FAQ
**Q: CUDA out of memory?**
- Reduce batch size → 16 or 8

**Q: Training is slow?**
- Confirm GPU is available
- Increase batch size if VRAM allows

**Q: Accuracy < 95%?**
- Tune LR
- Increase epochs
- Stronger augmentation

**Q: Chinese characters look broken?**
- Terminal encoding issue; files are unaffected

---

## 🏁 Start Training
```bash
python train.py
```
Or double‑click:
```
run_train.bat
```

Aim for 95%+ accuracy.

