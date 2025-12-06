# ResNet-18 Noodle Classification Project


---

## 🎯 Goal

Train a **ResNet‑18 model from scratch** to classify three noodle types with **95%+ accuracy** on the testing set.

**Classes:**

* Spaghetti
* Ramen
* Udon

---

## 📁 Project Structure

### Core Files

* **resnet18_model.py** — Custom ResNet‑18 implementation
* **train.py** — Training script (auto‑splits 90% train / 10% validation)
* **predict.py** — Generates predictions on the test set
* **evaluate.py** — Model evaluation script

### Executable Files (Double‑click to run)

* **run_train.bat** — Start training
* **run_predict.bat** — Run predictions
* **run_evaluate.bat** — Run evaluation

### Dataset Folders

* `dataset2025/train/` — Training images (5,280 images, 3 classes)
* `dataset2025/test/` — Testing images (4,179 images, no labels)

---

## 🚀 How to Use

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

* torch, torchvision
* matplotlib, seaborn
* pandas, scikit‑learn
* tqdm, pillow

---

### Step 2: Train the Model

**Method A:** Double‑click `run_train.bat`

**Method B:**

```bash
python train.py
```

Training notes:

* Automatically splits training data into 90% training / 10% validation
* No need to manually create `train/` or `val/`
* Training takes ~2–3 hours with GPU (CUDA recommended)
* Best model saved as: **best_resnet18_noodles.pth**
* Training curve saved as: **training_history.png**

---

### Step 3: Evaluate the Model (Optional)

**Method A:** Double‑click `run_evaluate.bat`

**Method B:**

```bash
python evaluate.py
```

Outputs:

* Overall accuracy and per‑class accuracy
* Classification report (precision, recall, F1 score)
* Confusion matrix saved as: **confusion_matrix.png**

---

### Step 4: Predict on the Test Set

**Method A:** Double‑click `run_predict.bat`

**Method B:**

```bash
python predict.py
```

Output:

* `submission.csv` containing predictions for 4,179 images

Format:

```
ID,Target
0,2
1,0
2,1
```

Where:

* `0 = spaghetti`
* `1 = ramen`
* `2 = udon`

---

## ⚠️ Important Notes

1. **GPU (CUDA) is required** for reasonable training speed.

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   If it prints `False`, training will be extremely slow.

2. **Automatic dataset splitting**

   * Uses `random_split`
   * Validation set uses no augmentation

3. **What the .bat files do**

   * Set environment variables (fix OpenMP issues)
   * Run the corresponding Python script
   * Recommended for Windows users

4. **Training Strategy**

   * Batch Size: 32
   * Learning Rate: 0.1 (Cosine Annealing)
   * Epochs: 100 (Early stopping after 15 epochs without improvement)
   * Data Augmentation: RandomCrop, Flip, Rotation, ColorJitter

---

## 🧾 Output Files

### After Training

* **best_resnet18_noodles.pth** — Best model weights (~43 MB)
* **training_history.png** — Accuracy/loss plot

### After Evaluation

* **confusion_matrix.png**

### After Prediction

* **submission.csv**

---

## 🔧 Hyperparameter Tuning

Modify these in `train.py` inside `main()`:

```python
LEARNING_RATE = 0.1      # Try 0.05 or 0.15
BATCH_SIZE = 32          # Try 16 or 64
NUM_EPOCHS = 100         # Increase to 150 if needed
patience = 15            # For early stopping
```

To adjust data augmentation, edit `get_data_transforms()`.

---

## ❓ FAQ

**Q: CUDA out of memory?**

* Reduce `BATCH_SIZE` to 16 or 8.

**Q: Training too slow?**

* Ensure you're using a GPU.
* Increase `BATCH_SIZE` if memory allows.

**Q: Accuracy below 95%?**

* Tune learning rate, increase epochs, or strengthen augmentation.

**Q: Chinese characters appear corrupted?**

* This is a terminal encoding issue. Outputs and files will not be affected.

---

## 🏁 Start Training

Run:

```bash
python train.py
```

Or double‑click:

```
run_train.bat
```
