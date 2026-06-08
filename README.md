# 🤟 Task 4 — ASL Sign Language Detection

A custom Convolutional Neural Network trained from scratch to detect and classify American Sign Language (ASL) hand gestures across 29 classes, with a time-restricted PyQt5 GUI supporting both image upload and real-time webcam inference.

---

## Problem Statement

Sign language is the primary communication medium for the hearing-impaired community. This task builds an AI-powered system that can recognise ASL alphabet gestures (A–Z plus `space`, `delete`, and `nothing`) in real time, making communication more accessible. The model is operational only between **6 PM and 10 PM** as per the task specification.

---

## Dataset

**ASL Alphabet** — Kaggle ([grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))

| Property | Details |
|---|---|
| Total images | ~87,000 |
| Classes | 29 (A–Z + space, delete, nothing) |
| Images per class | ~3,000 |
| Image format | JPEG, 200×200 px |
| Subset used for training | 1,000 images/class → 29,000 total |
| Input size (model) | 64×64 px, RGB |

The dataset is perfectly balanced — each class contains the same number of images, eliminating class-weighting requirements.

---

## Methodology

### 1. Preprocessing & Feature Engineering

- Images resized to **64×64** pixels (memory-efficient without loss of discriminative detail at this scale)
- Normalised pixel values to **[0, 1]** range (divide by 255)
- Labels encoded with `sklearn.LabelEncoder` + `to_categorical` (one-hot for 29 classes)
- Stratified 80/10/10 train/val/test split to ensure each class is represented equally across all splits

### 2. Data Augmentation

Augmentation is applied **only to training data** via `ImageDataGenerator`:

| Augmentation | Value | Rationale |
|---|---|---|
| Rotation range | ±15° | Hand tilt variation |
| Width shift | 10% | Slight horizontal movement |
| Height shift | 10% | Slight vertical movement |
| Zoom range | 10% | Distance from camera variation |
| Shear range | 10% | Wrist angle variation |
| Horizontal flip | **Disabled** | ASL signs are hand-specific; mirroring changes meaning |

### 3. Model Architecture — Custom CNN from Scratch

No pre-trained weights are used. The architecture was designed specifically for ASL's 29-class fine-grained classification problem.

```
Input: 64×64×3
  ├── Conv Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  ├── Conv Block 4: Conv2D(256) → BN → Conv2D(256) → BN → GlobalAvgPool → Dropout(0.4)
  ├── Dense(512) → BN → Dropout(0.5)
  ├── Dense(256) → Dropout(0.3)
  └── Dense(29, softmax)
```

**Why this architecture over alternatives:**

| Model | Params | Expected Accuracy | Notes |
|---|---|---|---|
| Flat MLP (baseline) | ~2M | ~40–55% | No spatial awareness |
| Simple CNN (2 blocks) | ~0.5M | ~70–80% | Faster, less discriminative |
| **Our CNN (4 blocks)** | **~3.5M** | **~90–95%** | **Chosen** |
| VGG16 pretrained | 138M | ~97–99% | Not allowed (task requirement) |

Key design decisions:
- **Progressive filter doubling (32→64→128→256):** shallow blocks detect edges and textures; deeper blocks detect full hand shapes and finger configurations
- **GlobalAveragePooling** in Block 4 instead of Flatten reduces overfitting on fine-grained 29-class data
- **BatchNorm after every Conv:** stabilises gradients and acts as regulariser
- **No horizontal flip:** mirroring an ASL sign changes its visual structure and would add noise

### 4. Training

| Parameter | Value |
|---|---|
| Optimiser | Adam (lr=0.001) |
| Loss | Categorical Cross-entropy |
| Epochs | Up to 30 (EarlyStopping) |
| Batch size | 64 |
| EarlyStopping | patience=8, monitor=val_accuracy |
| ReduceLROnPlateau | factor=0.5, patience=4, min_lr=1e-6 |
| ModelCheckpoint | Saves best val_accuracy |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~92–95% (update after training) |
| Test Loss | ~0.15–0.25 (update after training) |

Per-class metrics and the full confusion matrix are generated in the notebook (Cells 10 & 11).

---

## Visual Outputs

All plots are saved to Google Drive and included in the notebook outputs:

| Plot | Description |
|---|---|
| `sample_images.png` | 4×8 grid of sample images for all 29 classes |
| `eda_analysis.png` | 4-panel EDA: class distribution, loaded subset balance, RGB pixel intensity histogram, class type pie chart |
| `training_history.png` | Accuracy and loss curves (train vs validation) |
| `confusion_matrix.png` | 29×29 Seaborn heatmap of predictions vs ground truth |

---

## GUI — PyQt5 Desktop Application

The GUI (`asl_gui.py`) is designed to run locally after downloading the trained model:

**Features:**
- **Time restriction:** Model is active only from **6 PM to 10 PM**. All buttons are disabled outside this window, with a live status indicator.
- **Image upload mode:** Load any `.png/.jpg/.bmp` image for instant classification.
- **Real-time webcam mode:** Live frame-by-frame ASL detection via webcam thread.
- **Prediction panel:** Displays predicted letter in large font, confidence progress bar, and top-3 predictions with probabilities.
- **Live clock:** Updates every second showing current time.

**To run locally:**
```bash
# Download from Drive: asl_cnn_model.h5, classes.json, asl_gui.py
pip install tensorflow pyqt5 opencv-python pillow
python asl_gui.py
# NOTE: Only works between 6 PM and 10 PM
```

---

## Project Structure

```
Task4_Sign_Language_Detection/
├── Task4_SignLanguage_UPDATED.ipynb   ← Main Colab notebook
├── asl_gui.py                         ← PyQt5 GUI (run locally)
├── asl_cnn_model.h5                   ← Trained model weights
├── classes.json                       ← Class label mapping
├── sample_images.png                  ← Sample grid of 29 classes
├── eda_analysis.png                   ← EDA visualisations
├── training_history.png               ← Training curves
└── confusion_matrix.png               ← Confusion matrix heatmap
```

---

## How to Run (Google Colab)

1. Open `Task4_SignLanguage_UPDATED.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run **Cell 1** — mounts Drive and downloads the ASL dataset from Kaggle
4. Run **Cells 2–5** — installs libraries, verifies dataset, loads and preprocesses images
5. Run **Cell 5B** — generates EDA plots
6. Run **Cells 6–7** — splits data and builds the CNN
7. Run **Cell 7B** — prints model selection justification and baseline comparison
8. Run **Cells 8–11** — trains the model, plots history, evaluates, and generates confusion matrix
9. Run **Cell 12** — saves the PyQt5 GUI script to Drive
10. Run **Cell 13** — upload your own image to test the trained model

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| TensorFlow / Keras | Model building and training |
| OpenCV | Image reading and preprocessing |
| NumPy | Array operations |
| Matplotlib | Training curves, sample images |
| Seaborn | Confusion matrix heatmap, EDA plots |
| scikit-learn | Label encoding, train/val/test split, metrics |
| PyQt5 | Desktop GUI |
| Google Colab + GPU | Training environment |
| Google Drive | Dataset and model storage |
| Kaggle API | Dataset download |
