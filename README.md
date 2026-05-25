# 🤟 Task 4 — Sign Language Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-red)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![Dataset](https://img.shields.io/badge/Dataset-ASL%20Alphabet-orange)

---

## 📌 Project Overview

A custom Convolutional Neural Network (CNN) trained from scratch to recognize **American Sign Language (ASL)** hand signs. The model classifies 29 classes — letters **A–Z** plus `space`, `delete`, and `nothing` — from static images.

A **PyQt5 GUI** is included for real-time webcam detection and image upload testing. The application is **time-restricted** and only operational between **6:00 PM and 10:00 PM**.

---

## 🗂️ Project Structure

```
Task4_SignLanguage/
│
├── Task4_signlanguage_detection.ipynb   # Colab training notebook
├── asl_cnn_model.h5                     # Trained CNN model weights
├── classes.json                         # Class label mappings
├── asl_gui.py                           # PyQt5 GUI application
└── README.md                            # Project documentation
```

---

## 📦 Dataset

- **Name**: ASL Alphabet Dataset
- **Source**: [Kaggle — ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Size**: ~87,000 images
- **Classes**: 29 (A–Z + space, delete, nothing)
- **Image Size**: 200×200 px (resized to 64×64 for training)

---

## 🧠 Model Architecture

Custom CNN built from scratch using Keras — **no pre-trained models used**.

| Layer Block | Details |
|-------------|---------|
| Conv Block 1 | Conv2D(32) → BatchNorm → ReLU → MaxPool |
| Conv Block 2 | Conv2D(64) → BatchNorm → ReLU → MaxPool |
| Conv Block 3 | Conv2D(128) → BatchNorm → ReLU → MaxPool |
| Conv Block 4 | Conv2D(256) → BatchNorm → ReLU → MaxPool |
| Classifier Head | Flatten → Dense(512) → Dropout(0.5) → Dense(29, Softmax) |

**Training Configuration:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Data Augmentation: Rotation, zoom, horizontal flip, brightness

---

## 🖥️ GUI Features

Built with **PyQt5**:

- 📷 **Real-time Webcam Feed** — Live hand sign detection with predicted label overlay
- 🖼️ **Image Upload** — Upload any image to test the model
- ⏰ **Time Restriction** — App only runs between **6:00 PM – 10:00 PM**; blocked outside this window
- 📊 **Confidence Score** — Displays prediction confidence percentage

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install tensorflow keras opencv-python PyQt5 numpy pillow
```

### 2. Run the GUI

```bash
python asl_gui.py
```

> ⚠️ **Note**: The GUI will only launch between **6:00 PM and 10:00 PM**. Outside these hours, a time-restriction message will be shown.

### 3. Training (Optional — Colab)

Open `Task4_signlanguage_detection.ipynb` in Google Colab, set up your Kaggle API key, and run all cells. The trained model will be saved to your Google Drive.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~XX% |
| Validation Accuracy | ~XX% |
| Test Accuracy | ~XX% |
| Classes | 29 |

> Fill in your actual accuracy values after training.

---

## 🔒 Time Restriction Logic

```python
from datetime import datetime

now = datetime.now()
if not (18 <= now.hour < 22):
    show_error("This application is only available between 6:00 PM and 10:00 PM.")
    sys.exit()
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| TensorFlow / Keras | Model building & training |
| OpenCV | Webcam feed & image processing |
| PyQt5 | GUI framework |
| NumPy / Pillow | Data handling |
| Google Colab | Model training environment |
| Google Drive | Model storage |

---

## 👤 Author

Internship Project — Task 4  
Model built from scratch | No pre-trained weights used
