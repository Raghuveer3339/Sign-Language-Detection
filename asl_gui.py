
import sys
import cv2
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PIL import Image

# ── Config ──────────────────────────────────────────────────
MODEL_PATH  = "asl_cnn_model.h5"
CLASSES_PATH = "classes.json"
IMG_SIZE    = 64
ACTIVE_START = 18   # 6 PM
ACTIVE_END   = 22   # 10 PM

# ── Time Check ──────────────────────────────────────────────
def is_operational():
    hour = datetime.now().hour
    return ACTIVE_START <= hour < ACTIVE_END

# ── Preprocess ──────────────────────────────────────────────
def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ── Webcam Thread ────────────────────────────────────────────
class WebcamThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_signal.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# ── Main Window ──────────────────────────────────────────────
class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Sign Language Detection")
        self.setMinimumSize(900, 650)
        self.model = load_model(MODEL_PATH)
        with open(CLASSES_PATH) as f:
            self.classes = json.load(f)
        self.webcam_thread = None
        self.init_ui()
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time_status)
        self.time_timer.start(5000)
        self.update_time_status()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Title
        title = QLabel("ASL Sign Language Detection")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; padding: 8px;")
        main_layout.addWidget(title)

        # Time status bar
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setFont(QFont("Arial", 11))
        self.time_label.setFixedHeight(36)
        self.time_label.setStyleSheet("border-radius: 6px; padding: 4px 12px;")
        main_layout.addWidget(self.time_label)

        # Content row
        content = QHBoxLayout()
        content.setSpacing(12)

        # Left — Image/Video feed
        left_panel = QVBoxLayout()
        self.video_label = QLabel("No image loaded")
        self.video_label.setFixedSize(480, 380)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            border: 2px dashed #bdc3c7;
            border-radius: 8px;
            background: #f8f9fa;
            color: #7f8c8d;
            font-size: 14px;
        """)
        left_panel.addWidget(self.video_label)

        # Buttons row
        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton(" Upload Image")
        self.upload_btn.setFixedHeight(42)
        self.upload_btn.setFont(QFont("Arial", 11))
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background: #3498db; color: white;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #2980b9; }
            QPushButton:disabled { background: #bdc3c7; }
        """)
        self.upload_btn.clicked.connect(self.upload_image)

        self.webcam_btn = QPushButton(" Start Webcam")
        self.webcam_btn.setFixedHeight(42)
        self.webcam_btn.setFont(QFont("Arial", 11))
        self.webcam_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60; color: white;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #229954; }
            QPushButton:disabled { background: #bdc3c7; }
        """)
        self.webcam_btn.clicked.connect(self.toggle_webcam)

        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.webcam_btn)
        left_panel.addLayout(btn_row)
        content.addLayout(left_panel)

        # Right — Results panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        result_title = QLabel("Prediction")
        result_title.setFont(QFont("Arial", 14, QFont.Bold))
        result_title.setStyleSheet("color: #2c3e50;")
        right_panel.addWidget(result_title)

        # Predicted letter box
        self.pred_letter = QLabel("-")
        self.pred_letter.setFont(QFont("Arial", 72, QFont.Bold))
        self.pred_letter.setAlignment(Qt.AlignCenter)
        self.pred_letter.setFixedSize(180, 160)
        self.pred_letter.setStyleSheet("""
            background: #ecf0f1;
            border-radius: 12px;
            color: #2c3e50;
            border: 2px solid #bdc3c7;
        """)
        right_panel.addWidget(self.pred_letter, alignment=Qt.AlignCenter)

        # Confidence
        conf_label = QLabel("Confidence")
        conf_label.setFont(QFont("Arial", 11))
        conf_label.setStyleSheet("color: #7f8c8d;")
        right_panel.addWidget(conf_label)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFixedHeight(22)
        self.conf_bar.setStyleSheet("""
            QProgressBar { border-radius: 6px; background: #ecf0f1; }
            QProgressBar::chunk { background: #27ae60; border-radius: 6px; }
        """)
        right_panel.addWidget(self.conf_bar)

        self.conf_text = QLabel("0%")
        self.conf_text.setFont(QFont("Arial", 13, QFont.Bold))
        self.conf_text.setAlignment(Qt.AlignCenter)
        self.conf_text.setStyleSheet("color: #27ae60;")
        right_panel.addWidget(self.conf_text)

        # Top 3 predictions
        top3_label = QLabel("Top 3 Predictions")
        top3_label.setFont(QFont("Arial", 11, QFont.Bold))
        top3_label.setStyleSheet("color: #2c3e50; margin-top: 8px;")
        right_panel.addWidget(top3_label)

        self.top3_labels = []
        for i in range(3):
            lbl = QLabel(f"{i+1}. -")
            lbl.setFont(QFont("Arial", 11))
            lbl.setStyleSheet("color: #555; padding: 4px 8px; background: #f0f0f0; border-radius: 4px;")
            right_panel.addWidget(lbl)
            self.top3_labels.append(lbl)

        right_panel.addStretch()

        # Current time
        self.clock_label = QLabel()
        self.clock_label.setFont(QFont("Arial", 10))
        self.clock_label.setStyleSheet("color: #95a5a6;")
        self.clock_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.clock_label)

        content.addLayout(right_panel)
        main_layout.addLayout(content)

        # Clock update
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

    def update_clock(self):
        now = datetime.now().strftime("%I:%M:%S %p — %A, %d %b %Y")
        self.clock_label.setText(now)

    def update_time_status(self):
        if is_operational():
            self.time_label.setText(" Model is ACTIVE (6 PM – 10 PM)")
            self.time_label.setStyleSheet("""
                background: #d5f5e3; color: #1e8449;
                border-radius: 6px; padding: 4px 12px; font-weight: bold;
            """)
            self.upload_btn.setEnabled(True)
            self.webcam_btn.setEnabled(True)
        else:
            self.time_label.setText(" Model is INACTIVE — Only available from 6 PM to 10 PM")
            self.time_label.setStyleSheet("""
                background: #fadbd8; color: #922b21;
                border-radius: 6px; padding: 4px 12px; font-weight: bold;
            """)
            self.upload_btn.setEnabled(False)
            self.webcam_btn.setEnabled(False)
            if self.webcam_thread and self.webcam_thread.running:
                self.webcam_thread.stop()

    def predict_frame(self, frame):
        processed = preprocess(frame)
        preds = self.model.predict(processed, verbose=0)[0]
        top3_idx = np.argsort(preds)[::-1][:3]
        top_class = self.classes[top3_idx[0]]
        top_conf  = preds[top3_idx[0]] * 100

        self.pred_letter.setText(top_class)
        self.conf_bar.setValue(int(top_conf))
        self.conf_text.setText(f"{top_conf:.1f}%")

        for i, idx in enumerate(top3_idx):
            self.top3_labels[i].setText(f"{i+1}. {self.classes[idx]}  ({preds[idx]*100:.1f}%)")

        # Color code confidence
        if top_conf >= 80:
            color = "#27ae60"
        elif top_conf >= 50:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{ border-radius: 6px; background: #ecf0f1; }}
            QProgressBar::chunk {{ background: {color}; border-radius: 6px; }}
        """)
        self.conf_text.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")

    def show_frame(self, frame):
        if not is_operational():
            return
        self.predict_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            480, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def upload_image(self):
        if not is_operational():
            QMessageBox.warning(self, "Inactive", "Model is only active from 6 PM to 10 PM!")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            frame = cv2.imread(path)
            if frame is not None:
                self.predict_frame(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img).scaled(
                    480, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_label.setPixmap(pixmap)

    def toggle_webcam(self):
        if not is_operational():
            QMessageBox.warning(self, "Inactive", "Model is only active from 6 PM to 10 PM!")
            return
        if self.webcam_thread and self.webcam_thread.running:
            self.webcam_thread.stop()
            self.webcam_thread = None
            self.webcam_btn.setText(" Start Webcam")
            self.webcam_btn.setStyleSheet("""
                QPushButton { background: #27ae60; color: white; border-radius: 6px; font-weight: bold; }
                QPushButton:hover { background: #229954; }
            """)
            self.video_label.setText("Webcam stopped")
        else:
            self.webcam_thread = WebcamThread()
            self.webcam_thread.frame_signal.connect(self.show_frame)
            self.webcam_thread.start()
            self.webcam_btn.setText(" Stop Webcam")
            self.webcam_btn.setStyleSheet("""
                QPushButton { background: #e74c3c; color: white; border-radius: 6px; font-weight: bold; }
                QPushButton:hover { background: #c0392b; }
            """)

    def closeEvent(self, event):
        if self.webcam_thread:
            self.webcam_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
