# Real-Time Facial Emotion Detection (9 Emotions)

A real-time facial emotion recognition system that detects human emotions using deep learning and computer vision.  
The system captures webcam input, detects faces, and classifies emotions into **9 different categories** using a trained YOLOv8 classification model.

---

## 📌 Project Overview

This project implements a **real-time facial emotion detection system** using deep learning.  
The model is trained on a facial emotion dataset and can detect emotions directly from a webcam feed.

The system performs the following tasks:

- Detects human faces from webcam input
- Classifies facial expressions into emotions
- Displays the predicted emotion in real-time
- Supports multiple faces simultaneously

The model is trained using **YOLOv8 Classification architecture**, optimized for speed and real-time inference.

---

## 🎯 Objectives

- Build an efficient **emotion recognition system**
- Train a deep learning model for **multi-class facial expression classification**
- Implement **real-time webcam-based emotion detection**
- Evaluate model performance using **accuracy and validation metrics**

---

## 😊 Detected Emotions (9 Classes)

The model detects the following emotions:

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  
- Contempt  
- sleepy  

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

- Images organized into **train and validation folders**
- Images resized to **224 × 224 pixels**
- Data augmentation applied during training
- Class labels automatically generated from folder names

---

### 2️⃣ Model Architecture

The project uses **YOLOv8 Classification Model**.

Key features:

- Convolutional Neural Network (CNN)
- Pretrained YOLOv8 backbone
- Softmax classification layer
- Transfer learning for faster convergence

Model used:

```
yolov8n-cls.pt
```

---
## 📂 Dataset

The dataset used in this project is publicly available on Kaggle:

🔗[ https://www.kaggle.com/datasets/your-dataset-link](https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo)

> ⚠️ Due to GitHub file size limitations, the dataset is not included in this repository.

### 3️⃣ Training

Training configuration:

```
Epochs: 15
Batch Size: 128
Image Size: 160
Optimizer: Adam
Loss Function: Cross Entropy
```

Training generates:

```
runs/classify/train/
```

Inside this folder:

```
best.pt
last.pt
results.png
confusion_matrix.png
```

---

### 4️⃣ Real-Time Detection

The system performs:

1. Face detection using **OpenCV Haar Cascade**
2. Face cropping
3. Emotion classification using the trained model
4. Displaying predicted emotion on webcam frame

---

## 🛠 Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure

```
emotion-detection-9class/
│
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── sad/
│   │   └── ...
│   │
│   └── val/
│       ├── angry/
│       ├── happy/
│       ├── sad/
│       └── ...
│
├── train_model.py        # Model training script
├── predict.py            # Real-time emotion detection
├── acc.py                # Accuracy evaluation script
│
├── runs/
│   └── classify/
│       └── train/
│           └── weights/
│               └── best.pt
│
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection-9class.git
```

Move into the project folder:

```bash
cd emotion-detection-9class
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If requirements file is not available:

```bash
pip install ultralytics opencv-python numpy matplotlib torch
```

---

## 🚀 Training the Model

Run the training script:

```bash
python train_model.py
```

Training results will be saved inside:

```
runs/classify/train4/
```

The best trained model will be:

```
runs/classify/train/weights/best.pt
```

---

## 🎥 Running Real-Time Emotion Detection

Start webcam emotion detection:

```bash
python predict.py
```

The system will:

- Open webcam
- Detect faces
- Display predicted emotion in real time

Press **Q** to exit the webcam.

---

## 📊 Model Performance

Typical performance results:

- Training Accuracy: ~80–85%
- Validation Accuracy: ~65–70%

Accuracy depends on:

- Dataset balance
- Lighting conditions
- Facial visibility

---

## 📈 Results Visualization

During training, the following graphs are generated:

- Validation Accuracy vs Epochs
  <img width="1200" height="1200" alt="results" src="https://github.com/user-attachments/assets/e65e3a2d-3776-4526-9770-b0e6f040432f" />
- Confusion Matrix
  <img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/547d55eb-9d77-4dc6-9420-32e92352ac68" />



These graphs are automatically saved in:

```
runs/classify/train4/
```

---

## 🚀 Applications

This system can be used in:

- Human–Computer Interaction
- Smart classroom engagement analysis
- Driver fatigue detection
- Mental health monitoring
- AI assistants and robotics

---

## 👩‍💻 Author

**Akshara Jain**

B.Tech Student  
AI / Machine Learning Enthusiast

---

## 📜 License

This project is released under the **MIT License**.
