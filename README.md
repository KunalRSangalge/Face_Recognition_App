# 🧠 Real-Time Face Recognition using Metric Learning

A CPU-optimized real-time face recognition system built using **FaceNet embeddings** and cosine similarity.  
Instead of training a classifier, this project formulates identity recognition as a **metric learning problem**, enabling scalable user addition without retraining.

---

## 🚀 Features

- 📷 Live webcam face recognition  
- 📝 Register new users (live capture or image upload)  
- 🗑 Delete specific users or reset entire database  
- 💾 Persistent embedding storage using NumPy  
- ⚡ CPU-optimized inference pipeline  
- 🎯 Threshold + margin-based decision logic  

---

## 🧠 Why This is ML (Not Just CV)

This project demonstrates:

- Representation learning  
- Embedding space reasoning  
- Metric learning intuition  
- Decision threshold tuning  
- Similarity-based identity matching  
- Real-time inference optimization  

No classifier training is performed.  
Identity recognition is done by comparing embeddings in a learned feature space.

---

## 🏗 System Architecture
Webcam / Image
↓
Face Detection (OpenCV backend)
↓
FaceNet Embedding (128-D)
↓
Cosine Similarity vs Stored Embeddings
↓
Threshold + Margin Decision
↓
Identity / Unknown


---

## 🔬 Technical Approach

### 1️⃣ Embedding Extraction
- Pretrained FaceNet model via DeepFace
- Generates 128-dimensional embeddings per face
- Each embedding represents identity in metric space

### 2️⃣ Recognition Strategy
- Compute cosine similarity between live embedding and stored embeddings
- Select highest similarity score
- Apply:
  - Similarity threshold (~0.62)
  - Margin constraint between best and second-best match
- Output identity or "Unknown"

### 3️⃣ Threshold Selection
- Compared intra-class and inter-class similarity distributions
- Empirically tuned threshold to balance false acceptance and false rejection

---

## ⚡ CPU Optimization Techniques

Designed for low-end machines:

- Frame skipping (process every N frames)
- Resolution downscaling
- CLAHE lighting normalization
- Lightweight OpenCV detection backend
- Database loaded once into memory

---

## 🗂 Project Structure

```text
face-recognition-app/
│
├── app.py                  # Streamlit application
├── utils/
│   ├── storage.py          # Embedding database handling
│   └── recognition.py      # Similarity + decision logic
│
├── data/
│   ├── embeddings.npy      # Stored embeddings
│   └── names.npy           # Corresponding labels
│
├── experiments/            # Testing scripts
└── requirements.txt
```


---

## 🛠 Tech Stack

**Language**
- Python

**Libraries**
- OpenCV
- DeepFace (FaceNet)
- NumPy
- Scikit-learn
- Streamlit
- MediaPipe (experiments)

**Core Concepts**
- Representation Learning
- Metric Learning
- Cosine Similarity
- Decision Thresholding
- Real-Time ML Inference

---

## 📦 Installation

### 1️⃣ Clone Repository

```text
git clone https://github.com/KunalRSangalge/Face_Recognition_App.git
cd face-recognition-app
```

### 2️⃣ Create Virtual Environment

```text
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies
```text
pip install -r requirements.txt
```
###▶️ Run the App
```text
streamlit run app.py
```

