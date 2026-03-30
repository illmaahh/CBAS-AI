# 🧠 CBAS v2 — Cognitive Behavior Analysis System

**A Research-Grade AI System for Real-Time Human Behavior & Cognitive State Analysis**

---

## 🚀 Overview

CBAS v2 is an advanced multi-layer AI system that analyzes **human behavior in real-time** using computer vision and cognitive modeling.

It captures user activity (face, gaze, motion, posture) and processes it through a **4-layer intelligence pipeline** to generate:

* 🧠 Cognitive states (focus, stress, fatigue, load)
* 📊 Behavioral insights
* ⚠️ Anomaly detection
* 🔮 Predictive analysis

---

## 🌐 Live Demo

👉 https://cbas-ai.onrender.com

---

## 🧩 System Architecture

CBAS is built using a **4-layer AI pipeline**:

### 🔹 Layer 1 — Perception

* Captures real-time human signals using browser-based tracking
* Face landmarks, hand gestures, posture tracking

### 🔹 Layer 2 — Feature Engineering

* Extracts behavioral metrics:

  * Gaze variance
  * Motion entropy
  * Blink irregularity
  * Attention dynamics

### 🔹 Layer 3 — Cognitive Modeling

* Infers internal states:

  * Focus
  * Cognitive load
  * Stress
  * Fatigue

### 🔹 Layer 4 — Reasoning Engine

* Explains *why* a state occurred
* Detects anomalies
* Predicts future behavior
* Generates human-readable insights

---

## 🛠️ Tech Stack

* **Backend:** FastAPI (Python)
* **Server:** Uvicorn
* **Frontend:** HTML + JavaScript
* **ML/Stats:** NumPy
* **Deployment:** Render

---

## 📂 Project Structure

```
Cbas_AI/
│
├── index.html              # Frontend UI (camera + dashboard)
├── server.py               # Backend API (FastAPI + routing)
├── schemas.py              # Data models
├── feature_engineering.py  # Feature extraction
├── cognitive_model.py      # Cognitive state inference
├── reasoning_engine.py     # Behavioral reasoning & prediction
├── session_manager.py      # Session handling & pipeline
├── requirements.txt        # Dependencies
└── START.bat               # Local run (Windows)
```

---

## ⚙️ Installation (Local Setup)

### 1. Clone repository

```
git clone https://github.com/illmaahh/CBAS-AI.git
cd CBAS-AI/Cbas_AI
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Run server

```
python server.py
```

---

### 4. Open frontend

Open `index.html` in browser

---

## 🌍 Deployment

The backend is deployed on Render:

```
uvicorn server:app --host 0.0.0.0 --port $PORT
```

---

## 📡 API Endpoints

| Endpoint                | Description             |
| ----------------------- | ----------------------- |
| `/health`               | System status           |
| `/analyze`              | Run behavioral analysis |
| `/session/update`       | Update session          |
| `/session/{id}/report`  | Get session report      |
| `/session/{id}/insight` | Get insights            |
| `/sessions/stats`       | System stats            |

---

## 🔍 Key Features

✔ Real-time behavioral tracking
✔ Multi-layer cognitive inference
✔ AI-based reasoning & explanations
✔ Anomaly detection
✔ Predictive modeling
✔ Session-based analytics

---

## 💡 Use Cases

* Mental wellness monitoring
* Productivity & focus tracking
* Human-computer interaction research
* Behavioral analytics platforms
* AI-powered coaching systems

---

## ⚠️ Notes

* Camera access required for full functionality
* Works best on Chrome/Edge
* Free hosting may sleep after inactivity

---

## 👩‍💻 Author

**Ilma Rasheed**
Computer Science Undergraduate | AI & Systems Enthusiast

---

## ⭐ Future Improvements

* Deep learning integration (PyTorch)
* Dashboard visualizations (charts)
* Multi-user support
* Mobile compatibility
* Cloud database integration

---

## 📜 License

This project is for educational and research purposes.

---

✨ *Built to explore the intersection of AI, cognition, and human behavior.*

---

## System Requirements

- Windows 10/11 (or Mac/Linux)
- Python 3.9+
- numpy (`pip install numpy`)
- Chrome, Edge, or Firefox (latest)
- Webcam
