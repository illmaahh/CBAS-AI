# CBAS v2 — Cognitive Behavior Analysis System
### Research-Grade Multi-Layer Behavioral Intelligence Platform

---

## Files in This Folder

```
CBAS/
├── index.html              ← Open this in Chrome/Edge  (the UI)
├── server.py               ← Run this in CMD  (the backend API)
├── schemas.py              ← Data models
├── feature_engineering.py  ← Layer 2: temporal feature extraction
├── cognitive_model.py      ← Layer 3: cognitive state inference
├── reasoning_engine.py     ← Layer 4: reasoning, anomaly, prediction
├── session_manager.py      ← Session lifecycle + pipeline runner
├── requirements.txt        ← Python packages needed
├── START.bat               ← Windows double-click launcher
└── start.sh                ← Mac/Linux launcher
```

**ALL files must stay in the SAME folder. Do not move them.**

---

## Quick Start (Windows)

### Step 1 — Install Python
Download from https://python.org (3.9 or newer).
During install: ✅ check "Add Python to PATH"

### Step 2 — Install numpy
Open Command Prompt (CMD) and run:
```
pip install numpy
```

### Step 3 — Start the backend
In CMD, navigate to the CBAS folder:
```
cd C:\Users\DELL 5490\Downloads\CBAS
python server.py
```
You should see:
```
  CBAS v2  —  Native HTTP Server
  Listening on  http://localhost:8000
```

### Step 4 — Open the frontend
Double-click `index.html` to open it in Chrome or Edge.

### Step 5 — Connect backend
- Click **"BACKEND"** button in the top-right of the dashboard.
- The dot turns green: **"CONNECTED"**
- You now have full 4-layer AI analysis.

### Step 6 — Start analysis
- Click **"INITIALIZE SYSTEM"** in the camera panel.
- Allow camera permission.
- Watch all metrics come alive in real-time.

---

## Optional: Faster Server (FastAPI)

For production or high-load use, install FastAPI:
```
pip install fastapi uvicorn
```
Then run the same command — it auto-detects and uses FastAPI.

---

## What Each Layer Does

| Layer | Module | What it computes |
|-------|--------|-----------------|
| 1 — Perception | MediaPipe (browser) | 21 hand points, 468 face points, 33 pose points |
| 2 — Features | feature_engineering.py | Gaze variance, attention slope, motion entropy, blink irregularity |
| 3 — Cognitive | cognitive_model.py | Focus, Load, Stress, Fatigue scores (0-100) |
| 4 — Reasoning | reasoning_engine.py | WHY this state, anomaly z-score, Markov prediction, A-F grade |

---

## API Endpoints (when server is running)

| Method | URL | What it returns |
|--------|-----|-----------------|
| GET | http://localhost:8000/health | Server status |
| POST | http://localhost:8000/analyze | Full 4-layer analysis for one frame |
| GET | http://localhost:8000/session/ID/insight | AI behavioral insight text |
| GET | http://localhost:8000/session/ID/report | Full session report JSON |

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'backend'"**
→ You had old version. This flat build has no `backend/` folder. Fixed.

**"ModuleNotFoundError: No module named 'numpy'"**
→ Run: `pip install numpy`

**"python is not recognized"**
→ Re-install Python, tick "Add Python to PATH"

**Camera not working**
→ Use Chrome or Edge. Firefox may require extra permissions.
→ index.html must be opened from a local file path (file:// works fine).

**Backend shows OFFLINE**
→ Make sure `python server.py` is still running in CMD.
→ Check Windows Firewall isn't blocking port 8000.

---

## Performance

- Vision processing: 15–30 FPS (browser, MediaPipe)
- Backend analysis: ~0.5 ms per frame
- API send interval: every 1.5 seconds
- All biometric data stays 100% local — nothing is transmitted

---

## System Requirements

- Windows 10/11 (or Mac/Linux)
- Python 3.9+
- numpy (`pip install numpy`)
- Chrome, Edge, or Firefox (latest)
- Webcam
