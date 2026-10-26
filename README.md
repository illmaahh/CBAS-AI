# CBAS — Cognitive Behavior Analysis System

CBAS is a local browser app for camera-based hand, face, and pose tracking, with a Python API for derived metrics and session reports. It is an experimental wellness and productivity aid, not a medical or psychological assessment.

## Start on Windows

1. Install Python 3.10 or newer.
2. Double-click `START.bat`. The launcher installs NumPy if needed, starts the local server, and opens the app in your browser.
3. Click **INITIALIZE SYSTEM** and allow camera access.
4. To end camera use, click **STOP CAMERA**.

Open the app at `http://127.0.0.1:8000`. Do not open `index.html` as a file: browsers restrict camera access on file pages. The webcam is requested only after you click the initialize button. Video stays in your browser; the app sends derived measurements to the local API at `127.0.0.1`.

## Run manually

```powershell
python -m pip install -r requirements.txt
python server.py --host 127.0.0.1 --port 8000 --open-browser
```

Then open `http://127.0.0.1:8000`. The standard-library web server is used by default. FastAPI and Uvicorn are optional.

## Troubleshooting

- **Camera blocked:** allow camera access for `127.0.0.1` in browser site settings and try again.
- **Camera busy:** close video-call or camera apps that may be using it.
- **Preview works but tracking does not:** the vision models load from jsDelivr. Check internet access and reload the page.
- **Port already in use:** run `python server.py --port 8001 --open-browser`.

## Project files

- `index.html` — browser UI and local vision pipeline
- `server.py` — local site and analysis API
- `schemas.py`, `feature_engineering.py`, `cognitive_model.py`, `reasoning_engine.py`, `session_manager.py` — analysis pipeline
- `requirements.txt` — Python dependencies
- `START.bat` — Windows launcher
