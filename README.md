# 🐦 Gesture-Controlled Flappy Bird

Control Flappy Bird with your hand in front of your webcam — no keyboard needed!

| Action | Gesture |
|--------|---------|
| Bird goes **UP** | Raise your hand |
| Bird goes **DOWN** | Lower your hand |

---

## 📁 Folder Structure

```
gesture-flappy-bird/
│
├── game.py              ← Main game (run this)
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🚀 Setup & Run

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the game
```bash
python game.py
```

---

## 🎮 Controls

| Input | Action |
|-------|--------|
| **Hand height** | Controls bird Y position |
| `SPACE` or `Click` | Start game (if no hand detected) |
| `R` or `Click` | Restart after game over |
| `ESC` or `Q` | Quit |

---

## ⚙️ How It Works

```
Webcam Frame
    │
    ▼
MediaPipe Hands  ──►  Detects wrist landmark (Y coordinate)
    │
    ▼
Background Thread  ──►  Sends Y norm (0.0 top → 1.0 bottom) to game
    │
    ▼
pygame Game Loop  ──►  Maps hand Y → bird Y with smooth interpolation
    │
    ▼
Collision Detection  ──►  Pipes, ground, ceiling
```

- **Hand tracking** runs in a background thread so it never blocks the game loop
- **Direct Y mapping**: hand position is smoothly interpolated to the bird's Y
- **Two windows open**: the pygame game + a webcam preview with hand skeleton overlay

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam not opening | Change `cv2.VideoCapture(0)` to `1` or `2` in `game.py` |
| Laggy hand tracking | Lower `min_detection_confidence` in `HandTracker.__init__` |
| MediaPipe install fails | Try `pip install mediapipe --pre` |
| `pygame` window not appearing | Make sure you're not running headless/SSH |

---

## 📦 Dependencies

- **pygame** — game rendering & loop
- **opencv-python** — webcam capture
- **mediapipe** — hand landmark detection
- **numpy** — numerical ops
