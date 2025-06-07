# VisionRPS — Play Rock, Paper, Scissors with AI 🤜🤛🖐

**VisionRPS** is a real-time Rock-Paper-Scissors game powered by AI and computer vision. You play using hand gestures in front of a webcam while the AI opponent makes its own move. The game uses MediaPipe to detect gestures, OpenCV for display and rendering, and a simple AI strategy to compete against the player.

---

## ✨ Features

* 🤜 Real-time hand gesture detection using MediaPipe
* 🤖 AI opponent with basic strategy and gesture counter logic
* 📊 Score tracking and animated UI
* ⏱ Smooth transitions between game states (gesture selection, countdown, result)
* 🔮 Colorful gradient UI, gesture icons, and visual feedback
* 🎧 Optional sound effects support via pygame

---

## 🎓 How It Works

* The player shows one of three gestures (rock, paper, scissors) in front of a webcam.
* The system stabilizes detection over a few frames to confirm the gesture.
* A countdown triggers, and then the AI makes its move.
* The winner is calculated based on classic RPS rules.
* The UI shows icons, results, and updates the score accordingly.

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/VisionRPS.git
cd VisionRPS
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
opencv-python
mediapipe
numpy
pygame
```

3. **Run the game:**

```bash
python3 rps_game.py
```

---

## 🎮 Controls

| Key   | Action                |
| ----- | --------------------- |
| Q     | Quit the game         |
| R     | Reset scores          |
| SPACE | Proceed to next round |

---

## 🌌 Gesture Guide

* 🤚 **Rock**: Closed fist
* 🖐 **Paper**: Open palm
* 🤜 **Scissors**: Two fingers (index + middle)

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Karim Dwikat**
Computer Vision Developer & Game Designer
Built using Python, OpenCV, MediaPipe, and Pygame
