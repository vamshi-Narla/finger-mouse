

# 🖐️ Finger Mouse (Virtual Mouse using Hand Gestures)

A gesture-controlled virtual mouse that allows users to control cursor movement, clicking, and holding actions using real-time hand gestures with computer vision.

---

🚀 Features

* Move cursor using index finger
* Click using thumb + middle finger
* Hold mouse using all 5 fingers
* Smooth and real-time tracking
* Works with webcam

---

🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* PyAutoGUI

---

📦 Installation

Install required libraries:

pip install opencv-python mediapipe pyautogui

---

▶️ How to Run

python finger_mouse.py

---

⚠️ Important Note

Download the model file from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Place it in the same folder as the Python file.

---

🎮 Controls

* Index Finger → Move Cursor
* Thumb + Middle Finger → Click
* All 5 Fingers → Mouse Hold

---

📌 Future Improvements

* Right-click gesture
* Scroll functionality
* Improved accuracy

---

💼 Author

Narla Vamshi

---

⭐ Support

If you like this project, give it a ⭐ on GitHub!

### 🔧 Troubleshooting
If you see an error about `hand_landmarker.task`, run this command in your terminal:
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task