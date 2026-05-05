import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FingerMouse:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions.")
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        # Cursor movement settings
        self.smoothing_alpha = 0.3
        self.speed_gain_y = 1200.0
        self.speed_gain_x = 1200.0

        # Dead-zone (stop area)
        self.zone_top = 0.35
        self.zone_bottom = 0.65
        self.zone_left = 0.35
        self.zone_right = 0.65

        self.speed_buffer_y = deque(maxlen=5)
        self.speed_buffer_x = deque(maxlen=5)
        self.smoothed_speed_y = 0.0
        self.smoothed_speed_x = 0.0

        # Pinch threshold
        self.pinch_thresh_min = 0.035
        self.pinch_factor = 0.75

        # Mouse hold state
        self.holding_mouse = False

        # Click cooldown
        self.last_click_time = 0
        self.click_cooldown = 0.4

        # Finger smoothing buffers
        self.finger_buffers = {
            "thumb": deque(maxlen=5),
            "index": deque(maxlen=5),
            "middle": deque(maxlen=5),
            "ring": deque(maxlen=5),
            "pinky": deque(maxlen=5),
        }

        # Load model
        if not os.path.exists(MODEL_PATH):
            print("Model file not found. Downloading...")
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model file: {e}")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.start_time = time.time()
        self.prev_time = self.start_time

    def finger_up(self, lm, tip, pip):
        try:
            return lm[tip].y < lm[pip].y - 0.01
        except Exception:
            return False

    def detect_fingers(self, lm, handedness_name=None):
        try:
            index_up = self.finger_up(lm, 8, 6)
            middle_up = self.finger_up(lm, 12, 10)
            ring_up = self.finger_up(lm, 16, 14)
            pinky_up = self.finger_up(lm, 20, 18)

            thumb_dist = ((lm[4].x - lm[2].x) ** 2 + (lm[4].y - lm[2].y) ** 2) ** 0.5
            thumb_side = lm[4].x - lm[2].x
            thumb_up = thumb_dist > 0.05

            if handedness_name:
                if handedness_name.lower().startswith("right"):
                    thumb_up = thumb_up and thumb_side < -0.01
                elif handedness_name.lower().startswith("left"):
                    thumb_up = thumb_up and thumb_side > 0.01

            return {
                "thumb": thumb_up,
                "index": index_up,
                "middle": middle_up,
                "ring": ring_up,
                "pinky": pinky_up,
            }

        except Exception:
            return {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            }

    def smooth_fingers(self, states):
        out = {}
        for k, v in states.items():
            self.finger_buffers[k].append(bool(v))
            votes = sum(1 for x in self.finger_buffers[k] if x)
            out[k] = votes >= (len(self.finger_buffers[k]) // 2 + 1)
        return out

    def dynamic_pinch_thresh(self, lm):
        try:
            scale = ((lm[5].x - lm[17].x) ** 2 + (lm[5].y - lm[17].y) ** 2) ** 0.5
            return max(self.pinch_thresh_min, self.pinch_factor * scale)
        except Exception:
            return self.pinch_thresh_min

    def compute_velocity(self, x_norm, y_norm, dt):
        # Vertical
        if y_norm < self.zone_top:
            deflect_y = (self.zone_top - y_norm) / self.zone_top
            target_speed_y = -self.speed_gain_y * deflect_y
            status_y = "Up"
        elif y_norm > self.zone_bottom:
            deflect_y = (y_norm - self.zone_bottom) / (1.0 - self.zone_bottom)
            target_speed_y = self.speed_gain_y * deflect_y
            status_y = "Down"
        else:
            target_speed_y = 0.0
            status_y = "StopY"

        self.smoothed_speed_y = (
            (1 - self.smoothing_alpha) * self.smoothed_speed_y
            + self.smoothing_alpha * target_speed_y
        )
        self.speed_buffer_y.append(self.smoothed_speed_y)
        avg_speed_y = sum(self.speed_buffer_y) / max(1, len(self.speed_buffer_y))
        dy = int(avg_speed_y * dt)

        # Horizontal
        if x_norm < self.zone_left:
            deflect_x = (self.zone_left - x_norm) / self.zone_left
            target_speed_x = -self.speed_gain_x * deflect_x
            status_x = "Left"
        elif x_norm > self.zone_right:
            deflect_x = (x_norm - self.zone_right) / (1.0 - self.zone_right)
            target_speed_x = self.speed_gain_x * deflect_x
            status_x = "Right"
        else:
            target_speed_x = 0.0
            status_x = "StopX"

        self.smoothed_speed_x = (
            (1 - self.smoothing_alpha) * self.smoothed_speed_x
            + self.smoothing_alpha * target_speed_x
        )
        self.speed_buffer_x.append(self.smoothed_speed_x)
        avg_speed_x = sum(self.speed_buffer_x) / max(1, len(self.speed_buffer_x))
        dx = int(avg_speed_x * dt)

        if status_x.startswith("Stop") and status_y.startswith("Stop"):
            status = "Stopped"
        else:
            status = f"Moving {status_y}+{status_x}"

        return dx, dy, status

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.time()
            timestamp_ms = int((now - self.start_time) * 1000)

            dt = min(0.05, max(1e-3, now - self.prev_time))
            self.prev_time = now

            result = self.detector.detect_for_video(mp_image, timestamp_ms)

            status_action = ""
            status = "No Hand"
            dx_pixels = 0
            dy_pixels = 0

            if result and result.hand_landmarks:
                lm = result.hand_landmarks[0]

                handedness_name = None
                try:
                    if result.handedness and result.handedness[0]:
                        handedness_name = result.handedness[0][0].category_name
                except Exception:
                    handedness_name = None

                idx = lm[8]
                x_px = int(idx.x * w)
                y_px = int(idx.y * h)

                cv2.circle(frame, (x_px, y_px), 6, (0, 255, 0), -1)

                # Dead-zone lines
                y_top = int(self.zone_top * h)
                y_bottom = int(self.zone_bottom * h)
                x_left = int(self.zone_left * w)
                x_right = int(self.zone_right * w)

                cv2.line(frame, (0, y_top), (w, y_top), (0, 255, 255), 2)
                cv2.line(frame, (0, y_bottom), (w, y_bottom), (0, 255, 255), 2)
                cv2.line(frame, (x_left, 0), (x_left, h), (0, 255, 255), 2)
                cv2.line(frame, (x_right, 0), (x_right, h), (0, 255, 255), 2)

                # Finger states
                finger_states = self.detect_fingers(lm, handedness_name)
                smooth_states = self.smooth_fingers(finger_states)

                # Display finger states
                info_y = 70
                for name, up in [
                    ("Thumb", smooth_states["thumb"]),
                    ("Index", smooth_states["index"]),
                    ("Middle", smooth_states["middle"]),
                    ("Ring", smooth_states["ring"]),
                    ("Pinky", smooth_states["pinky"]),
                ]:
                    color = (0, 255, 0) if up else (0, 0, 255)
                    cv2.putText(frame, f"{name}: {'UP' if up else 'DOWN'}",
                                (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    info_y += 24

                # Check if all 5 fingers up
                all_fingers_up = all([
                    smooth_states["thumb"],
                    smooth_states["index"],
                    smooth_states["middle"],
                    smooth_states["ring"],
                    smooth_states["pinky"]
                ])

                # Hold mouse if all fingers up
                if all_fingers_up:
                    if not self.holding_mouse:
                        try:
                            pyautogui.mouseDown()
                            self.holding_mouse = True
                            status_action = "Mouse HOLD (5 Fingers)"
                        except Exception:
                            pass
                    else:
                        status_action = "Holding..."
                else:
                    if self.holding_mouse:
                        try:
                            pyautogui.mouseUp()
                            self.holding_mouse = False
                            status_action = "Mouse Released"
                        except Exception:
                            pass

                # Thumb + Middle pinch = click
                thumb_tip = lm[4]
                middle_tip = lm[12]
                dm = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
                pth = self.dynamic_pinch_thresh(lm)

                if dm < pth:
                    if time.time() - self.last_click_time > self.click_cooldown:
                        try:
                            pyautogui.click()
                            self.last_click_time = time.time()
                            status_action = "Click (Thumb+Middle)"
                        except Exception:
                            pass

                # Compute movement
                dx_pixels, dy_pixels, status = self.compute_velocity(idx.x, idx.y, dt)

            # Move cursor
            try:
                if dx_pixels != 0 or dy_pixels != 0:
                    pyautogui.moveRel(dx_pixels, dy_pixels)
            except Exception:
                pass

            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 255), 2)
            if status_action:
                cv2.putText(frame, status_action, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)

            cv2.imshow("Finger Mouse - 5 Fingers Hold", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        # Release mouse if program exits while holding
        if self.holding_mouse:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass

        self.detector.close()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        FingerMouse().run()
    except Exception as e:
        print("Error:", e)
