#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Green Target Tracker + Servo Trigger
- Uses USB camera (default: /dev/video0 → index 0)
- Detects green objects crossing a horizontal line
- Fires servo (GPIO 18) when crossing occurs
"""

import cv2
import numpy as np
import time
import sys

# =============================
#        CONFIGURATION
# =============================

# 🔧 CAMERA
CAMERA_INDEX = 0          # /dev/video0 — change if needed

# 🎯 DETECTION LINE (will auto-adjust to frame height)
LINE_Y_REL = 0.70         # 70% down the frame (e.g., y = 0.7 * height)
LINE_THICKNESS = 10
SEGMENT_MARGIN = 0.2     # Keep line centered, covering middle 60%: [20%, 80%] of width

# 🟢 GREEN DETECTION (HSV)
GREEN_LOW  = np.array([40,  50,  50])
GREEN_HIGH = np.array([80, 255, 255])
MIN_AREA = 1500           # Minimum contour area (pixels)
ASPECT_MAX = 4.0          # Max w/h or h/w ratio for blob
FILL_MIN = 0.3            # Min area/(w*h) to reject noise

# 🧠 TRACKING
MAX_DIST = 80             # Max pixel distance to match track
MAX_MISSES = 5
MIN_FRAMES_TO_CONFIRM = 2

# ⚙️ SERVO (GPIO 18 = physical Pin 12)
SERVO_PIN = 18
SERVO_FREQ = 50           # Hz
SERVO_MIN_US = 500
SERVO_MAX_US = 2500
REST_ANGLE = 90
FIRE_ANGLE = 30
FIRE_HOLD = 0.12
RETURN_HOLD = 0.08

# =============================
#        SERVO CONTROLLER
# =============================

class ServoController:
    def __init__(self, pin=SERVO_PIN, freq=SERVO_FREQ,
                 min_us=SERVO_MIN_US, max_us=SERVO_MAX_US, debug=False):
        self.pin = pin
        self.freq = freq
        self.min_us = min_us
        self.max_us = max_us
        self.debug = debug
        self.current_angle = REST_ANGLE
        self._dry_run = False

        try:
            import lgpio
            self.lgpio = lgpio
            self.chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.chip, self.pin)
            print("[✅ SERVO] Hardware PWM active (lgpio)")
            self.set_angle(self.current_angle, hold=0.1)
        except Exception as e:
            print(f"[⚠️ SERVO] lgpio failed → dry-run mode only ({e})")
            self._dry_run = True

    def _angle_to_duty(self, angle):
        angle = np.clip(angle, 0, 180)
        pulse_us = self.min_us + (angle / 180.0) * (self.max_us - self.min_us)
        period_us = 1_000_000 / self.freq
        duty = (pulse_us / period_us) * 100.0
        return duty

    def set_angle(self, angle, hold=0.05):
        angle = float(np.clip(angle, 0, 180))
        self.current_angle = angle
        if self._dry_run:
            if self.debug:
                print(f"[DRY] servo → {angle:.1f}° (hold {hold}s)")
            time.sleep(hold)
            return

        try:
            duty = self._angle_to_duty(angle)
            res = self.lgpio.tx_pwm(self.chip, self.pin, self.freq, duty)
            if res < 0:
                print(f"[❌ SERVO] tx_pwm error: {res}")
            time.sleep(hold)
        except Exception as e:
            print(f"[❌ SERVO] set_angle error: {e}")

    def fire(self):
        print("[🔫 FIRING TRIGGER!]")
        self.set_angle(FIRE_ANGLE, hold=FIRE_HOLD)
        self.set_angle(REST_ANGLE, hold=RETURN_HOLD)

    def cleanup(self):
        if not self._dry_run:
            try:
                self.lgpio.tx_pwm(self.chip, self.pin, 0, 0)
                self.lgpio.gpiochip_close(self.chip)
                print("[✅ SERVO] Cleanup done")
            except Exception as e:
                print(f"[❌ SERVO] Cleanup error: {e}")


# =============================
#        TARGET TRACKER
# =============================

class TargetTracker:
    def __init__(self, servo: ServoController):
        self.servo = servo
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        # Verify camera
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {CAMERA_INDEX}. Try 0, 1, or check with 'v4l2-ctl --list-devices'")
        print(f"[✅ CAMERA] Opened /dev/video{CAMERA_INDEX}")

        # Get frame size
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed on first frame")
        self.height, self.width = frame.shape[:2]
        print(f"[📷] Resolution: {self.width}×{self.height}")

        # Compute line
        self.line_y = int(LINE_Y_REL * self.height)
        self.seg_min_x = int(SEGMENT_MARGIN * self.width)
        self.seg_max_x = int((1 - SEGMENT_MARGIN) * self.width)
        print(f"[📏] Line at y={self.line_y}, segment x=[{self.seg_min_x}, {self.seg_max_x}]")

        # Tracker state
        self.tracks = {}          # id → {cx, cy, x, y, w, h, rel}
        self.next_id = 0
        self.cross_count = 0
        self.frame_idx = 0
        self.last_time = time.time()

        # Background subtractor (optional, but helps in static scenes)
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=36, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def relation_to_line(self, top, bottom, ly, tol):
        if bottom < ly - tol: return -1   # above
        if top > ly + tol: return +1      # below
        return 0                           # touching

    def x_overlaps_segment(self, x, w):
        left, right = x, x + w
        return not (right < self.seg_min_x or left > self.seg_max_x)

    def detect_green_objects(self, frame):
        # Optional: use background subtraction to reduce noise
        fg_mask = self.backSub.apply(frame, learningRate=0.005)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # HSV green mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
        green_mask = cv2.medianBlur(green_mask, 5)

        # Combine: motion + green
        combined = cv2.bitwise_and(green_mask, fg_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            fill_ratio = area / (w * h)
            if aspect > ASPECT_MAX or fill_ratio < FILL_MIN:
                continue
            cx, cy = x + w // 2, y + h // 2
            detections.append((x, y, w, h, cx, cy))
        return detections, combined

    def match_and_update_tracks(self, detections):
        # Simple IoU-free nearest-neighbor matching
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        # Compute distances
        dist_list = []
        for i, (_, _, _, _, cx, cy) in enumerate(detections):
            for tid, t in self.tracks.items():
                dx, dy = cx - t['cx'], cy - t['cy']
                dist = (dx**2 + dy**2)**0.5
                if dist <= MAX_DIST:
                    dist_list.append((dist, i, tid))

        # Greedy match (smallest distance first)
        dist_list.sort()
        for _, i, tid in dist_list:
            if i in unmatched_dets and tid in unmatched_tracks:
                x, y, w, h, cx, cy = detections[i]
                self.tracks[tid].update({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': cx, 'cy': cy,
                    'last_seen': self.frame_idx
                })
                unmatched_dets.remove(i)
                unmatched_tracks.remove(tid)

        # Promote unmatched detections to new tracks (after confirmation)
        for i in unmatched_dets:
            x, y, w, h, cx, cy = detections[i]
            self.tracks[self.next_id] = {
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': cx, 'cy': cy,
                'rel': self.relation_to_line(y, y+h, self.line_y, LINE_THICKNESS//2),
                'first_seen': self.frame_idx,
                'last_seen': self.frame_idx
            }
            self.next_id += 1

        # Remove stale tracks
        for tid in list(self.tracks.keys()):
            if self.frame_idx - self.tracks[tid]['last_seen'] > MAX_MISSES:
                del self.tracks[tid]

    def check_crossings_and_draw(self, frame, detections):
        now = time.time()
        fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now

        # Process tracks
        for tid, t in list(self.tracks.items()):
            x, y, w, h = t['x'], t['y'], t['w'], t['h']
            cx, cy = t['cx'], t['cy']
            prev_rel = t['rel']
            cur_rel = self.relation_to_line(y, y + h, self.line_y, LINE_THICKNESS//2)

            # Crossing: was above (-1) → now touching (0), and overlaps segment
            if prev_rel == -1 and cur_rel == 0 and self.x_overlaps_segment(x, w):
                self.cross_count += 1
                self.servo.fire()

            t['rel'] = cur_rel

            # Draw bounding box & center
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Draw line segment
        cv2.line(frame,
                 (self.seg_min_x, self.line_y),
                 (self.seg_max_x, self.line_y),
                 (0, 0, 255), LINE_THICKNESS)

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Hits: {self.cross_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        return frame

    def run(self):
        print("\n[▶️] Starting live tracker...")
        print("Controls: 'q' = quit | 'r' = reset counter | '+'/'-' = adjust line up/down")
        print("          Hold 't' to test fire servo (dry or real)")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[❌] Camera read failed")
                    break

                self.frame_idx += 1

                # Warm-up: skip first 10 frames for background model
                if self.frame_idx <= 10:
                    self.backSub.apply(frame, learningRate=0.1)
                    cv2.imshow("Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Detect
                detections, mask = self.detect_green_objects(frame)

                # Track
                self.match_and_update_tracks(detections)

                # Render
                out = frame.copy()
                out = self.check_crossings_and_draw(out, detections)

                # Show
                cv2.imshow("Tracker", out)
                cv2.imshow("Mask", mask)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.cross_count = 0
                    print("[🔁] Counter reset")
                elif key == ord('+'):
                    self.line_y = min(self.height - 20, self.line_y + 5)
                    print(f"[⬆️] Line y = {self.line_y}")
                elif key == ord('-'):
                    self.line_y = max(20, self.line_y - 5)
                    print(f"[⬇️] Line y = {self.line_y}")
                elif key == ord('t'):
                    print("[🧪] Manual fire test")
                    self.servo.fire()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


# =============================
#            MAIN
# =============================

if __name__ == "__main__":
    print("=" * 50)
    print("🎯 Hackathon 2025 — Target Tracker (Live)")
    print("=" * 50)

    servo = ServoController(debug=False)  # Set debug=True to see pulse details
    tracker = None

    try:
        tracker = TargetTracker(servo)
        tracker.run()
    except KeyboardInterrupt:
        print("\n[🛑] Interrupted by user")
    except Exception as e:
        print(f"\n[❌] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tracker:
            tracker.cap.release()
        cv2.destroyAllWindows()
        servo.cleanup()

    print("[👋] Done.")
