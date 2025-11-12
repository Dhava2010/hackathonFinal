#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Hackathon 2025 — Target Tracker (Low-Power Version)
- Optimized for USB cameras on Raspberry Pi
- 180p @ 10 FPS for low power draw
- Detects green targets & triggers servo (GPIO 18)
"""

import cv2
import numpy as np
import time
import sys

# =============================
#        CONFIGURATION
# =============================

CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_V4L2
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 180
CAMERA_FPS = 10

LINE_Y_REL = 0.70
LINE_THICKNESS = 10
SEGMENT_MARGIN = 0.2

GREEN_LOW  = np.array([40, 50, 50])
GREEN_HIGH = np.array([80, 255, 255])
MIN_AREA = 800
ASPECT_MAX = 4.0
FILL_MIN = 0.3

MAX_DIST = 80
MAX_MISSES = 5
MIN_FRAMES_TO_CONFIRM = 2

# ⚙️ SERVO (GPIO 18 = Pin 12)
SERVO_PIN = 18
SERVO_FREQ = 50
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
        return (pulse_us / period_us) * 100.0

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
            self.lgpio.tx_pwm(self.chip, self.pin, self.freq, duty)
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
        print("[📸] Initializing camera...")
        self.servo = servo

        self.cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        time.sleep(2)  # give camera time to warm up

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera (check /dev/video0 or permissions)")

        print(f"[✅ CAMERA] Opened /dev/video{CAMERA_INDEX}")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed on first frame — check USB power/connection")

        self.height, self.width = frame.shape[:2]
        print(f"[📷] Resolution: {self.width}x{self.height} @ {CAMERA_FPS} FPS")

        self.line_y = int(LINE_Y_REL * self.height)
        self.seg_min_x = int(SEGMENT_MARGIN * self.width)
        self.seg_max_x = int((1 - SEGMENT_MARGIN) * self.width)
        print(f"[📏] Line y={self.line_y}, segment x=[{self.seg_min_x}, {self.seg_max_x}]")

        self.tracks = {}
        self.next_id = 0
        self.cross_count = 0
        self.frame_idx = 0
        self.last_time = time.time()

        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=36, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def relation_to_line(self, top, bottom, ly, tol):
        if bottom < ly - tol: return -1
        if top > ly + tol: return +1
        return 0

    def x_overlaps_segment(self, x, w):
        left, right = x, x + w
        return not (right < self.seg_min_x or left > self.seg_max_x)

    def detect_green_objects(self, frame):
        fg_mask = self.backSub.apply(frame, learningRate=0.005)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
        green_mask = cv2.medianBlur(green_mask, 3)

        combined = cv2.bitwise_and(green_mask, fg_mask)
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
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        dist_list = []
        for i, (_, _, _, _, cx, cy) in enumerate(detections):
            for tid, t in self.tracks.items():
                dx, dy = cx - t['cx'], cy - t['cy']
                dist = (dx**2 + dy**2)**0.5
                if dist <= MAX_DIST:
                    dist_list.append((dist, i, tid))

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

        for tid in list(self.tracks.keys()):
            if self.frame_idx - self.tracks[tid]['last_seen'] > MAX_MISSES:
                del self.tracks[tid]

    def check_crossings_and_draw(self, frame, detections):
        now = time.time()
        fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now

        for tid, t in list(self.tracks.items()):
            x, y, w, h = t['x'], t['y'], t['w'], t['h']
            cx, cy = t['cx'], t['cy']
            prev_rel = t['rel']
            cur_rel = self.relation_to_line(y, y + h, self.line_y, LINE_THICKNESS//2)

            if prev_rel == -1 and cur_rel == 0 and self.x_overlaps_segment(x, w):
                self.cross_count += 1
                self.servo.fire()

            t['rel'] = cur_rel
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        cv2.line(frame, (self.seg_min_x, self.line_y),
                 (self.seg_max_x, self.line_y), (0, 0, 255), LINE_THICKNESS)
        cv2.putText(frame, f"FPS:{fps:.1f}  Hits:{self.cross_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def run(self):
        print("\n[▶️] Starting live tracker...")
        print("Controls: 'q' quit | 'r' reset | '+'/'-' move line | 't' test servo")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[❌] Camera read failed")
                    time.sleep(0.5)
                    continue

                self.frame_idx += 1
                detections, mask = self.detect_green_objects(frame)
                self.match_and_update_tracks(detections)
                out = self.check_crossings_and_draw(frame.copy(), detections)

                cv2.imshow("Tracker", out)
                cv2.imshow("Mask", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.cross_count = 0
                    print("[🔁] Counter reset")
                elif key == ord('+'):
                    self.line_y = min(self.height - 20, self.line_y + 5)
                elif key == ord('-'):
                    self.line_y = max(20, self.line_y - 5)
                elif key == ord('t'):
                    print("[🧪] Manual fire test")
                    self.servo.fire()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

# =============================
#             MAIN
# =============================

if __name__ == "__main__":
    print("=" * 50)
    print("🎯 Hackathon 2025 — Target Tracker (Low-Power 180p Version)")
    print("=" * 50)

    servo = ServoController(debug=False)
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
