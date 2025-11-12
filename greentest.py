import cv2 as cv
import numpy as np
import time

# Try common V4L2 device nodes.
for dev in ("/dev/video0", "/dev/video1"):
    cap = cv.VideoCapture(dev, cv.CAP_V4L2)
    if cap.isOpened():
        break

if not cap or not cap.isOpened():
    print("Error: Could not open camera (USB). Check /dev/video* and permissions.")
    raise SystemExit

# Optional: Make many USB cams happier.
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

print("Running... Press Ctrl+C to stop.")
lower_green = np.array([35, 60, 50], dtype=np.uint8)
upper_green = np.array([85, 255, 255], dtype=np.uint8)

last_print = 0.0
try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Warning: Failed to grab frame."); continue

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_green, upper_green)
        if cv.countNonZero(mask) > 1000:
            now = time.time()
            if now - last_print > 0.5:  # Don’t spam
                print("working")
                last_print = now
except KeyboardInterrupt:
    pass
finally:
    cap.release()
