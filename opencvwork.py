import cv2
import numpy as np
import time
import os  # <-- added

VIDEO_PATH = 0  # use live camera; change to 0/1/2 if needed

MIN_CONTOUR_AREA = 1500
GREEN_LOW  = np.array([40,  50,  50])
GREEN_HIGH = np.array([80, 255, 255])
KERNEL     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

backSub = cv2.createBackgroundSubtractorMOG2(
    history=120, varThreshold=40, detectShadows=False)

# Auto-headless: if no DISPLAY, don't try to open windows
HEADLESS = not os.environ.get("DISPLAY")  # <-- added

# Prefer V4L2 backend on Linux
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_V4L2)
# (Optional) nudge capture params for stability
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

print("Press 'q' to quit  |  's' to save current frame")
frame_idx = 0
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    print(frame)
    if not ret:
        print("End of video / camera stream")
        break
    frame_idx += 1

    motion_mask = backSub.apply(frame)
    _, motion_bin = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)
    motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_OPEN, KERNEL)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)

    green_motion = cv2.bitwise_and(motion_bin, motion_bin, mask=green_mask)
    green_motion = cv2.morphologyEx(green_motion, cv2.MORPH_CLOSE, KERNEL)

    contours, _ = cv2.findContours(
        green_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_green = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    out = frame.copy()

    # --- red line + intersection print ---
    LINE_X1, LINE_Y, LINE_X2, LINE_THICK = 380, 1100, 650, 10
    cv2.line(out, (LINE_X1, LINE_Y), (LINE_X2, LINE_Y), (0, 0, 200), LINE_THICK)

    touched = False
    half_t = LINE_THICK // 2

    for cnt in large_green:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"{int(area)}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        vertical_overlap   = (y <= LINE_Y + half_t) and (y + h >= LINE_Y - half_t)
        horizontal_overlap = (max(LINE_X1, x) <= min(LINE_X2, x + w))
        if vertical_overlap and horizontal_overlap:
            touched = True
        
        if touched:
            print("working")

    # FPS text (draw before any resize/show)
    now = time.time()
    fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
    prev_time = now
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- display only if not headless ---
    if not HEADLESS:
        new_width = 800
        new_height = 600
        new_resolution = (new_width, new_height)
        resized_img = cv2.resize(out, new_resolution, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Original", frame)
        cv2.imshow("Motion (binary)", motion_bin)
        cv2.imshow("Green mask", green_mask)
        cv2.imshow("Moving GREEN only", green_motion)
        cv2.imshow("Detected Targets", resized_img)
        key = cv2.waitKey(1) & 0xFF
    else:
        # Headless: no windows, simulate "no key pressed"
        key = 255
        # Optional: save a debug frame every N frames
        # if frame_idx % 60 == 0:
        #     cv2.imwrite(f"debug_{frame_idx}.jpg", out)

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"target_frame_{frame_idx}.jpg", out)
        print(f"Saved target_frame_{frame_idx}.jpg")

cap.release()
cv2.destroyAllWindows()
