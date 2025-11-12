import cv2 as cv
import numpy as np
import time

# --- HSV range for green (tweak if your lighting differs) ---
LOWER_GREEN = np.array([35, 60, 50], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)

def open_camera():
    """Try to open a USB or Pi camera automatically."""
    for dev in ("/dev/video0", "/dev/video1", "/dev/video2"):
        cap = cv.VideoCapture(dev, cv.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    # fallback for Pi Camera Module
    pipeline = (
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 "
        "! videoconvert ! appsink"
    )
    cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    return cap if cap.isOpened() else None

def main():
    cap = open_camera()
    if not cap:
        print("Error: Could not open camera.")
        return

    print("Running green detector... Press Ctrl+C to stop.")
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: Failed to grab frame.")
                continue

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

            # Morphological cleanup
            k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=1)

            # Find contours (blobs of green)
            cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            # Pick the largest green blob (approx the target)
            c = max(cnts, key=cv.contourArea)
            area = cv.contourArea(c)
            if area < 800:  # ignore tiny noise
                continue

            # Compute centroid of that contour
            M = cv.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            now = time.time()
            if now - last_print > 0.3:  # avoid spamming output
                print(f"working — center=({cx}, {cy})")
                last_print = now

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
