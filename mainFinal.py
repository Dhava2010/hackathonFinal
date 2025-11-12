import cv2 as cv
import numpy as np
import time
import RPi.GPIO as GPIO

# =======================
# Servo setup (BCM mode)
# =======================
GPIO.setmode(GPIO.BCM)
SERVO_PIN = 18
GPIO.setup(SERVO_PIN, GPIO.OUT)

# 50 Hz PWM (typical hobby servo)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def set_angle(angle, settle=0.4):
    """Move servo to an angle (0–180)."""
    duty = 2 + (angle / 18.0)  # ~2%→0°, ~12%→180°
    pwm.ChangeDutyCycle(duty)
    time.sleep(settle)
    # reduce jitter
    pwm.ChangeDutyCycle(0)

def fire_servo():
    """Simple 'press trigger' motion: forward then back."""
    print("FIRING")
    # adjust angles & delays to match your mechanism throw
    set_angle(180, settle=0.25)
    time.sleep(0.15)
    set_angle(90, settle=0.25)

# =======================
# Vision setup
# =======================
LOWER_GREEN = np.array([35, 60, 50], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)

# Target window for the centroid
X_MIN, X_MAX = 335, 345
Y_MIN, Y_MAX = 395, 405

# Cooldown to avoid multiple rapid fires
FIRE_COOLDOWN_SEC = 1.5

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
        GPIO.cleanup()
        return

    print("Running green detector... Press Ctrl+C to stop.")
    last_print = 0.0
    last_fire  = 0.0

    try:
        # park servo at neutral to start
        set_angle(90, settle=0.3)

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

            # Pick largest green blob
            c = max(cnts, key=cv.contourArea)
            area = cv.contourArea(c)
            if area < 800:  # ignore tiny noise
                continue

            # Compute centroid
            M = cv.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            now = time.time()
            if now - last_print > 0.3:
                print(f"working — center=({cx}, {cy})")
                last_print = now

            # Check if centroid is inside the target window
            if (X_MIN <= cx <= X_MAX) and (Y_MIN <= cy <= Y_MAX):
                # respect cooldown
                if (now - last_fire) >= FIRE_COOLDOWN_SEC:
                    time.sleep(0.76)
                    fire_servo()
                    last_fire = now

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
