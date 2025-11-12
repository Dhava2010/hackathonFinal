import cv2
import numpy as np
import time
from collections import OrderedDict

VIDEO_SOURCE = 0
LINE_THICKNESS = 10
x1, y1 = 380, 1100
x2, y2 = 650, 1100
line_y = y1
SEG_MIN_X, SEG_MAX_X = min(x1, x2), max(x1, x2)
TOL = LINE_THICKNESS // 2 + 2

GREEN_LOW = np.array([40, 50, 50])
GREEN_HIGH = np.array([80, 255, 255])
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

WARMUP_FRAMES = 45
MIN_COMPONENT_AREA = 2500
ASPECT_MAX = 6.0
FILL_MIN = 0.35

MAX_MATCH_DIST = 80
MAX_MISSES = 10
RETAIN_MIN_FRAMES = 2

SERVO_PIN = 18
SERVO_FREQ = 50
SERVO_MIN_US = 500
SERVO_MAX_US = 2500
FIRE_ANGLE = 30
REST_ANGLE = 90
FIRE_HOLD = 0.12
RETURN_HOLD = 0.08

FIRE_COOLDOWN = 3.0

def relation_to_line(top, bottom, ly, tol):
    if bottom < ly - tol:
        return -1
    if top > ly + tol:
        return +1
    return 0

def x_overlaps_segment(x, w, seg_min_x, seg_max_x):
    l, r = x, x + w
    return not (r < seg_min_x or l > seg_max_x)

def iou(a, b):
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+bh, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    u = aw*ah + bw*bh - inter
    return inter / u if u > 0 else 0.0

class ServoController:
    def __init__(self, pin=SERVO_PIN, frequency=SERVO_FREQ, min_us=SERVO_MIN_US, max_us=SERVO_MAX_US, debug=False):
        self.pin = pin
        self.freq = frequency
        self.min_us = min_us
        self.max_us = max_us
        self.debug = debug
        self.current_angle = 90
        self.last_fire_time = 0.0
        self._dry = False
        try:
            import lgpio
            self.lgpio = lgpio
            self.chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.chip, self.pin)
            self.set_angle(self.current_angle, hold=0.1)
            print("[SERVO] lgpio OK")
        except Exception as e:
            self._dry = True
            self.lgpio = None
            self.chip = None
            print("[SERVO] dry mode, lgpio unavailable:", e)

    def _angle_to_duty(self, angle):
        angle = max(0, min(180, float(angle)))
        pulse_range = self.max_us - self.min_us
        pulse = self.min_us + (angle / 180.0) * pulse_range
        period = 1_000_000 / self.freq
        duty = (pulse / period) * 100.0
        if self.debug:
            print(f"[SERVO DEBUG] angle={angle:.1f} -> {pulse:.0f}us -> duty={duty:.2f}%")
        return duty

    def set_angle(self, angle, hold=0.1):
        self.current_angle = max(0, min(180, float(angle)))
        if self._dry:
            if self.debug:
                print(f"[SERVO DRY] set_angle({self.current_angle:.1f})")
            time.sleep(hold)
            return
        try:
            duty = self._angle_to_duty(self.current_angle)
            res = self.lgpio.tx_pwm(self.chip, self.pin, self.freq, duty)
            if res < 0:
                print("tx_pwm error", res)
            time.sleep(hold)
        except Exception as e:
            print("set_angle error", e)

    def fire(self, angle=FIRE_ANGLE, rest=REST_ANGLE, hold=FIRE_HOLD, back=RETURN_HOLD):
        now = time.time()
        self.last_fire_time = now
        print("[SERVO] FIRE")
        self.set_angle(angle, hold=hold)
        self.set_angle(rest, hold=back)

    def center(self):
        self.set_angle(90, hold=0.1)

    def off(self):
        if self._dry:
            return
        try:
            self.lgpio.tx_pwm(self.chip, self.pin, 0, 0)
        except Exception as e:
            print("off error", e)

    def cleanup(self):
        if self._dry:
            return
        try:
            self.lgpio.tx_pwm(self.chip, self.pin, 0, 0)
            self.lgpio.gpiochip_close(self.chip)
        except Exception as e:
            print("cleanup error", e)

class GreenMotionCounter:
    def __init__(self, servo: ServoController):
        self.servo = servo
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=40, detectShadows=False)
        self.tracks = OrderedDict()
        self.candidates = OrderedDict()
        self.next_id = 0
        self.cand_next_id = 0
        self.cross_count = 0
        self.frame_idx = 0
        self.prev_time = time.time()
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")

    def _make_masks(self, frame):
        blur = cv2.GaussianBlur(frame, (5,5), 0)
        motion_mask = self.backSub.apply(blur, learningRate=0.005)
        _, motion_bin = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)
        motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_OPEN, KERNEL)
        motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_CLOSE, KERNEL)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
        green_mask = cv2.medianBlur(green_mask, 5)
        gm = cv2.bitwise_and(motion_bin, motion_bin, mask=green_mask)
        gm = cv2.morphologyEx(gm, cv2.MORPH_OPEN, KERNEL)
        gm = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, KERNEL)
        return gm

    def _connected_components(self, gm):
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(gm, connectivity=8)
        dets = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < MIN_COMPONENT_AREA:
                continue
            cx, cy = map(int, centroids[i])
            aspect = max(w,1)/max(h,1)
            fill = area/float(w*h)
            if aspect > ASPECT_MAX or fill < FILL_MIN:
                continue
            dets.append((x,y,w,h,cx,cy,int(area)))
        dets.sort(key=lambda d: d[6], reverse=True)
        kept = []
        for d in dets:
            if all(iou(d,k) < 0.5 for k in kept):
                kept.append(d)
        return kept

    def _match_tracks(self, detections):
        unmatched_det = set(range(len(detections)))
        unmatched_trk = set(self.tracks.keys())
        dists = []
        for i, (_,_,_,_, cx, cy, _) in enumerate(detections):
            for tid, t in self.tracks.items():
                dx, dy = cx - t['cx'], cy - t['cy']
                d2 = dx*dx + dy*dy
                dists.append((d2, i, tid))
        for d2, i, tid in sorted(dists, key=lambda z: z[0]):
            if i not in unmatched_det or tid not in unmatched_trk:
                continue
            if d2 <= MAX_MATCH_DIST*MAX_MATCH_DIST:
                x,y,w,h,cx,cy,_ = detections[i]
                self.tracks[tid].update({'cx':cx,'cy':cy,'x':x,'y':y,'w':w,'h':h,'last_seen':self.frame_idx})
                unmatched_det.remove(i)
                unmatched_trk.remove(tid)
        return unmatched_det

    def _promote_candidates(self, unmatched_det, detections):
        for i in unmatched_det:
            x,y,w,h,cx,cy,_ = detections[i]
            chosen, best = None, 1e18
            for cid, c in self.candidates.items():
                d2 = (cx - c['cx'])**2 + (cy - c['cy'])**2
                if d2 < best:
                    best, chosen = d2, cid
            if best <= (MAX_MATCH_DIST*MAX_MATCH_DIST):
                c = self.candidates[chosen]
                c.update({'cx':cx,'cy':cy,'x':x,'y':y,'w':w,'h':h,'seen_frames': c['seen_frames']+1, 'last_seen': self.frame_idx})
                if c['seen_frames'] >= RETAIN_MIN_FRAMES:
                    rel = relation_to_line(y, y+h, line_y, TOL)
                    self.tracks[self.next_id] = {'cx':cx,'cy':cy,'x':x,'y':y,'w':w,'h':h,'rel':rel,'last_seen':self.frame_idx}
                    self.next_id += 1
                    del self.candidates[chosen]
            else:
                self.candidates[self.cand_next_id] = {'cx':cx,'cy':cy,'x':x,'y':y,'w':w,'h':h,'seen_frames':1,'last_seen':self.frame_idx}
                self.cand_next_id += 1
        for cid, c in list(self.candidates.items()):
            if self.frame_idx - c['last_seen'] > MAX_MISSES:
                del self.candidates[cid]
        for tid, t in list(self.tracks.items()):
            if self.frame_idx - t['last_seen'] > MAX_MISSES:
                del self.tracks[tid]

    def _process_tracks_and_draw(self, out):
        for tid, t in list(self.tracks.items()):
            x,y,w,h = t['x'], t['y'], t['w'], t['h']
            cx,cy = t['cx'], t['cy']
            prev_rel = t['rel']
            cur_rel = relation_to_line(y, y+h, line_y, TOL)
            if prev_rel != 0 and cur_rel == 0 and x_overlaps_segment(x, w, SEG_MIN_X, SEG_MAX_X):
                now = time.time()
                if now - self.servo.last_fire_time >= FIRE_COOLDOWN:
                    self.cross_count += 1
                    self.servo.fire()
                else:
                    pass
            t['rel'] = cur_rel if cur_rel != 0 else 0
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(out, (cx,cy), 4, (0,255,0), -1)
        cv2.line(out, (x1,y1), (x2,y2), (0,0,255), LINE_THICKNESS)
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(out, f"Count: {self.cross_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

    def run(self):
        print("Press 'q' to quit  |  's' to save current frame")
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                print("End of stream")
                break
            self.frame_idx += 1
            if self.frame_idx <= WARMUP_FRAMES:
                self.backSub.apply(frame, learningRate=0.5)
                cv2.imshow("Detected Targets", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue
            gm = self._make_masks(frame)
            detections = self._connected_components(gm)
            unmatched_det = self._match_tracks(detections)
            self._promote_candidates(unmatched_det, detections)
            out = frame.copy()
            self._process_tracks_and_draw(out)
            cv2.imshow("Detected Targets", out)
            cv2.imshow("Moving GREEN only", gm)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fn = f"target_frame_{self.frame_idx}.jpg"
                cv2.imwrite(fn, out)
                print("Saved", fn)
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    servo = ServoController(pin=SERVO_PIN, frequency=SERVO_FREQ, min_us=SERVO_MIN_US, max_us=SERVO_MAX_US, debug=False)
    try:
        app = GreenMotionCounter(servo)
        app.run()
    finally:
        servo.off()
        servo.cleanup()
