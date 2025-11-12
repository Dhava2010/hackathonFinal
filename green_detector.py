#!/usr/bin/env python3
import cv2
import numpy as np
import time
from collections import OrderedDict
from servo_controller import ServoController

VIDEO_SOURCE = 0

LINE_THICKNESS = 10
x1, y1 = 380, 1100
x2, y2 = 650, 1100
line_y = y1
SEG_MIN_X, SEG_MAX_X = min(x1, x2), max(x1, x2)
TOL = LINE_THICKNESS // 2 + 2

GREEN_LOW  = np.array([40, 50, 50])
GREEN_HIGH = np.array([80, 255, 255])
KERNEL     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

WARMUP_FRAMES = 45
MIN_COMPONENT_AREA = 2500
ASPECT_MAX = 6.0
FILL_MIN   = 0.35

MAX_MATCH_DIST = 80
MAX_MISSES = 10
RETAIN_MIN_FRAMES = 2

def relation_to_line(top, bottom, ly, tol):
    if bottom < ly - tol:
        return -1
    if top    > ly + tol:
        return +1
    return 0

def x_overlaps_segment(x, w, seg_min_x, seg_max_x):
    l, r = x, x + w
    return not (r < seg_min_x or l > seg_max_x)

def iou(a, b):
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    x1i, y1i = max(ax, bx), max(ay, by)
    x2i, y2i = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2i-x1i) * max(0, y2i-y1i)
    u = aw*ah + bw*bh - inter
    return inter / u if u > 0 else 0.0

class GreenMotionCounter:
    def __init__(self, servo: ServoController, video_source=VIDEO_SOURCE):
        self.servo = servo
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=40, detectShadows=False)
        self.tracks = OrderedDict()
        self.candidates = OrderedDict()
        self.next_id = 0
        self.cand_next_id = 0
        self.cross_count = 0
        self.frame_idx = 0
        self.prev_time = time.time()
        self.cap = cv2.VideoCapture(video_source)
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
            cx,cy   = t['cx'], t['cy']
            prev_rel = t.get('rel', 0)
            cur_rel  = relation_to_line(y, y+h, line_y, TOL)
            if prev_rel != 0 and cur_rel == 0 and x_overlaps_segment(x, w, SEG_MIN_X, SEG_MAX_X):
                fired = self.servo.fire()
                if fired:
                    self.cross_count += 1
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
                print(f"Saved {fn}")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    servo = ServoController(pin=18, frequency=50, min_us=500, max_us=2500, cooldown=3.0, debug=False)
    try:
        app = GreenMotionCounter(servo, video_source=VIDEO_SOURCE)
        app.run()
    finally:
        servo.off()
        servo.cleanup()
