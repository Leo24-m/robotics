#!/usr/bin/env python3
"""
Evaluator 4th: Sidestep Navigation
- Based on 'eval_3rd_final.py' logic.
- Uses 'eval_2nd_modi_v2.py' for Robot Control (Sidestep Gaits).
- Logic:
  1. Search 1 or 2 (Left Turn).
  2. Approach -> Reach 40cm.
  3. Action: Sidestep (Left if Odd, Right if Even) until NEW marker found.
  4. Repeat until Marker 9.
  5. At Marker 9: Center it -> Activate YOLO -> Track Target.
"""

import time
import sys
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# --- IMPORT ROBOT MODULES ---
try:
    # Changed to v2 as requested
    from eval_2nd_modi_v2 import DynamixelController, QuadrupedRobot, LEG_IDS, DEVICENAME, BAUDRATE, GaitSpeed
except ImportError:
    print("Error: eval_2nd_modi_v2.py not found. Robot control will fail.")
    sys.exit(1)

# --- CONFIGURATION ---
MARKER_DICT_TYPE = cv2.aruco.DICT_4X4_100 
MARKER_LENGTH = 0.0268       # meters
YOLO_MODEL_PATH = "yolov8n.pt"

# Navigation Params
FOV_MARGIN = 160           # Center locking margin
CENTER_THRESHOLD = 80      # Forward alignment
APPROACH_STOP_DIST = 0.40  # Stop ArUco approach at 40cm
M9_CHECK_DIST = 1.00       # Distance to stop and center for M9 (approx 1m)
TURN_WAIT_TIME = 2.0       

# YOLO Target Mapping
TARGET_CLASS_MAP = {
    '1': 'AI',
    '2': 'AWEAR',
    '3': 'IMR'
}

# --- HELPER FUNCTIONS ---

def get_distance_at_point(depth_frame, x, y, radius=3, max_dist=10.0):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    if x < radius or y < radius or x >= w - radius or y >= h - radius:
        return None
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = depth_frame.get_distance(int(x + dx), int(y + dy))
            if d > 0 and d < max_dist:
                vals.append(d)
    return float(np.median(vals)) if vals else None

# --- HYBRID EVALUATOR CLASS ---

class HybridEvaluator:
    def __init__(self, target_class):
        self.target_class = target_class
        
        # RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.frame_width = 640
        self.frame_height = 480
        
        # Intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics = None
        
        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MARKER_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # YOLO (Async)
        self.model = None
        self.model_ready = False
        threading.Thread(target=self._load_yolo, daemon=True).start()
        
        # State Machine
        self.state = 'SEARCHING' # SEARCHING, APPROACHING, ACTION_SIDESTEP, M9_CENTERING, YOLO_TRACKING
        self.buffered_id = None
        self.initial_scan_done = False
        self.approach_lost_time = None
        
        # Performance
        self.last_frame_time = time.time()
        self.fps = 0.0

    def _load_yolo(self):
        print(">>> YOLO Loading (Async)...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model_ready = True
            print(">>> YOLO Loaded.")
        except Exception as e:
            print(f"!!! YOLO Load Failed: {e}")

    def start(self):
        profile = self.pipeline.start(self.config)
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx], 
                                       [0, self.intrinsics.fy, self.intrinsics.ppy], 
                                       [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.array(self.intrinsics.coeffs)
        return True

    def stop(self):
        self.pipeline.stop()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        now = time.time()
        dt = now - self.last_frame_time
        if dt > 0: self.fps = 1.0/dt
        self.last_frame_time = now
        return color_frame, depth_frame

    def detect_aruco(self, img, depth_frame):
        corners, ids, _ = self.detector.detectMarkers(img)
        detections = []
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, self.camera_matrix, self.dist_coeffs
            )
            ids = ids.flatten()
            for i, mid in enumerate(ids):
                c = corners[i][0]
                cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                
                rv = rvecs[i].reshape(3,1)
                tv = tvecs[i].reshape(3,1)
                
                dist_depth = get_distance_at_point(depth_frame, cx, cy)
                
                if dist_depth:
                    p = rs.rs2_deproject_pixel_to_point(
                        self.intrinsics, [float(cx), float(cy)], float(dist_depth)
                    )
                    tvec_final = np.array([[p[0]], [p[1]], [p[2]]], dtype=np.float32)
                else:
                    tvec_final = tv

                dist_3d = float(np.linalg.norm(tvec_final))
                
                detections.append({
                    'type': 'aruco',
                    'id': int(mid),
                    'corners': c,
                    'center': (cx, cy),
                    'distance': dist_3d,
                    'tvec': tvec_final
                })
        return detections

    def detect_yolo(self, img):
        if not self.model_ready: return []
        results = self.model(img, conf=0.5, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                detections.append({
                    'type': 'yolo',
                    'class': cls_name,
                    'center': (cx, cy),
                    'bbox': (x1,y1,x2,y2)
                })
        return detections

    def process(self):
        c_frame, d_frame = self.get_frame()
        if not c_frame: return None, None, 'none', self.state, None
        
        img = np.asanyarray(c_frame.get_data())
        aruco_dets = self.detect_aruco(img, d_frame)
        yolo_dets = []
        
        # Run YOLO only if in YOLO tracking state
        if self.state == 'YOLO_TRACKING' and self.model_ready:
             yolo_dets = self.detect_yolo(img) 
            
        all_dets = aruco_dets + yolo_dets
        command = 'stop'
        target_info = None
        curr_time = time.time()

        # --- STATE MACHINE ---
        
        if self.state == 'SEARCHING':
            # 1. Check for M9 override
            m9 = next((d for d in aruco_dets if d['id'] == 9), None)
            if m9:
                 self.buffered_id = 9
                 self.state = 'M9_CENTERING'
                 print(f"[STATE] Found ID 9 during SEARCH! Switching to M9_CENTERING.")
                 return img, all_dets, 'stop', self.state, m9

            # 2. Search Logic
            valid_candidates = []
            for d in aruco_dets:
                mid = d['id']
                if mid == 9: continue
                if d['distance'] > 3.0: continue 

                if not self.initial_scan_done:
                    if mid in [1, 2]: valid_candidates.append(d)
                else:
                    valid_candidates.append(d)

            if valid_candidates:
                target = min(valid_candidates, key=lambda x: x['distance'])
                self.buffered_id = target['id']
                self.state = 'APPROACHING'
                self.initial_scan_done = True
                print(f"[STATE] Found ID {self.buffered_id}. APPROACHING.")
                command = 'stop'
            else:
                command = 'left' # Rotate left to find

        elif self.state == 'APPROACHING':
            # Find Buffered ID
            target = next((d for d in aruco_dets if d['id'] == self.buffered_id), None)
            
            # M9 Check
            m9 = next((d for d in aruco_dets if d['id'] == 9), None)
            if m9:
                 self.buffered_id = 9
                 self.state = 'M9_CENTERING'
                 print(f"[STATE] Found ID 9! Switching to M9_CENTERING.")
                 return img, all_dets, 'stop', self.state, m9
            
            if target:
                self.approach_lost_time = None
                target_info = target
                dist = target['distance']
                cx = target['center'][0]
                
                if dist <= APPROACH_STOP_DIST:
                    self.state = 'ACTION_SIDESTEP'
                    print(f"[STATE] Reached ID {self.buffered_id} ({dist:.2f}m). Starting Sidestep.")
                    command = 'stop'
                else:
                    if abs(cx - 320) > CENTER_THRESHOLD:
                         command = 'left' if (cx < 320) else 'right'
                    else:
                         command = 'forward'
            else:
                # Persistence
                if self.approach_lost_time is None: self.approach_lost_time = curr_time
                if curr_time - self.approach_lost_time > 1.0:
                    print(f"[STATE] Lost ID {self.buffered_id}. Back to SEARCH.")
                    self.state = 'SEARCHING'
                    self.approach_lost_time = None
                    command = 'stop'
                else:
                    command = 'stop'

        elif self.state == 'ACTION_SIDESTEP':
            # Determine direction based on buffered_id
            is_odd = (self.buffered_id % 2 != 0)
            sidestep_cmd = 'sidestep_left' if is_odd else 'sidestep_right'
            
            # Check for NEW marker
            new_candidates = [d for d in aruco_dets if d['id'] != self.buffered_id]
            
            # Check M9 first
            m9 = next((d for d in new_candidates if d['id'] == 9), None)
            if m9:
                 self.buffered_id = 9
                 self.state = 'M9_CENTERING'
                 print(f"[STATE] Found ID 9 during Sidestep! Switching to M9_CENTERING.")
                 return img, all_dets, 'stop', self.state, m9
            
            if new_candidates:
                target = min(new_candidates, key=lambda x: x['distance'])
                if target['distance'] < 3.0:
                    self.buffered_id = target['id']
                    self.state = 'APPROACHING'
                    print(f"[STATE] Found NEW ID {self.buffered_id}. APPROACHING.")
                    command = 'stop'
                else:
                    # Alignment Check during Sidestep (User Allow: "Rotation matches vertical")
                    old_target = next((d for d in aruco_dets if d['id'] == self.buffered_id), None)
                    if old_target:
                         cx = old_target['center'][0]
                         if abs(cx - 320) > CENTER_THRESHOLD:
                             command = 'left' if (cx < 320) else 'right'
                             print(f"[STATE] Aligning Old ID {self.buffered_id} during Sidestep.")
                         else:
                             command = sidestep_cmd
                    else:
                        command = sidestep_cmd
            else:
                 # Alignment Check during Sidestep
                old_target = next((d for d in aruco_dets if d['id'] == self.buffered_id), None)
                if old_target:
                     cx = old_target['center'][0]
                     if abs(cx - 320) > CENTER_THRESHOLD:
                         command = 'left' if (cx < 320) else 'right'
                         print(f"[STATE] Aligning Old ID {self.buffered_id} during Sidestep.")
                     else:
                         command = sidestep_cmd
                else:
                    command = sidestep_cmd

        elif self.state == 'M9_CENTERING':
            # Approach/Center M9 until it is centered and close enough?
            # User requirement: "9번 마커가 가운데 인식된 뒤에" (After M9 is detected in center)
            target = next((d for d in aruco_dets if d['id'] == 9), None)
            if target:
                target_info = target
                cx = target['center'][0]
                dist = target['distance']
                
                # Check Center
                if abs(cx - 320) < CENTER_THRESHOLD:
                    # Centered!
                    print(f"[STATE] M9 Centered. Switching to YOLO_TRACKING.")
                    self.state = 'YOLO_TRACKING'
                    command = 'stop'
                else:
                    # Align
                    command = 'left' if (cx < 320) else 'right'
            else:
                print("[STATE] M9 Lost. Searching...")
                command = 'left'

        elif self.state == 'YOLO_TRACKING':
            if not self.model_ready:
                print("Waiting for YOLO...")
                command = 'stop'
            else:
                # Find target class
                found = next((d for d in yolo_dets if d['class'] == self.target_class), None)
                if found:
                    cx = found['center'][0]
                    print(f"[STATE] Tracking {self.target_class} at {cx}")
                    if abs(cx - 320) > CENTER_THRESHOLD:
                        command = 'left' if (cx < 320) else 'right'
                    else:
                        command = 'forward'
                else:
                     print(f"[STATE] Target {self.target_class} Not Found. Scanning...")
                     command = 'right' # Default scan?

        return img, all_dets, command, self.state, target_info

def visualize_hud(img, state, command, fps, dets, target_info, target_class):
    disp = img.copy()
    h, w = disp.shape[:2]
    cv2.line(disp, (320, 220), (320, 260), (0,0,255), 2)
    cv2.line(disp, (300, 240), (340, 240), (0,0,255), 2)
    
    for d in dets:
        if d['type'] == 'aruco':
            cv2.polylines(disp, [d['corners'].astype(int)], True, (0,255,0), 2)
            cx, cy = d['center']
            cv2.putText(disp, f"ID:{d['id']}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(disp, f"{d['distance']:.2f}m", (cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        elif d['type'] == 'yolo':
            x1,y1,x2,y2 = d['bbox']
            cv2.rectangle(disp, (x1,y1), (x2,y2), (255,0,255), 2)
            cv2.putText(disp, f"{d['class']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    cv2.rectangle(disp, (0,0), (w, 60), (0,0,0), -1)
    cv2.putText(disp, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(disp, f"CMD: {command} | FPS: {fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(disp, f"TARGET: {target_class}", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return disp

def main():
    print("=== Evaluator 4: Sidestep Navigation ===")
    t_class = None
    while not t_class:
        k = input("Select Target (1:AI, 2:AWEAR, 3:IMR): ").strip()
        t_class = TARGET_CLASS_MAP.get(k)
    print(f">>> Target: {t_class} <<<")
    
    print("Init Robot Controller (v2)...")
    ctrl = DynamixelController(DEVICENAME, BAUDRATE)
    if not ctrl.connect(): sys.exit(1)
    robot = QuadrupedRobot(ctrl, LEG_IDS)
    robot.enable_all_torque()
    robot.initialize_pose()
    time.sleep(1)
    robot.stand_pose()

    evaluator = HybridEvaluator(t_class)
    if not evaluator.start(): sys.exit(1)
    
    print("Starting Loop. 'q'=quit, 's'=toggle robot.")
    robot_active = True
    
    try:
        while True:
            img, dets, cmd, state, t_info = evaluator.process()
            if img is None: continue
            
            if robot_active:
                if cmd == 'forward': robot.move_forward(speed=GaitSpeed.FAST)
                elif cmd == 'left': robot.turn_left()
                elif cmd == 'right': robot.turn_right()
                elif cmd == 'sidestep_left': robot.sidestep_left()
                elif cmd == 'sidestep_right': robot.sidestep_right()
                elif cmd == 'stop': pass
            
            viz = visualize_hud(img, state, cmd, evaluator.fps, dets, t_info, t_class)
            cv2.imshow("Eval 4 Sidestep", viz)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('s'): 
                robot_active = not robot_active
                print(f"Robot Active: {robot_active}")
                
    except KeyboardInterrupt: pass
    finally:
        evaluator.stop()
        robot.disable_all_torque()
        ctrl.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
