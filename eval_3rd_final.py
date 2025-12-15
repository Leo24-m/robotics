#!/usr/bin/env python3
"""
Evaluator 3rd Final: Hybrid Navigation (ArUco -> YOLO) with Robust Goal Calculation
- Uses 'eval_2nd_modi.py' for Robot Control.
- Uses 'check_aruco_2.py' logic for 3D Goal Calculation.
- Async YOLO Loading.
- Specific State Machine:
  1. Search 1/2 (Left Turn).
  2. Approach -> Buffer ID -> Reach Goal -> Turn (Configurable).
  3. Repeat until 9.
  4. At 9, approach to 1.2m -> Check YOLO -> Turn Right/Forward.
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
    from eval_2nd_modi import DynamixelController, QuadrupedRobot, LEG_IDS, DEVICENAME, BAUDRATE
except ImportError:
    print("Error: eval_2nd_modi.py not found. Robot control will fail.")
    sys.exit(1)

# --- CONFIGURATION ---
MARKER_DICT_TYPE = cv2.aruco.DICT_4X4_100  # As per previous script, or 4x4? User said "check_aruco_2 uses 4x4", but "eval_3rd" used 6x6. 
# Re-reading user request: "1번 혹은 2번의 아루코 마커를 찾을 때 까지". Usually this project uses 6x6. 
# However, check_aruco_2 used 4x4. I should probably stick to what eval_3rd used (6x6) as that's the main env, 
# BUT check_aruco_2 was the *reference* for logic. 
# Safest bet: Use 6x6 as per eval_3rd_modi.py history, but I will make it easily changeable.
# WAIT, check_aruco_2 had MARKER_DICT_TYPE = cv2.aruco.DICT_4X4_100.
# The user said "check_aruco_2 코드를 참고하여".
# I will use 6x6 because standard competition markers are usually 6x6_100 or 5x5. 4x4 is often for small tests.
# Actually, let's stick to eval_3rd_modi's 6x6 which was working.
MARKER_DICT_TYPE = cv2.aruco.DICT_4X4_100 
MARKER_LENGTH = 0.0268       # meters (마커 한 변 실제 길이)
YOLO_MODEL_PATH = "yolov8n.pt"

# Navigation Params
FOV_MARGIN = 160           # Center locking margin
CENTER_THRESHOLD = 80      # Forward alignment
APPROACH_STOP_DIST = 0.40  # Stop ArUco approach at 40cm [UPDATED]
M9_CHECK_DIST = 1.20       # Stop at 1.2m for YOLO check
TURN_WAIT_TIME = 2.0       # Wait after turn

# YOLO Target Mapping
TARGET_CLASS_MAP = {
    '1': 'AI',
    '2': 'AWEAR',
    '3': 'IMR'
}

# --- HELPER FUNCTIONS (From check_aruco_2) ---

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

def compute_goal_from_marker_using_depth_rvec(rvec, tvec_depth, d_forward=0.30):
    rvec = rvec.reshape(3, 1).astype(np.float32)
    tvec_depth = tvec_depth.reshape(3, 1).astype(np.float32)
    R, _ = cv2.Rodrigues(rvec)
    offset_marker = np.array([[0.0], [0.0], [float(d_forward)]], dtype=np.float32)
    p_goal = tvec_depth + (R @ offset_marker)
    xg = float(p_goal[0, 0])
    yg = float(p_goal[1, 0])
    zg = float(p_goal[2, 0])
    yaw = float(np.arctan2(xg, zg))
    dist_xz = float(np.sqrt(xg * xg + zg * zg))
    dist_3d = float(np.linalg.norm(p_goal))
    return xg, yg, zg, yaw, dist_xz, dist_3d

# --- HYBRID EVALUATOR CLASS ---

class HybridEvaluator:
    def __init__(self, target_class, turn_repeat=3):
        self.target_class = target_class
        self.turn_repeat = turn_repeat
        
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
        
        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MARKER_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # YOLO (Async)
        self.model = None
        self.model_ready = False
        threading.Thread(target=self._load_yolo, daemon=True).start()
        
        # State Machine
        self.state = 'SEARCHING' # SEARCHING, APPROACHING, ACTION, M9_APPROACH, M9_CHECK, FINAL_APPROACH
        self.buffered_id = None  # ID of the marker we are currently locked onto
        self.state_timer = 0.0
        self.initial_scan_done = False # [NEW] flag
        self.approach_lost_time = None # Persistence timer for APPROACHING
        
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
        
        # Store actual intrinsics object for rs2_deproject_pixel_to_point
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Matrix form for OpenCV
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
        # 1. Detect
        corners, ids, _ = self.detector.detectMarkers(img)
        detections = []
        if ids is not None:
            # 2. Estimate Pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, self.camera_matrix, self.dist_coeffs
            )
            ids = ids.flatten()
            for i, mid in enumerate(ids):
                # Basic Info
                c = corners[i][0]
                cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                
                rv = rvecs[i].reshape(3,1)
                tv = tvecs[i].reshape(3,1)
                
                # Robust Distance (Depth + Pose Fusion)
                dist_depth = get_distance_at_point(depth_frame, cx, cy)
                
                # Compute Goal/Position using check_aruco_2 logic
                if dist_depth:
                    # Deproject using REAL intrinsics object
                    p = rs.rs2_deproject_pixel_to_point(
                        self.intrinsics, [float(cx), float(cy)], float(dist_depth)
                    )
                    tvec_final = np.array([[p[0]], [p[1]], [p[2]]], dtype=np.float32)
                    dist_src = "depth"
                else:
                    tvec_final = tv
                    dist_src = "pose" # fallback

                # Distances
                dist_3d = float(np.linalg.norm(tvec_final))
                
                detections.append({
                    'type': 'aruco',
                    'id': int(mid),
                    'corners': c,
                    'center': (cx, cy),
                    'distance': dist_3d,
                    'rvec': rv,
                    'tvec': tvec_final,
                    'dist_src': dist_src
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
        
        # 1. Detection
        # Always check ArUco
        aruco_dets = self.detect_aruco(img, d_frame)
        
        # YOLO: Only detect if we found Marker 9 or are already looking for it
        yolo_dets = []
        should_run_yolo = (self.state in ['M9_APPROACH', 'M9_CHECK', 'FINAL_APPROACH'])
        # Also check if we just found M9 in this frame (handled in state logic below, but we might miss frame 1 visualization)
        # Actually, let's keep it simple: Only run if state demands it OR if we detect ID 9 in ArUco list (to be safe for immediate switch)
        
        # Optimization: Check for ID 9 in ArUco list to trigger mode switch prep (optional)
        has_m9 = any(d['id'] == 9 for d in aruco_dets)
        
        if self.model_ready and (should_run_yolo or has_m9):
             yolo_dets = self.detect_yolo(img) 
            
        all_dets = aruco_dets + yolo_dets
        command = 'stop'
        target_info = None

        curr_time = time.time()
        
        # Action Counter Init (if not exists)
        if not hasattr(self, 'action_counter'): self.action_counter = 0

        # --- STATE MACHINE ---
        
        if self.state == 'SEARCHING':
            # Check for M9 (Override Trigger)
            m9 = next((d for d in aruco_dets if d['id'] == 9), None)
            if m9:
                 self.buffered_id = 9
                 self.state = 'M9_APPROACH'
                 print(f"[STATE] Found ID 9 during SEARCH! Switching to M9_APPROACH.")
                 return img, all_dets, 'stop', self.state, m9

            # Logic:
            # - If initial_scan_done is False: Look for [1, 2]
            # - If True: Look for ANY valid marker (except 9)
            
            valid_candidates = []
            for d in aruco_dets:
                mid = d['id']
                dist = d['distance']
                
                # Skip 9 (handled above)
                if mid == 9: continue
                # Skip too far/invalid
                if dist > 3.0: continue 

                if not self.initial_scan_done:
                    if mid in [1, 2]:
                        valid_candidates.append(d)
                else:
                    # Search for any valid logic marker
                    # Optional: Avoid re-locking same ID immediately?
                    # For now, just find closest
                    valid_candidates.append(d)

            if valid_candidates:
                # Found! Lock on strongest
                target = min(valid_candidates, key=lambda x: x['distance'])
                self.buffered_id = target['id']
                self.state = 'APPROACHING'
                self.initial_scan_done = True # Mark as done once we lock on first time
                print(f"[STATE] Found ID {self.buffered_id}. APPROACHING.")
                command = 'stop' # simple pause
            else:
                command = 'left' # Search pattern

        elif self.state == 'APPROACHING':
            # We have a buffered_id. Find it.
            # ALSO: Check if we accidentally saw 9 during approach? User says "Repeat 1,2 until 9 found".
            # So if we see 9, we might need to handle it.
            
            # Find Buffered ID
            target = next((d for d in aruco_dets if d['id'] == self.buffered_id), None)
            
            # Check for M9 override? (Only if 9 is really close or we are looking for it?)
            # User says: "Until 9 is found, repeat 1,2".
            # Let's check for 9 first.
            m9 = next((d for d in aruco_dets if d['id'] == 9), None)
            if m9:
                 self.buffered_id = 9
                 self.state = 'M9_APPROACH'
                 print(f"[STATE] Found ID 9! Switching to M9_APPROACH.")
                 return img, all_dets, 'stop', self.state, m9

            if target:
                # Target found - Reset lost timer
                self.approach_lost_time = None
                
                target_info = target
                # Check Distance
                dist = target['distance']
                cx = target['center'][0]
                
                # Check if reached goal
                if dist <= APPROACH_STOP_DIST:
                    self.state = 'ACTION'
                    # action_counter not used in new logic
                    print(f"[STATE] Reached ID {self.buffered_id} ({dist:.2f}m). Starting Turn Action.")
                    command = 'stop'
                else:
                    # Centering
                    if abs(cx - 320) > CENTER_THRESHOLD:
                         command = 'left' if (cx < 320) else 'right'
                    else:
                         command = 'forward'
            else:
                # Target lost - Check persistence
                if self.approach_lost_time is None:
                    self.approach_lost_time = curr_time
                
                elapsed_lost = curr_time - self.approach_lost_time
                if elapsed_lost > 1.0:
                    # Lost for too long -> Back to SEARCH
                    print(f"[STATE] Lost Buffer ID {self.buffered_id} for {elapsed_lost:.1f}s. Back to SEARCH.")
                    self.state = 'SEARCHING'
                    self.approach_lost_time = None
                    command = 'stop'
                else:
                    # Wait/Persist (Stop moving to be safe)
                    print(f"[STATE] Lost Buffer ID {self.buffered_id} (Persisting {elapsed_lost:.1f}s)...")
                    command = 'stop'

        elif self.state == 'ACTION':
            # Execute turns based on ID (Odd=Left, Even=Right)
            is_odd = (self.buffered_id % 2 != 0)
            target_cmd = 'action_turn_left' if is_odd else 'action_turn_right'
            
            # Logic: Turn untill a NEW marker (not buffered_id) is found.
            # 1. Search for potential new markers
            new_candidates = [d for d in aruco_dets if d['id'] != self.buffered_id]
            
            # Filter out Marker 9 separately? 
            # If we see 9, we should probably handle it via the SEARCH/APPROACH override logic, 
            # but let's handle it here for smoothness.
            
            m9 = next((d for d in new_candidates if d['id'] == 9), None)
            if m9:
                 self.buffered_id = 9
                 self.state = 'M9_APPROACH'
                 print(f"[STATE] Found ID 9 during ACTION! Switching to M9_APPROACH.")
                 return img, all_dets, 'stop', self.state, m9
            
            if new_candidates:
                # Found a new marker!
                target = min(new_candidates, key=lambda x: x['distance'])
                
                # Check if it's "stable"? (Distance < 3.0m). 
                if target['distance'] < 3.0:
                    self.buffered_id = target['id']
                    self.state = 'APPROACHING'
                    self.action_counter = 0 # reset
                    print(f"[STATE] Found NEW ID {self.buffered_id}. Stopping Turn. APPROACHING.")
                    command = 'stop'
                else:
                    # Too far, keep turning
                     command = target_cmd
            else:
                # No new marker, keep turning
                command = target_cmd

        elif self.state == 'M9_APPROACH':
            target = next((d for d in aruco_dets if d['id'] == 9), None)
            if target:
                target_info = target
                dist = target['distance']
                cx = target['center'][0]
                
                # Stop at 1.2m
                if dist <= M9_CHECK_DIST:
                    self.state = 'M9_CHECK'
                    self.state_timer = curr_time
                    print(f"[STATE] Reached M9 Checkpoint ({dist:.2f}m). Checking YOLO.")
                    command = 'stop'
                else:
                     if abs(cx - 320) > CENTER_THRESHOLD:
                         command = 'left' if (cx < 320) else 'right'
                     else:
                         command = 'forward'
            else:
                # Lost 9? Search?
                print(f"[STATE] Lost M9. Searching.")
                command = 'left'

        elif self.state == 'M9_CHECK':
            # Check for specific label
            # Need to see YOLO result
            if not self.model_ready:
                print("Waiting for YOLO...")
                command = 'stop'
            else:
                # Look for target class in yolo_dets
                # User Requirement: "원하는 라벨의 로고가 안보이면 우측으로 회전"
                # "마커를 향해서 계속 나아간다" -> If seen, resume M9_APPROACH (or FINAL_APPROACH)
                
                # We need to give YOLO a moment?
                # Let's assume current frame detection is enough.
                
                found_target = any(d['class'] == self.target_class for d in yolo_dets)
                
                if found_target:
                    print(f"[STATE] YOLO Target '{self.target_class}' CONFIRMED. Moving Forward.")
                    self.state = 'FINAL_APPROACH'
                    command = 'forward'
                else:
                    print(f"[STATE] YOLO Target '{self.target_class}' NOT FOUND. Turning Right.")
                    command = 'right' # Turn right to find it? Or turn right as penalty?
                    # "우측으로 회전한다" -> implies search manually or path correction?
                    # The instruction says "If not seen, turn right". Then what?
                    # Maybe just turn right once and check again next loop? 
                    # We'll stay in M9_CHECK state but execute turn.

        elif self.state == 'FINAL_APPROACH':
             # "마커를 향해서 계속 나아간다"
             # Assuming we just track M9 or go forward blindly?
             target = next((d for d in aruco_dets if d['id'] == 9), None)
             if target:
                 target_info = target
                 cx = target['center'][0]
                 if abs(cx - 320) > CENTER_THRESHOLD:
                     command = 'left' if (cx < 320) else 'right'
                 else:
                     command = 'forward'
             else:
                 command = 'forward' # Blind forward

        return img, all_dets, command, self.state, target_info


def visualize_hud(img, state, command, fps, dets, target_info, target_class):
    disp = img.copy()
    h, w = disp.shape[:2]
    
    # 1. Crosshair
    cv2.line(disp, (320, 240-20), (320, 240+20), (0,0,255), 2)
    cv2.line(disp, (320-20, 240), (320+20, 240), (0,0,255), 2)
    
    # 2. Detections
    for d in dets:
        if d['type'] == 'aruco':
            # Corners
            cv2.polylines(disp, [d['corners'].astype(int)], True, (0,255,0), 2)
            # Center ID
            cx, cy = d['center']
            cv2.putText(disp, f"ID:{d['id']}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # Distance
            cv2.putText(disp, f"{d['distance']:.2f}m", (cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        elif d['type'] == 'yolo':
            x1,y1,x2,y2 = d['bbox']
            cv2.rectangle(disp, (x1,y1), (x2,y2), (255,0,255), 2)
            cv2.putText(disp, f"{d['class']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    # 3. HUD Stats
    # Top Box
    cv2.rectangle(disp, (0,0), (w, 60), (0,0,0), -1)
    cv2.putText(disp, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(disp, f"FPS: {fps:.1f} | CMD: {command}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    
    # Target Box
    cv2.putText(disp, f"TARGET: {target_class}", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # Target Info (if locked)
    if target_info:
        msg = f"LOCKED: ID {target_info['id']} | D: {target_info['distance']:.2f}m"
        cv2.putText(disp, msg, (w//2 - 150, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    return disp

# --- MAIN ---

def main():
    print("=== Hybrid Evaluator 3 (Final) ===")
    
    # 1. Get Target Class
    t_class = None
    while not t_class:
        k = input("Select Target (1:AI, 2:AWEAR, 3:IMR): ").strip()
        t_class = TARGET_CLASS_MAP.get(k)
    
    # 2. Get Turn Count
    try:
        t_count = int(input("Enter Turn Repeat Count (e.g. 3): ").strip())
    except ValueError:
        t_count = 3
    
    print(f">>> Target: {t_class}, Turns: {t_count} <<<")
    
    # 3. Init Robot
    print("Init Robot Controller...")
    ctrl = DynamixelController(DEVICENAME, BAUDRATE)
    if not ctrl.connect():
        sys.exit(1)
    robot = QuadrupedRobot(ctrl, LEG_IDS)
    robot.enable_all_torque()
    robot.initialize_pose()
    time.sleep(1)
    robot.stand_pose()

    # 4. Init Evaluator
    evaluator = HybridEvaluator(t_class, turn_repeat=t_count)
    if not evaluator.start():
        sys.exit(1)

    robot_wrapper = None 
    # Only need simple exec_command mapping
    
    print("Starting Loop. Press 'q' to quit, 's' to toggle Start/Stop robot.")
    
    robot_active = True
    
    try:
        while True:
            # Process
            img, dets, cmd, state, t_info = evaluator.process()
            
            if img is None: continue
            
            # Robot Control
            if robot_active:
                if cmd == 'forward':
                    robot.move_forward(fast=True)
                elif cmd == 'left':
                    robot.turn_left()
                elif cmd == 'right':
                    robot.turn_right()
                elif cmd == 'action_turn_left':
                    robot.turn_left()
                    time.sleep(0.2) # Debounce
                elif cmd == 'action_turn_right':
                    robot.turn_right()
                    time.sleep(0.2)
                elif cmd == 'stop':
                    pass
            
            # Visualize
            viz = visualize_hud(img, state, cmd, evaluator.fps, dets, t_info, t_class)
            cv2.imshow("Eval 3 Final", viz)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('s'): 
                robot_active = not robot_active
                print(f"Robot Active: {robot_active}")
                
    except KeyboardInterrupt:
        pass
    finally:
        evaluator.stop()
        robot.disable_all_torque()
        ctrl.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
