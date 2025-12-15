#!/usr/bin/env python3
"""
Hybrid Navigation Robot for 'eval_2nd_modi' robot platform.
1. Mode 'ARUCO': Find Markers 1-8. Odd->Left, Even->Right.
2. Trigger: If Marker 9 is detected (and close), switch to 'YOLO'.
3. Mode 'YOLO': Track selected target (AI, AWEAR, IMR).
"""
import time
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
import threading
from ultralytics import YOLO

# Import robot control
try:
    from eval_2nd_modi import DynamixelController, QuadrupedRobot, LEG_IDS, DEVICENAME, BAUDRATE
except ImportError:
    print("Error: eval_2nd_modi.py not found.")
    sys.exit(1)

# --- Configuration ---
MARKER_DICT_TYPE = cv2.aruco.DICT_6X6_100
YOLO_MODEL_PATH = "yolov8n.pt"

# ArUco Config
CENTER_THRESHOLD = 80      # Pixel threshold for centering
FOV_MARGIN = 160           # Only detect markers within +/- this pixels from center x
TARGET_DISTANCE = 0.30     # Distance to stop approaching and execute turn/switch
BURST_REPEAT = 3           # Number of times to repeat action commands
BURST_FORWARD_DIST = 0.45  # Distance threshold to trigger burst forward
MIN_DISTANCE = 0.20        # Minimum safety distance
MAX_DISTANCE = 3.0         # Ignored if further
MARKER_LENGTH = 0.06       # Marker side length in meters (Total 6cm)

# Control Config
CONTROL_COOLDOWN = 0.5     
CONTROL_FRAME_INTERVAL = 3 

# Map User Input to Class Names
TARGET_CLASS_MAP = {
    '1': 'AI',
    '2': 'AWEAR',
    '3': 'IMR'
}

class HybridTracker:
    def __init__(self, target_class):
        self.mode = 'ARUCO' # Initial Mode
        self.target_class = target_class
        
        # RealSense Setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        
        self.frame_width = 640
        self.frame_height = 480
        self.running = False
        
        # Camera Intrinsics (Calculated after start)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MARKER_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # YOLO Setup
        self.model = None
        self.model_ready = False
        # Start async loading
        threading.Thread(target=self._load_model, daemon=True).start()

        
        # Timing
        self.last_frame_time = None
        self.current_fps = 0.0
        self.last_infer_ms = 0.0

        # Navigation State
        self.nav_state = 'SEARCHING' # SEARCHING, APPROACHING, TURNING, MOVING_AFTER_TURN
        self.current_marker_id = None
        self.state_timer = 0
        self.turn_duration = 2.0  # Seconds to allow for turn
        self.move_after_turn_duration = 2.0 # Seconds to move forward after turn
        self.initial_scan_done = False # [NEW] Restrict initial search to 1 or 2


    def _load_model(self):
        print(f"Loading YOLO model: {YOLO_MODEL_PATH} (Async)...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            print("YOLO model loaded successfully.")
            self.model_ready = True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")

    def start(self):
        try:
            profile = self.pipeline.start(self.config)
            
            # Get Intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.camera_matrix = np.array([[intr.fx, 0, intr.ppx], 
                                           [0, intr.fy, intr.ppy], 
                                           [0, 0, 1]], dtype=float)
            self.dist_coeffs = np.array(intr.coeffs)
            
            self.running = True
            print("RealSense camera started.")
            print(f"Intrinsics: \n{self.camera_matrix}\nDist: {self.dist_coeffs}")
            return True
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False

    def stop(self):
        self.running = False
        self.pipeline.stop()
        print("Camera stopped.")

    def get_distance_at_point(self, depth_frame, x, y, radius=5):
        """Calculate median distance at (x,y) within radius"""
        if x < radius or y < radius or x >= self.frame_width - radius or y >= self.frame_height - radius:
            return None
        
        depth_data = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                d = depth_frame.get_distance(x + dx, y + dy)
                if d > 0:
                    depth_data.append(d)
                    
        return np.median(depth_data) if depth_data else None

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
            
        # Timing update
        now = time.time()
        if self.last_frame_time:
            dt = now - self.last_frame_time
            if dt > 0: self.current_fps = 1.0 / dt
        self.last_frame_time = now
            
        return color_frame, depth_frame

    def detect_aruco(self, color_image, depth_frame):
        t0 = time.time()
        
        # Use ArucoDetector 
        corners, ids, rejected = self.detector.detectMarkers(color_image)
        self.last_infer_ms = (time.time() - t0) * 1000.0
        
        detections = []
        if ids is not None:
            # Estimate Pose for all markers
            if self.camera_matrix is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, self.camera_matrix, self.dist_coeffs)
            else:
                rvecs, tvecs = [None]*len(ids), [None]*len(ids)

            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                c = corners[i][0]
                cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                
                # Safe access to rvec/tvec
                rv = rvecs[i] if self.camera_matrix is not None else None
                tv = tvecs[i] if self.camera_matrix is not None else None
                
                # Reshape for drawing stability
                if rv is not None: rv = rv.reshape(3, 1)
                if tv is not None: tv = tv.reshape(3, 1)

                # Use tvec for distance if available (more robust than single pixel depth)
                if tv is not None:
                    dist = float(np.linalg.norm(tv))
                else:
                    dist = self.get_distance_at_point(depth_frame, cx, cy)
                
                detections.append({
                    'type': 'aruco',
                    'id': int(marker_id),
                    'corners': c.astype(int),
                    'center': (cx, cy),
                    'distance': dist,
                    'rvec': rv,
                    'tvec': tv
                })
        return detections

    def detect_yolo(self, color_image, depth_frame):
        t0 = time.time()
        results = self.model(color_image, conf=0.5, verbose=False)
        self.last_infer_ms = (time.time() - t0) * 1000.0
        
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Filter by Target Class
                if class_name != self.target_class:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                dist = self.get_distance_at_point(depth_frame, cx, cy)
                
                detections.append({
                    'type': 'yolo',
                    'class': class_name,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (cx, cy),
                    'distance': dist
                })
        return detections

    def process_frame(self):
        """
        Main processing function.
        Returns: (color_image, target, all_detections, command, center_depth)
        """
        color_frame, depth_frame = self.get_frame()
        if color_frame is None: return None, None, [], 'none', 0.0
        
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get Center Depth for Visualization
        center_depth = self.get_distance_at_point(depth_frame, self.frame_width//2, self.frame_height//2)
        
        all_detections = []
        target = None
        command = 'none'

        if self.mode == 'ARUCO':
            # --- ARUCO MODE ---
            detections = self.detect_aruco(color_image, depth_frame)
            all_detections = detections
            
            # Helper: FOV Filter (Still useful for some checks, but searched is relaxed)
            def is_in_fov(d):
                cx = d['center'][0]
                return abs(cx - self.frame_width // 2) < FOV_MARGIN

            # State Machine for Navigation
            current_time = time.time()
            
            # 1. Check for Trigger Marker 9
            marker_9 = next((d for d in detections if d['id'] == 9 and is_in_fov(d)), None)
            
            # If we see Marker 9 and are close strictly switch
            if marker_9 and marker_9['distance'] and marker_9['distance'] < TARGET_DISTANCE:
                 # Trigger Switch!
                print(f">>> MARKER 9 DETECTED ({marker_9['distance']:.2f}m). SWITCHING TO YOLO MODE. TARGET: {self.target_class} <<<")
                
                if not self.model_ready:
                    print(">>> WARNING: YOLO Model not ready yet! Waiting... <<<")
                    # Stop briefly
                    return color_image, None, detections, 'stop', center_depth

                self.mode = 'YOLO'
                self.nav_state = 'TRACKING'
                return color_image, None, detections, 'stop', center_depth

            # 2. Navigation Logic
            if self.nav_state == 'SEARCHING':
                valid_markers = []
                for d in detections:
                    mid = d['id']
                    dist = d['distance']
                    
                    if mid == 9: continue
                    
                    # Ignore if too far (noise)
                    if dist is None or dist >= MAX_DISTANCE:
                        # print(f"DEBUG: Marker {mid} ignored (Dist: {dist})")
                        continue
                        
                    # NOTE: Removed FOV check for SEARCHING. If we see it, we lock and center.
                    
                    # Initial Search Restriction: Only look for 1 or 2 first
                    if not self.initial_scan_done:
                        if mid not in [1, 2]: 
                            # print(f"DEBUG: Marker {mid} ignored (Initial Scan Wait)")
                            continue
                        
                    valid_markers.append(d)
                
                if valid_markers:
                    # Found a marker!
                    self.initial_scan_done = True
                    target = min(valid_markers, key=lambda d: d['distance'])
                    self.current_marker_id = target['id']
                    self.nav_state = 'APPROACHING'
                    print(f">>> FOUND MARKER {self.current_marker_id}. APPROACHING. <<<")
                    command = self.get_aruco_command(target)
                else:
                    # No marker, keep rotating left
                    command = 'left' 
            
            elif self.nav_state == 'APPROACHING':
                 # Look for our cached marker id
                 target = next((d for d in detections if d['id'] == self.current_marker_id), None)
                 
                 if target:
                     command = self.get_aruco_command(target)
                     if command.startswith('action_turn'):
                         # We reached the distance. Transition to turning
                         self.nav_state = 'TURNING'
                         self.state_timer = current_time
                         print(f">>> REACHED MARKER {self.current_marker_id}. TURNING. ({command}) <<<")
                 else:
                     # Lost marker?
                     print(">>> LOST MARKER DURING APPROACH. SEARCHING. <<<")
                     self.nav_state = 'SEARCHING'
                     command = 'stop'

            elif self.nav_state == 'TURNING':
                if self.current_marker_id % 2 != 0:
                    command = 'action_turn_left'
                else:
                    command = 'action_turn_right'
                
                self.nav_state = 'MOVING_AFTER_TURN'
                self.state_timer = current_time
                
            elif self.nav_state == 'MOVING_AFTER_TURN':
                # Move forward blindly for X seconds
                if current_time - self.state_timer < self.move_after_turn_duration:
                    command = 'forward'
                else:
                    # Done moving, back to search
                    print(">>> MOVE COMPLETE. RESUMING SEARCH. <<<")
                    self.nav_state = 'SEARCHING'
                    self.current_marker_id = None
                    command = 'stop' # Brief stop before search loop
                
        elif self.mode == 'YOLO':
            # --- YOLO MODE ---
            detections = self.detect_yolo(color_image, depth_frame)
            all_detections = detections
            
            if detections:
                # Pick closest target class object
                valid_objs = [d for d in detections if d['distance'] is not None]
                if valid_objs:
                    target = min(valid_objs, key=lambda d: d['distance'])
                    command = self.get_yolo_command(target)
            
        return color_image, target, all_detections, command, center_depth

    def get_aruco_command(self, target):
        if not target: return 'none'
        
        cx, _ = target['center']
        dist = target['distance']
        marker_id = target['id']

        # Centering
        offset = cx - (self.frame_width // 2)
        if abs(offset) > CENTER_THRESHOLD:
            return 'left' if offset < 0 else 'right'
            
        # Distance/Action
        if dist > TARGET_DISTANCE:
            if dist > BURST_FORWARD_DIST:
                return 'forward_burst'
            return 'forward'
        else:
            # Action!
            if marker_id % 2 != 0: return 'action_turn_left'
            else:                  return 'action_turn_right'

    def get_yolo_command(self, target):
        if not target: return 'none'
        
        cx, _ = target['center']
        dist = target['distance']
        
        # Centering
        offset = cx - (self.frame_width // 2)
        if abs(offset) > CENTER_THRESHOLD:
            return 'left' if offset < 0 else 'right'
            
        # Approach
        if dist > TARGET_DISTANCE: 
             if dist > BURST_FORWARD_DIST:
                 return 'forward_burst'
             return 'forward'
        else:
            return 'stop' # Reached target

class RobotControllerWrapper:
    def __init__(self, robot):
        self.robot = robot
        self.last_command_time = 0
        self.current_command = None
        self.is_moving = False

    def execute_command(self, command):
        # Initial check if robot is None (Viz mode)
        if self.robot is None: return False
        
        if self.is_moving: return False
        
        current_time = time.time()
        if command != 'stop' and current_time - self.last_command_time < CONTROL_COOLDOWN:
            return False
            
        if command == self.current_command and command == 'none':
            return False

        print(f"[CTRL] Executing: {command}")
        self.is_moving = True

        if command == 'forward':
            self.robot.move_forward(fast=True)
        elif command == 'forward_burst':
            print(f">>> BURST FORWARD ({BURST_REPEAT}x) <<<")
            for _ in range(BURST_REPEAT):
                self.robot.move_forward(fast=True)
                time.sleep(0.1)
        elif command == 'left':
            self.robot.turn_left()
        elif command == 'right':
            self.robot.turn_right()
        elif command == 'action_turn_left':
            print(f">>> ODD MARKER -> LEFT BURST ({BURST_REPEAT}x) <<<")
            for _ in range(BURST_REPEAT):
                self.robot.turn_left()
                time.sleep(0.1)
        elif command == 'action_turn_right':
            print(f">>> EVEN MARKER -> RIGHT BURST ({BURST_REPEAT}x) <<<")
            for _ in range(BURST_REPEAT):
                self.robot.turn_right()
                time.sleep(0.1)
        elif command == 'stop':
             pass # Standby

        self.last_command_time = current_time
        self.current_command = command
        self.is_moving = False
        return True

def visualize(frame, target, detections, command, fps, mode, target_class, center_depth, camera_matrix=None, dist_coeffs=None):
    disp = frame.copy()
    h, w = disp.shape[:2]
    cx0, cy0 = w // 2, h // 2
    
    # 1. VISUALIZE DEPTH (Crosshair & Value)
    # Draw crosshair at center
    def draw_crosshair(img, x, y, size=15, color=(0,0,255), weight=2):
        cv2.line(img, (x - size, y), (x + size, y), color, weight)
        cv2.line(img, (x, y - size), (x, y + size), color, weight)

    draw_crosshair(disp, cx0, cy0)

    # Display Center Depth Value next to crosshair
    if center_depth:
        depth_text = f"C-Dist: {center_depth:.2f}m"
        cv2.putText(disp, depth_text, (cx0 + 20, cy0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
       
    # 2. Guides
    cv2.line(disp, (cx0, 0), (cx0, h), (255,255,255), 1)
    cv2.line(disp, (cx0-CENTER_THRESHOLD, 0), (cx0-CENTER_THRESHOLD, h), (0,255,0), 1)
    cv2.line(disp, (cx0+CENTER_THRESHOLD, 0), (cx0+CENTER_THRESHOLD, h), (0,255,0), 1)
    cv2.line(disp, (cx0-FOV_MARGIN, 0), (cx0-FOV_MARGIN, h), (255,0,0), 1)
    cv2.line(disp, (cx0+FOV_MARGIN, 0), (cx0+FOV_MARGIN, h), (255,0,0), 1)
    
    for d in detections:
        dist = d.get('distance')
        is_target = (target is not None and d == target)
        color = (0, 255, 0) if is_target else (0, 165, 255)
        
        if d['type'] == 'aruco':
            corners = d['corners'].reshape((-1, 1, 2))
            cv2.polylines(disp, [corners], True, color, 2)
            
            # Text: ID and Distance
            text = f"ID:{d['id']}"
            if dist: text += f" {dist:.2f}m"
            
            # Draw text with check for contrast/visibility
            text_pos = (int(corners[0][0][0]), int(corners[0][0][1]-10))
            cv2.putText(disp, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4) # Outline
            cv2.putText(disp, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)   # Text

            # Draw Axis if available (Robustly)
            if 'rvec' in d and 'tvec' in d and camera_matrix is not None and dist_coeffs is not None:
                try:
                    # Scale axis length based on marker size
                    axis_len = MARKER_LENGTH * 0.75
                    cv2.drawFrameAxes(disp, camera_matrix, dist_coeffs, d['rvec'], d['tvec'], axis_len)
                except Exception:
                    pass

        elif d['type'] == 'yolo':
            x1,y1,x2,y2 = d['bbox']
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            cv2.putText(disp, f"{d['class']} {dist:.2f}m" if dist else f"{d['class']}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. HUD: Locked Target Info (Prominent)
    # Shown if target is locked
    if target and target.get('type') == 'aruco':
        t_id = target['id']
        t_dist = target['distance']
        status_msg = f"LOCKED: ID {t_id} | DIST: {t_dist:.2f}m"
        
        # Center-Bottom box
        fs = 1.0
        th = 2
        sz, _ = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        bx = (w - sz[0]) // 2
        by = h - 60
        
        cv2.rectangle(disp, (bx-10, by-sz[1]-10), (bx+sz[0]+10, by+10), (0,0,0), -1)
        cv2.putText(disp, status_msg, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)

    # HUD Text Info (Bottom Left)
    cv2.putText(disp, f"MODE: {mode}", (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(disp, f"CMD: {command.upper()}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if mode == 'YOLO':
        cv2.putText(disp, f"TARGET: {target_class}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    return disp

def main():
    print("====================================")
    print(" Hybrid Navigation Robot")
    print(" 1. ARUCO Mode (1-8 Nav)")
    print(" 2. Switch on Marker 9")
    print(" 3. YOLO Mode (Track Target)")
    print("====================================")
    
    # User Input
    target_class = None
    while target_class is None:
        print("\nSelect Target Class for End Phase:")
        print(" 1: AI")
        print(" 2: AWEAR")
        print(" 3: IMR")
        choice = input("Enter choice (1/2/3): ").strip()
        target_class = TARGET_CLASS_MAP.get(choice)
        if not target_class:
            print("Invalid choice. Try again.")
            
    print(f"\n>>> Selected Target: {target_class} <<<\n")

    # Init Robot (Check connection)
    print("Connecting to Robot...")
    robot = None
    try:
        ctrl = DynamixelController(DEVICENAME, BAUDRATE)
        if not ctrl.connect():
            print("Failed to connect Dynamixel (Controller). Running in Viz-Only mode?")
        else:
            robot = QuadrupedRobot(ctrl, LEG_IDS)
            robot.enable_all_torque()
            robot.initialize_pose()
            time.sleep(1)
            robot.stand_pose()
            print("Robot Initialized.")
    except (ImportError, NameError) as e:
        print(f"Robot Driver Error ({e}). Running in Viz-Only mode.")
        robot = None
    
    # Init Tracker
    tracker = HybridTracker(target_class)
    if not tracker.start():
        sys.exit(1)
        
    robot_ctrl = RobotControllerWrapper(robot)
    
    frame_count = 0
    robot_enabled = True
    
    try:
        while True:
            frame_count += 1
            
            # Process Frame
            img, target, detections, command, center_depth = tracker.process_frame()
            
            if img is None: continue
            
            # Control
            if robot_enabled and (frame_count % CONTROL_FRAME_INTERVAL == 0):
                robot_ctrl.execute_command(command)
                
            # Visualize
            disp = visualize(img, target, detections, command, tracker.current_fps, tracker.mode, tracker.target_class, center_depth, tracker.camera_matrix, tracker.dist_coeffs)
            
            status_text = "ENABLED" if robot_enabled else "DISABLED"
            cv2.putText(disp, f"ROBOT: {status_text}", (disp.shape[1]-200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if robot_enabled else (0,0,255), 2)
            
            cv2.imshow("Hybrid Nav", disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): 
                robot_enabled = not robot_enabled
                print(f"Robot Control: {robot_enabled}")

    except KeyboardInterrupt:
        print("Stop.")
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.disable_all_torque()
            ctrl.disconnect()

if __name__ == "__main__":
    main()