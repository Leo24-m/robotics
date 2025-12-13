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
from ultralytics import YOLO

# Import robot control
try:
    from eval_2nd_modi import DynamixelController, QuadrupedRobot, LEG_IDS, DEVICENAME, BAUDRATE
except ImportError:
    print("Error: eval_2nd_modi.py not found.")
    sys.exit(1)

# --- Configuration ---
MARKER_DICT_TYPE = cv2.aruco.DICT_5X5_100
YOLO_MODEL_PATH = "yolov8n.pt"

# ArUco Config
CENTER_THRESHOLD = 80      # Pixel threshold for centering
TARGET_DISTANCE = 0.40     # Distance to stop approaching and execute turn/switch
MIN_DISTANCE = 0.20        # Minimum safety distance
MAX_DISTANCE = 3.0         # Ignored if further

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
        
        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MARKER_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # YOLO Setup
        print(f"Loading YOLO model: {YOLO_MODEL_PATH}...")
        self.model = YOLO(YOLO_MODEL_PATH)
        
        # Timing
        self.last_frame_time = None
        self.current_fps = 0.0
        self.last_infer_ms = 0.0

    def start(self):
        try:
            self.pipeline.start(self.config)
            self.running = True
            print("RealSense camera started.")
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
        corners, ids, _ = cv2.aruco.detectMarkers(
            color_image, self.aruco_dict, parameters=self.aruco_params
        )
        self.last_infer_ms = (time.time() - t0) * 1000.0
        
        detections = []
        if ids is not None:
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                c = corners[i][0]
                cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                dist = self.get_distance_at_point(depth_frame, cx, cy)
                
                detections.append({
                    'type': 'aruco',
                    'id': int(marker_id),
                    'corners': c.astype(int),
                    'center': (cx, cy),
                    'distance': dist
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
        Returns: (color_image, target, all_detections, command)
        """
        color_frame, depth_frame = self.get_frame()
        if color_frame is None: return None, None, [], 'none'
        
        color_image = np.asanyarray(color_frame.get_data())
        
        all_detections = []
        target = None
        command = 'none'

        if self.mode == 'ARUCO':
            # --- ARUCO MODE ---
            detections = self.detect_aruco(color_image, depth_frame)
            all_detections = detections
            
            # 1. Check for Trigger Marker 9
            marker_9 = next((d for d in detections if d['id'] == 9), None)
            
            if marker_9 and marker_9['distance'] and marker_9['distance'] < TARGET_DISTANCE:
                # Trigger Switch!
                print(f">>> MARKER 9 DETECTED ({marker_9['distance']:.2f}m). SWITCHING TO YOLO MODE. TARGET: {self.target_class} <<<")
                self.mode = 'YOLO'
                return color_image, None, detections, 'stop' # Stop briefly
            
            # 2. Select Marker Target (Closest 1-8)
            # Filter out 9 if it's far, since we only care if close
            valid_markers = [d for d in detections if d['id'] != 9 and d['distance'] is not None and d['distance'] < MAX_DISTANCE]
            if valid_markers:
                target = min(valid_markers, key=lambda d: d['distance'])
                command = self.get_aruco_command(target)
                
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
            
        return color_image, target, all_detections, command

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
        if dist > TARGET_DISTANCE: # Stop at same distance? Or get closer? YOLO usually follows.
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
        elif command == 'left':
            self.robot.turn_left()
        elif command == 'right':
            self.robot.turn_right()
        elif command == 'action_turn_left':
            print(">>> ODD MARKER -> LEFT <<<")
            self.robot.turn_left()
        elif command == 'action_turn_right':
            print(">>> EVEN MARKER -> RIGHT <<<")
            self.robot.turn_right()
        elif command == 'stop':
             pass # Standby

        self.last_command_time = current_time
        self.current_command = command
        self.is_moving = False
        return True

def visualize(frame, target, detections, command, fps, mode, target_class):
    disp = frame.copy()
    h, w = disp.shape[:2]
    cx = w // 2
    
    # Guides
    cv2.line(disp, (cx, 0), (cx, h), (255,255,255), 1)
    cv2.line(disp, (cx-CENTER_THRESHOLD, 0), (cx-CENTER_THRESHOLD, h), (0,255,0), 1)
    cv2.line(disp, (cx+CENTER_THRESHOLD, 0), (cx+CENTER_THRESHOLD, h), (0,255,0), 1)
    
    for d in detections:
        dist = d.get('distance')
        is_target = (target is not None and d == target)
        color = (0, 255, 0) if is_target else (0, 165, 255)
        
        if d['type'] == 'aruco':
            corners = d['corners'].reshape((-1, 1, 2))
            cv2.polylines(disp, [corners], True, color, 2)
            cv2.putText(disp, f"ID:{d['id']} {dist:.2f}m" if dist else f"ID:{d['id']}", 
                        (int(corners[0][0][0]), int(corners[0][0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif d['type'] == 'yolo':
            x1,y1,x2,y2 = d['bbox']
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            cv2.putText(disp, f"{d['class']} {dist:.2f}m" if dist else f"{d['class']}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # HUD
    # Mode Indicator
    mode_color = (0, 255, 255) if mode == 'ARUCO' else (255, 0, 255)
    cv2.putText(disp, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    cv2.putText(disp, f"CMD: {command.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if mode == 'YOLO':
        cv2.putText(disp, f"TARGET: {target_class}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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

    # Init Robot
    print("Connecting to Robot...")
    ctrl = DynamixelController(DEVICENAME, BAUDRATE)
    if not ctrl.connect():
        sys.exit(1)
    robot = QuadrupedRobot(ctrl, LEG_IDS)
    robot.enable_all_torque()
    robot.initialize_pose()
    time.sleep(1)
    robot.stand_pose()
    
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
            # Note: tracker.process_frame() encapsulates detect and logic
            img, target, detections, command = tracker.process_frame()
            
            if img is None: continue
            
            # Control
            if robot_enabled and (frame_count % CONTROL_FRAME_INTERVAL == 0):
                robot_ctrl.execute_command(command)
                
            # Visualize
            disp = visualize(img, target, detections, command, tracker.current_fps, tracker.mode, tracker.target_class)
            
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
        robot.disable_all_torque()
        ctrl.disconnect()

if __name__ == "__main__":
    main()
