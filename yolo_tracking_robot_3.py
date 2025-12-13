#!/usr/bin/env python3
"""
YOLO 모델과 RealSense 카메라를 사용한 객체 추적 로봇 (기능 확장 버전)
- 초기에는 탐지 X, 이동 X
- 사용자 키 입력(1,2,3,0) 전에는 아무것도 하지 않음
- 타겟이 사라지면 마지막 위치 방향으로 검색 회전
"""
import time
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO

from eval_2nd_modi import (
    DynamixelController,
    QuadrupedRobot,
    LEG_IDS,
    DEVICENAME,
    BAUDRATE
)

# --- YOLO 모델 설정 ---
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
AVAILABLE_CLASSES = ["AI", "AWEAR", "IMR"]

# --- 로봇 제어 설정 ---
CENTER_THRESHOLD = 80
TARGET_DISTANCE = 0.30
DISTANCE_TOLERANCE = 0.05
MIN_DISTANCE = 0.35
MAX_DISTANCE = 10
CONTROL_COOLDOWN = 0.5
CONTROL_FRAME_INTERVAL = 2


class YOLOObjectTracker:
    def __init__(self, model_path, target_classes=None):
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.target_classes = target_classes

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.align = rs.align(rs.stream.color)

        self.frame_width = 640
        self.frame_height = 480
        self.running = False

        self.last_frame_time = None
        self.current_fps = 0.0
        self.last_frame_idx = None
        self.last_color_ts = None
        self.last_depth_ts = None
        self.last_infer_ms = 0.0

    def start(self):
        try:
            self.pipeline.start(self.config)
            self.running = True
            print("RealSense camera started.")
            return True
        except Exception as e:
            print(f"RealSense start failed: {e}")
            return False

    def stop(self):
        self.running = False
        self.pipeline.stop()
        print("Camera stopped.")

    def detect_objects(self, frame):
        t0 = time.time()
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        self.last_infer_ms = (time.time() - t0) * 1000.0

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                if self.target_classes and class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (center_x, center_y)
                })
        return detections

    def get_distance_at_point(self, depth_frame, x, y, radius=5):
        if x < radius or y < radius or x >= self.frame_width - radius or y >= self.frame_height - radius:
            return None
        depth_values = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                d = depth_frame.get_distance(x + dx, y + dy)
                if d > 0:
                    depth_values.append(d)
        if len(depth_values) == 0:
            return None
        return np.median(depth_values)

    def get_frame_and_detect(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame:
            return None, None, None

        now = time.time()
        if self.last_frame_time:
            dt = now - self.last_frame_time
            if dt > 0:
                self.current_fps = 1 / dt
        self.last_frame_time = now

        self.last_frame_idx = color_frame.get_frame_number()
        self.last_color_ts = color_frame.get_timestamp()
        self.last_depth_ts = depth_frame.get_timestamp()

        color_image = np.asanyarray(color_frame.get_data())

        detections = self.detect_objects(color_image)

        for d in detections:
            cx, cy = d["center"]
            d["distance"] = self.get_distance_at_point(depth_frame, cx, cy)

        return color_image, depth_frame, detections

    def select_target(self, detections, selected_class=None):
        valid = [d for d in detections if d.get("distance") is not None and d["distance"] > 0]
        if not valid:
            return None
        if selected_class:
            valid = [d for d in valid if d["class_name"] == selected_class]
            if not valid:
                return None
        return min(valid, key=lambda d: d["distance"])

    def get_control_command(self, target):
        if target is None:
            return "search_left"

        cx, cy = target["center"]
        dist = target["distance"]

        if dist is None or dist <= 0:
            return "none"
        if dist < MIN_DISTANCE:
            return "stop"
        if dist > MAX_DISTANCE:
            return "none"

        offset = cx - self.frame_width // 2
        dist_error = dist - TARGET_DISTANCE

        if abs(offset) > CENTER_THRESHOLD:
            return "left" if offset < 0 else "right"
        else:
            if abs(dist_error) < DISTANCE_TOLERANCE:
                return "stop"
            elif dist_error > 0:
                return "forward"
            else:
                return "stop"


class RobotController:
    def __init__(self, robot):
        self.robot = robot
        self.last_command_time = 0
        self.current_command = None
        self.is_moving = False
        self.extra_step = True

    def execute_command(self, command):
        if self.is_moving:
            return False

        now = time.time()
        if command != "stop" and now - self.last_command_time < CONTROL_COOLDOWN:
            return False
        if command == self.current_command and command in ["stop", "none"]:
            return False

        print(f"Command: {command}")
        self.is_moving = True

        if command == "forward":
            self.robot.move_forward(fast=True)
        elif command == "left":
            self.robot.turn_left()
        elif command == "right":
            self.robot.turn_right()
        elif command == "stop":
            if self.extra_step:
                self.robot.move_forward(fast=False)
                self.extra_step = False
            else:
                print("Target reached.")
        elif command == "none":
            pass
        elif command == "search_left":
            self.robot.turn_left()

        self.is_moving = False
        self.last_command_time = now
        self.current_command = command
        return True


# ================================
# MAIN PROGRAM
# ================================
def main():
    print("=" * 60)
    print("YOLO Object Tracking Quadruped Robot (Extended Version)")
    print("=" * 60)
    print("Press 1/2/3 to select class, 0 for closest, s to enable movement")
    print("Before selecting a class → NO DETECTION, NO MOVEMENT")
    print("=" * 60)

    controller = DynamixelController(DEVICENAME, BAUDRATE)
    if not controller.connect():
        print("Dynamixel connect failed.")
        sys.exit(1)

    robot = QuadrupedRobot(controller, LEG_IDS)
    robot.enable_all_torque()
    robot.initialize_pose()
    time.sleep(1)
    robot.stand_pose()

    tracker = YOLOObjectTracker(MODEL_PATH, AVAILABLE_CLASSES)
    if not tracker.start():
        robot.disable_all_torque()
        controller.disconnect()
        sys.exit(1)

    robot_ctrl = RobotController(robot)

    # --- 초기 상태 ---
    detection_enabled = False      # 사용자가 1/2/3/0 누를 때까지 탐지 X
    robot_control_enabled = False  # 로봇 이동도 처음에는 X
    selected_target_class = None   # 초기에는 타겟 없음
    last_target_direction = None   # 마지막 타겟 방향 (-1 left, +1 right)

    frame_count = 0
    last_detections = []
    last_target = None
    last_command = "none"

    try:
        while True:
            frame_count += 1

            # 로봇이 이미 움직이는 중이면 YOLO 생략
            if robot_ctrl.is_moving:
                frames = tracker.pipeline.wait_for_frames()
                aligned = tracker.align.process(frames)
                color_frame = aligned.get_color_frame()
                if color_frame:
                    color = np.asanyarray(color_frame.get_data())
                    cv2.imshow("YOLO Tracking", color)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            # ---------------------
            # 1) 탐지 ON/OFF 처리
            # ---------------------
            if detection_enabled:
                color_frame, depth_frame, detections = tracker.get_frame_and_detect()
            else:
                frames = tracker.pipeline.wait_for_frames()
                aligned = tracker.align.process(frames)
                color_frame_raw = aligned.get_color_frame()
                if not color_frame_raw:
                    continue
                color_frame = np.asanyarray(color_frame_raw.get_data())
                depth_frame = None
                detections = []

            last_detections = detections

            # ---------------------
            # 2) 타겟 선택
            # ---------------------
            target = tracker.select_target(detections, selected_target_class)
            last_target = target

            # 타겟 방향 저장
            if target is not None:
                cx = target["center"][0]
                if cx < tracker.frame_width // 2:
                    last_target_direction = -1
                else:
                    last_target_direction = +1

            # ---------------------
            # 3) 명령 생성
            # ---------------------
            if detection_enabled:
                command = tracker.get_control_command(target)

                # 타겟 사라지면 마지막 방향으로 검색
                if target is None:
                    if last_target_direction == -1:
                        command = "left"
                    elif last_target_direction == 1:
                        command = "right"
                    else:
                        command = "left"
            else:
                command = "none"

            last_command = command

            # ---------------------
            # 4) 로봇 이동
            # ---------------------
            if robot_control_enabled and (frame_count % CONTROL_FRAME_INTERVAL == 0):
                robot_ctrl.execute_command(command)

            # ---------------------
            # 5) 화면 표시
            # ---------------------
            cv2.imshow("YOLO Tracking", color_frame)

            # ---------------------
            # 6) 키 입력 처리
            # ---------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # 타겟 선택 → 탐지 ON
            if key == ord("1"):
                selected_target_class = "AI"
                detection_enabled = True
                print(">> Target = AI")

            elif key == ord("2"):
                selected_target_class = "AWEAR"
                detection_enabled = True
                print(">> Target = AWEAR")

            elif key == ord("3"):
                selected_target_class = "IMR"
                detection_enabled = True
                print(">> Target = IMR")

            elif key == ord("0"):
                selected_target_class = None
                detection_enabled = True
                print(">> Target = Closest object")

            # 이동 토글
            elif key == ord("s"):
                robot_control_enabled = not robot_control_enabled
                print(f">> Robot control = {robot_control_enabled}")

    finally:
        print("\nCleaning up...")
        tracker.stop()
        cv2.destroyAllWindows()
        robot.disable_all_torque()
        controller.disconnect()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
