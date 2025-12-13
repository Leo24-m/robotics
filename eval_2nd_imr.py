#!/usr/bin/env python3
"""
YOLO 모델과 RealSense 카메라를 사용한 객체 추적 로봇
- 객체 중심 위치에 따라 회전
- 깊이 센서로 거리 측정하여 30cm까지 접근
"""
import time
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO

# eval_1st_modi의 로봇 제어 클래스 임포트
from eval_2nd_modi import DynamixelController, QuadrupedRobot, LEG_IDS, DEVICENAME, BAUDRATE

# --- YOLO 모델 설정 ---
MODEL_PATH = "yolov8n.pt"  # 학습된 모델 경로
CONFIDENCE_THRESHOLD = 0.5  # 신뢰도 임계값
AVAILABLE_CLASSES = ["AI", "AWEAR", "IMR"]  # 사용 가능한 클래스 목록
SELECTED_TARGET_CLASS = "IMR"  # 현재 선택된 타겟 클래스 (None이면 수동 선택 모드)

# --- 로봇 제어 설정 ---
CENTER_THRESHOLD = 80      # 화면 중심 허용 범위 (픽셀)
TARGET_DISTANCE = 0.30     # 목표 거리 (미터) - 30cm
DISTANCE_TOLERANCE = 0.05  # 거리 허용 오차 (±5cm)
MIN_DISTANCE = 0.35        # 최소 안전 거리
MAX_DISTANCE = 10          # 최대 감지 거리 (m)

# 제어 쿨다운
CONTROL_COOLDOWN = 0.5     # 명령 간 최소 간격 (초)

# YOLO 프레임 N개마다 한 번만 로봇을 움직이기
CONTROL_FRAME_INTERVAL = 2  # 필요하면 5~15 사이로 조절


class YOLOObjectTracker:
    """
    YOLO 모델과 RealSense 카메라를 사용한 객체 추적 클래스
    """
    def __init__(self, model_path, target_classes=None):
        # YOLO 모델 로드
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.target_classes = target_classes if target_classes else None

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 컬러 및 깊이 스트림 설정
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Align 객체 생성 (깊이를 컬러에 정렬)
        self.align = rs.align(rs.stream.color)

        self.frame_width = 640
        self.frame_height = 480
        self.running = False

        # --- FPS / TIMING 상태 변수 ---
        self.last_frame_time = None
        self.current_fps = 0.0
        self.last_frame_idx = None
        self.last_color_ts = None
        self.last_depth_ts = None
        self.last_infer_ms = 0.0

    def start(self):
        """카메라 스트리밍 시작"""
        try:
            self.pipeline.start(self.config)
            self.running = True
            print("RealSense camera started successfully.")
            return True
        except Exception as e:
            print(f"Failed to start RealSense camera: {e}")
            return False

    def stop(self):
        """카메라 스트리밍 종료"""
        self.running = False
        self.pipeline.stop()
        print("RealSense camera stopped.")

    def detect_objects(self, frame):
        """
        YOLO로 객체를 감지합니다.
        :return: [(class_name, confidence, bbox, center), ...]
        """
        # --- YOLO 추론 시간 측정 ---
        t0 = time.time()
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        self.last_infer_ms = (time.time() - t0) * 1000.0  # ms

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # 클래스 정보
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                # 타겟 클래스 필터링
                if self.target_classes and class_name not in self.target_classes:
                    continue

                # 바운딩 박스 좌표 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                # 중심점 계산
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)

                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'center': center
                })

        return detections

    def get_distance_at_point(self, depth_frame, x, y, radius=5):
        """
        특정 점에서의 깊이(거리) 값을 가져옵니다.
        :param depth_frame: 깊이 프레임
        :param x, y: 측정할 좌표
        :param radius: 평균을 계산할 반경
        :return: 거리(미터) 또는 None
        """
        # 유효 범위 체크
        if x < radius or y < radius or x >= self.frame_width - radius or y >= self.frame_height - radius:
            return None

        # 반경 내의 깊이 값들의 중앙값 계산 (노이즈 제거)
        depth_values = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                depth = depth_frame.get_distance(x + dx, y + dy)
                if depth > 0:  # 유효한 깊이 값만
                    depth_values.append(depth)

        if len(depth_values) == 0:
            return None

        # 중앙값 반환 (아웃라이어 제거)
        return np.median(depth_values)

    def get_frame_and_detect(self):
        """
        프레임을 가져와 객체를 감지하고 거리를 측정합니다.
        :return: (color_frame, depth_frame, detections_with_distance)
        """
        frames = self.pipeline.wait_for_frames()

        # 프레임 정렬 (깊이를 컬러에 맞춤)
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # --- FPS 계산 및 프레임/타임스탬프 저장 ---
        now = time.time()
        if self.last_frame_time is None:
            self.last_frame_time = now
        else:
            dt = now - self.last_frame_time
            if dt > 0:
                self.current_fps = 1.0 / dt
            self.last_frame_time = now

        self.last_frame_idx = color_frame.get_frame_number()
        self.last_color_ts = color_frame.get_timestamp()  # ms
        self.last_depth_ts = depth_frame.get_timestamp()  # ms

        # numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 객체 감지
        detections = self.detect_objects(color_image)

        # 각 감지된 객체의 거리 측정
        for detection in detections:
            center_x, center_y = detection['center']
            distance = self.get_distance_at_point(depth_frame, center_x, center_y)
            detection['distance'] = distance

        return color_image, depth_frame, detections

    def select_target(self, detections, selected_class=None):
        """
        여러 감지된 객체 중 타겟을 선택합니다.
        :param selected_class: 선택된 클래스 이름 (None이면 가장 가까운 객체)
        :return: 선택된 detection 또는 None
        """
        if not detections:
            return None

        # 유효한 거리를 가진 객체만 필터링
        valid_detections = [d for d in detections if d.get('distance') is not None and d['distance'] > 0]

        if not valid_detections:
            return None

        # 선택된 클래스가 있으면 해당 클래스만 필터링
        if selected_class:
            class_detections = [d for d in valid_detections if d['class_name'] == selected_class]
            if not class_detections:
                return None
            valid_detections = class_detections

        # 가장 가까운 객체 선택
        target = min(valid_detections, key=lambda d: d['distance'])

        return target

    def get_control_command(self, target):
        """
        타겟 객체의 위치와 거리에 따라 제어 명령을 결정합니다.
        :return: 'forward', 'left', 'right', 'stop', 'none', 'search_left'
        """
        if target is None:
            return 'search_left'

        center_x, center_y = target['center']
        distance = target['distance']

        if distance is None or distance <= 0:
            return 'none'

        # 거리 체크
        if distance < MIN_DISTANCE:
            return 'stop'  # 너무 가까움 - 정지

        if distance > MAX_DISTANCE:
            return 'none'  # 너무 멀어서 추적 안함

        # 화면 중심과의 차이 계산
        screen_center = self.frame_width // 2
        offset = center_x - screen_center

        # 목표 거리 도달 여부 확인
        distance_error = distance - TARGET_DISTANCE

        # 방향 결정 로직
        if abs(offset) > CENTER_THRESHOLD:
            # 중앙에 없으면 회전 우선
            if offset < 0:
                return 'left'
            else:
                return 'right'
        else:
            # 중앙에 있으면 거리에 따라 전진/정지
            if abs(distance_error) < DISTANCE_TOLERANCE:
                return 'stop'  # 목표 거리 도달
            elif distance_error > 0:
                return 'forward'  # 목표보다 멀면 전진
            else:
                return 'stop'  # 목표보다 가까우면 정지


class RobotController:
    """
    로봇 제어를 담당하는 클래스
    """
    def __init__(self, robot):
        self.robot = robot
        self.last_command_time = 0
        self.current_command = None
        self.is_moving = False  # 현재 움직이는 중인지 표시
        self.extra_step = True  # 첫 stop에서 한 번 더 전진

    def execute_command(self, command):
        """
        명령에 따라 로봇을 제어합니다.
        :param command: 'forward', 'left', 'right', 'stop', 'none', 'search_left'
        :return: True if command executed, False otherwise
        """

        # 이미 움직이는 중이면 명령 무시
        if self.is_moving:
            return False

        current_time = time.time()

        # 쿨다운 체크 (stop은 즉시 실행)
        if command != 'stop' and current_time - self.last_command_time < CONTROL_COOLDOWN:
            return False

        # 같은 명령 반복 방지
        if command == self.current_command and command in ['stop', 'none']:
            return False

        print(f"Command: {command}")

        self.is_moving = True  # 동작 시작

        if command == 'forward':
            print("Moving forward...")
            self.robot.move_forward(fast=True)

        elif command == 'left':
            print("Turning left...")
            self.robot.turn_left()

        elif command == 'right':
            print("Turning right...")
            self.robot.turn_right()

        elif command == 'stop':
            if self.extra_step is True:
                print("Extra step forward before final stop...")
                self.robot.move_forward(fast=False)
                self.extra_step = False
            else:
                print("Target reached! Standing by...")

        elif command == 'search_left':
            print("Searching LEFT...")
            self.robot.turn_left()

        elif command == 'none':
            print("No target detected. Standing by...")

        self.is_moving = False  # 동작 완료
        self.last_command_time = current_time
        self.current_command = command

        return True


def visualize_tracking(frame,
                       depth_frame,
                       target,
                       all_detections,
                       command,
                       fps=None,
                       frame_idx=None,
                       infer_ms=None,
                       ts_ms=None):
    """
    객체 추적 결과를 시각화합니다.
    """
    display_frame = frame.copy()
    height, width = display_frame.shape[:2]
    screen_center = width // 2

    # 화면 중앙선 및 허용 범위 그리기
    cv2.line(display_frame, (screen_center, 0), (screen_center, height), (255, 255, 255), 2)
    cv2.line(display_frame, (screen_center - CENTER_THRESHOLD, 0),
             (screen_center - CENTER_THRESHOLD, height), (0, 255, 0), 1)
    cv2.line(display_frame, (screen_center + CENTER_THRESHOLD, 0),
             (screen_center + CENTER_THRESHOLD, height), (0, 255, 0), 1)

    # 모든 감지된 객체 표시
    for detection in all_detections:
        bbox = detection['bbox']
        center = detection['center']
        class_name = detection['class_name']
        confidence = detection['confidence']
        distance = detection.get('distance')

        # 타겟 여부에 따라 색상 변경
        is_target = (target is not None and detection == target)
        color = (0, 255, 0) if is_target else (255, 0, 0)
        thickness = 3 if is_target else 2

        # 바운딩 박스
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

        # 중심점
        cv2.circle(display_frame, center, 8, color, -1)

        # 정보 텍스트
        label = f"{class_name}: {confidence:.2f}"
        if distance is not None:
            label += f" | {distance:.2f}m"

        cv2.putText(display_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 제어 명령 표시
    command_colors = {
        'forward': (0, 255, 0),
        'left': (255, 0, 0),
        'right': (0, 0, 255),
        'stop': (0, 255, 255),
        'none': (128, 128, 128),
        'search_left': (255, 128, 0),
    }
    command_color = command_colors.get(command, (255, 255, 255))
    cv2.putText(display_frame, f"Command: {command.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, command_color, 2)

    # 목표 거리 정보 표시
    info_text = f"Target: {TARGET_DISTANCE*100:.0f}cm (+/-{DISTANCE_TOLERANCE*100:.0f}cm)"
    cv2.putText(display_frame, info_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 타겟 객체의 거리 표시
    if target is not None and target.get('distance') is not None:
        distance_text = f"Distance: {target['distance']*100:.1f}cm"
        distance_color = (0, 255, 0) if abs(target['distance'] - TARGET_DISTANCE) < DISTANCE_TOLERANCE else (0, 255, 255)
        cv2.putText(display_frame, distance_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)

    # --- FPS / TIMING 오버레이 ---
    y_base = 120
    if fps is not None:
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_base += 25

    if frame_idx is not None and ts_ms is not None:
        cv2.putText(display_frame, f"Frame: {frame_idx}  TS: {ts_ms:.1f} ms",
                    (10, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_base += 25

    if infer_ms is not None:
        cv2.putText(display_frame, f"Infer: {infer_ms:.1f} ms",
                    (10, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 깊이 맵 시각화 (컬러맵 적용)
    if depth_frame is not None:
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        cv2.imshow('Depth Map', depth_colormap)

    return display_frame


def main():
    print("=" * 60)
    print("YOLO Object Tracking Quadruped Robot")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Available Classes: {AVAILABLE_CLASSES}")
    print(f"Target Distance: {TARGET_DISTANCE * 100:.0f}cm")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 60)
    print("\nControls:")
    print("  - Press '1' to select 'AI'")
    print("  - Press '2' to select 'AWEAR'")
    print("  - Press '3' to select 'IMR'")
    print("  - Press '0' to track closest object (any class)")
    print("  - Press 's' to toggle robot control (start/stop)")
    print("  - Press 'q' to quit")
    print("=" * 60)
    print()

    # 1. Dynamixel 컨트롤러 초기화
    print("Initializing Dynamixel controller...")
    controller = DynamixelController(DEVICENAME, BAUDRATE)
    if not controller.connect():
        print("Failed to connect to Dynamixel. Exiting...")
        sys.exit(1)

    # 2. 로봇 객체 생성 및 초기화
    robot = QuadrupedRobot(controller, LEG_IDS)
    robot.enable_all_torque()
    robot.initialize_pose()
    time.sleep(1)
    robot.stand_pose()
    print("Robot initialized and ready.\n")

    # 3. YOLO 추적기 초기화
    print("Initializing YOLO tracker...")
    tracker = YOLOObjectTracker(MODEL_PATH, AVAILABLE_CLASSES)
    if not tracker.start():
        print("Failed to start camera. Exiting...")
        robot.disable_all_torque()
        controller.disconnect()
        sys.exit(1)

    # 4. 로봇 컨트롤러 생성
    robot_controller = RobotController(robot)

    # 5. 메인 루프
    robot_control_enabled = True
    selected_target_class = SELECTED_TARGET_CLASS  # 전역 변수에서 초기값 가져오기
    frame_count = 0

    # 추론 및 제어 변수
    last_target = None
    last_command = 'none'
    last_detections = []

    # 초기 타겟 클래스 안내
    if selected_target_class:
        print(f"\n>>> Current target: {selected_target_class}")
    else:
        print("\n>>> Current target: Closest object (any class)")
    print(">>> Press 1/2/3 to select specific class, or 0 for closest object\n")

    try:
        while True:
            frame_count += 1

            # 로봇이 움직이는 중인지 확인
            if robot_controller.is_moving:
                # 움직이는 중에는 이전 프레임 표시만 하고 추론 스킵
                if last_detections:
                    # 카메라에서 프레임은 가져오지만 YOLO 추론은 하지 않음
                    frames = tracker.pipeline.wait_for_frames()
                    aligned_frames = tracker.align.process(frames)
                    color_frame_raw = aligned_frames.get_color_frame()

                    if color_frame_raw:
                        color_frame = np.asanyarray(color_frame_raw.get_data())
                        # 이전 결과로 화면만 표시
                        display_frame = visualize_tracking(
                            color_frame,
                            None,
                            last_target,
                            last_detections,
                            last_command,
                            fps=tracker.current_fps,
                            frame_idx=tracker.last_frame_idx,
                            infer_ms=tracker.last_infer_ms,
                            ts_ms=tracker.last_color_ts
                        )

                        height = display_frame.shape[0]
                        status_text = "Robot: MOVING..."
                        cv2.putText(display_frame, status_text, (10, height - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        target_class_text = f"Target: {selected_target_class if selected_target_class else 'Closest'}"
                        cv2.putText(display_frame, target_class_text, (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        cv2.imshow('YOLO Object Tracking', display_frame)

                # 키 입력만 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    robot_control_enabled = not robot_control_enabled
                    status = "ENABLED" if robot_control_enabled else "DISABLED"
                    print(f"\nRobot control {status}")

                continue  # 루프 처음으로

            # 로봇이 움직이지 않을 때만 YOLO 추론 수행
            color_frame, depth_frame, detections = tracker.get_frame_and_detect()

            if color_frame is None:
                continue

            # 결과 저장
            last_detections = detections

            # 타겟 선택 (선택된 클래스 기반)
            target = tracker.select_target(detections, selected_target_class)
            last_target = target

            # 제어 명령 결정
            command = tracker.get_control_command(target)
            last_command = command

            # 로봇 제어 (활성화된 경우)
            # → YOLO/시각화는 매 프레임, 로봇은 N프레임마다 한 번만 움직이게 함
            if robot_control_enabled and (frame_count % CONTROL_FRAME_INTERVAL == 0):
                print(f"[CONTROL] Executing '{command}' on frame {frame_count}")
                robot_controller.execute_command(command)

            # 시각화
            display_frame = visualize_tracking(
                color_frame,
                depth_frame,
                target,
                detections,
                command,
                fps=tracker.current_fps,
                frame_idx=tracker.last_frame_idx,
                infer_ms=tracker.last_infer_ms,
                ts_ms=tracker.last_color_ts
            )

            # 제어 상태 및 타겟 클래스 표시
            height = display_frame.shape[0]
            status_text = "Robot: ENABLED" if robot_control_enabled else "Robot: DISABLED"
            status_color = (0, 255, 0) if robot_control_enabled else (0, 0, 255)
            cv2.putText(display_frame, status_text, (10, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 선택된 타겟 클래스 표시
            target_class_text = f"Target: {selected_target_class if selected_target_class else 'Closest'}"
            cv2.putText(display_frame, target_class_text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('YOLO Object Tracking', display_frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                robot_control_enabled = not robot_control_enabled
                status = "ENABLED" if robot_control_enabled else "DISABLED"
                print(f"\nRobot control {status}")
            elif key == ord('1'):
                selected_target_class = "AI"
                print(f"\n>>> Target changed to: {selected_target_class}")
            elif key == ord('2'):
                selected_target_class = "AWEAR"
                print(f"\n>>> Target changed to: {selected_target_class}")
            elif key == ord('3'):
                selected_target_class = "IMR"
                print(f"\n>>> Target changed to: {selected_target_class}")
            elif key == ord('0'):
                selected_target_class = None
                print("\n>>> Target changed to: Closest object (any class)")

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected. Shutting down...")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 6. 종료 처리
        print("\nCleaning up...")
        tracker.stop()
        cv2.destroyAllWindows()
        robot.disable_all_torque()
        controller.disconnect()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
