#!/usr/bin/env python3
"""
Improved Hybrid Navigation Robot for 'eval_2nd_modi' robot platform.

Mission Flow:
1. ARUCO Mode: Navigate through maze using markers 1-8
   - Odd markers (1,3,5,7) -> Turn LEFT after reaching
   - Even markers (2,4,6,8) -> Turn RIGHT after reaching
2. Marker 9: Final ArUco marker - NO TURN, just switch to YOLO
3. YOLO Mode: Detect and approach target logo (AI, AWEAR, or IMR)
   - Navigate to logo and stop when reached

Improvements:
- Fixed critical bugs in detect_aruco()
- Added proper error handling
- Removed code duplication
- Better state management
- Configuration management with ratios
- Thread safety for model loading
- Logging system
"""
import time
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
import threading
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Tuple
from ultralytics import YOLO

# Import robot control
try:
    from eval_2nd_modi import DynamixelController, QuadrupedRobot, LEG_IDS, DEVICENAME, BAUDRATE
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: eval_2nd_modi.py not found. Running in visualization-only mode.")
    ROBOT_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Configuration ---
@dataclass
class NavigationConfig:
    """Configuration for navigation parameters"""
    # Camera resolution
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    
    # ArUco Configuration
    MARKER_DICT_TYPE: int = cv2.aruco.DICT_5X5_100
    MARKER_LENGTH: float = 0.05  # Marker side length in meters
    
    # Navigation thresholds (ratio-based for resolution independence)
    CENTER_THRESHOLD_RATIO: float = 0.125  # 80/640
    FOV_MARGIN_RATIO: float = 0.25  # 160/640
    
    # Distance thresholds
    TARGET_DISTANCE: float = 0.30  # Distance to execute turn (for markers 1-8)
    MARKER_9_DISTANCE: float = 0.30  # Distance to switch to YOLO at marker 9
    BURST_FORWARD_DIST: float = 0.45
    MIN_DISTANCE: float = 0.20
    MAX_DISTANCE: float = 3.0
    
    # Control parameters
    CONTROL_COOLDOWN: float = 0.5
    CONTROL_FRAME_INTERVAL: int = 3
    BURST_REPEAT: int = 3
    
    # State timing
    TURN_DURATION: float = 2.0
    MOVE_AFTER_TURN_DURATION: float = 2.0
    
    # YOLO
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_TARGET_DISTANCE: float = 0.40  # Stop distance for logo
    
    # Depth calculation
    DEPTH_SAMPLE_RADIUS: int = 5
    
    @property
    def center_threshold(self) -> int:
        return int(self.FRAME_WIDTH * self.CENTER_THRESHOLD_RATIO)
    
    @property
    def fov_margin(self) -> int:
        return int(self.FRAME_WIDTH * self.FOV_MARGIN_RATIO)


class NavigationMode(Enum):
    """Navigation mode enum"""
    ARUCO = "ARUCO"
    YOLO = "YOLO"
    COMPLETED = "COMPLETED"


class NavigationState(Enum):
    """Navigation state enum"""
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    TURNING = "TURNING"
    MOVING_AFTER_TURN = "MOVING_AFTER_TURN"
    TRACKING = "TRACKING"
    MISSION_COMPLETE = "MISSION_COMPLETE"


class TargetClass(Enum):
    """Target class for YOLO tracking"""
    AI = "AI"
    AWEAR = "AWEAR"
    IMR = "IMR"
    
    @classmethod
    def from_input(cls, choice: str) -> Optional['TargetClass']:
        """Convert user input to TargetClass"""
        mapping = {'1': cls.AI, '2': cls.AWEAR, '3': cls.IMR}
        return mapping.get(choice)


class Detection:
    """Base class for detections"""
    def __init__(self, center: Tuple[int, int], distance: Optional[float]):
        self.center = center
        self.distance = distance


class ArucoDetection(Detection):
    """ArUco marker detection"""
    def __init__(self, marker_id: int, corners: np.ndarray, center: Tuple[int, int],
                 distance: Optional[float], rvec: Optional[np.ndarray] = None,
                 tvec: Optional[np.ndarray] = None):
        super().__init__(center, distance)
        self.marker_id = marker_id
        self.corners = corners
        self.rvec = rvec
        self.tvec = tvec
        self.type = 'aruco'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility"""
        return {
            'type': 'aruco',
            'id': self.marker_id,
            'corners': self.corners,
            'center': self.center,
            'distance': self.distance,
            'rvec': self.rvec,
            'tvec': self.tvec
        }


class YoloDetection(Detection):
    """YOLO object detection"""
    def __init__(self, class_name: str, bbox: Tuple[int, int, int, int],
                 center: Tuple[int, int], distance: Optional[float]):
        super().__init__(center, distance)
        self.class_name = class_name
        self.bbox = bbox
        self.type = 'yolo'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility"""
        return {
            'type': 'yolo',
            'class': self.class_name,
            'bbox': self.bbox,
            'center': self.center,
            'distance': self.distance
        }


class HybridTracker:
    """Main tracker class for hybrid navigation"""
    
    def __init__(self, target_class: TargetClass, config: NavigationConfig = None):
        self.config = config or NavigationConfig()
        self.mode = NavigationMode.ARUCO
        self.target_class = target_class
        
        # RealSense Setup
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color,
            self.config.FRAME_WIDTH,
            self.config.FRAME_HEIGHT,
            rs.format.bgr8,
            30
        )
        self.rs_config.enable_stream(
            rs.stream.depth,
            self.config.FRAME_WIDTH,
            self.config.FRAME_HEIGHT,
            rs.format.z16,
            30
        )
        self.align = rs.align(rs.stream.color)
        self.running = False
        
        # Camera Intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.config.MARKER_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # YOLO Setup
        self.model = None
        self.model_ready = False
        self._model_lock = threading.Lock()
        threading.Thread(target=self._load_model, daemon=True).start()
        
        # Timing
        self.last_frame_time = None
        self.current_fps = 0.0
        self.last_infer_ms = 0.0
        
        # Navigation State
        self.nav_state = NavigationState.SEARCHING
        self.current_marker_id = None
        self.state_timer = 0
        self.initial_scan_done = False
        self.visited_markers = set()  # Track visited markers
        
        logger.info(f"HybridTracker initialized. Target: {target_class.value}")
        logger.info("Mission: Navigate ArUco maze (1-8) → Marker 9 (NO TURN) → YOLO logo tracking")
    
    def _load_model(self):
        """Load YOLO model asynchronously"""
        logger.info(f"Loading YOLO model: {self.config.YOLO_MODEL_PATH}")
        try:
            model = YOLO(self.config.YOLO_MODEL_PATH)
            with self._model_lock:
                self.model = model
                self.model_ready = True
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def start(self) -> bool:
        """Start the camera and initialize intrinsics"""
        try:
            profile = self.pipeline.start(self.rs_config)
            
            # Get camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.camera_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ], dtype=float)
            self.dist_coeffs = np.array(intr.coeffs)
            
            self.running = True
            logger.info("RealSense camera started")
            logger.info(f"Camera matrix:\n{self.camera_matrix}")
            return True
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        try:
            self.pipeline.stop()
            logger.info("Camera stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def get_distance_at_point(self, depth_frame, x: int, y: int,
                             radius: int = None) -> Optional[float]:
        """
        Calculate median distance at (x,y) within radius
        
        Args:
            depth_frame: RealSense depth frame
            x, y: Pixel coordinates
            radius: Sample radius (default from config)
            
        Returns:
            Median distance in meters or None if invalid
        """
        if depth_frame is None:
            return None
        
        radius = radius or self.config.DEPTH_SAMPLE_RADIUS
        
        # Boundary check
        if (x < radius or y < radius or 
            x >= self.config.FRAME_WIDTH - radius or 
            y >= self.config.FRAME_HEIGHT - radius):
            return None
        
        try:
            depth_data = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    d = depth_frame.get_distance(x + dx, y + dy)
                    if d > 0:
                        depth_data.append(d)
            
            return float(np.median(depth_data)) if depth_data else None
        except Exception as e:
            logger.warning(f"Error getting distance: {e}")
            return None
    
    def get_frame(self) -> Tuple[Optional[rs.frame], Optional[rs.frame]]:
        """Get aligned color and depth frames"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # Update FPS
            now = time.time()
            if self.last_frame_time:
                dt = now - self.last_frame_time
                if dt > 0:
                    self.current_fps = 1.0 / dt
            self.last_frame_time = now
            
            return color_frame, depth_frame
        except Exception as e:
            logger.warning(f"Error getting frame: {e}")
            return None, None
    
    def detect_aruco(self, color_image: np.ndarray,
                    depth_frame) -> List[ArucoDetection]:
        """
        Detect ArUco markers in the image
        
        Args:
            color_image: BGR image from camera
            depth_frame: Depth frame for distance calculation
            
        Returns:
            List of ArucoDetection objects
        """
        t0 = time.time()
        
        try:
            # Fixed: Use self.detector and correct variable names
            corners, ids, _ = self.detector.detectMarkers(color_image)
            self.last_infer_ms = (time.time() - t0) * 1000.0
            
            detections = []
            if ids is not None:
                # Estimate pose for all markers
                rvecs, tvecs = None, None
                if self.camera_matrix is not None:
                    try:
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners,
                            self.config.MARKER_LENGTH,
                            self.camera_matrix,
                            self.dist_coeffs
                        )
                    except Exception as e:
                        logger.warning(f"Pose estimation failed: {e}")
                
                ids = ids.flatten()
                for i, marker_id in enumerate(ids):
                    corner = corners[i][0]
                    cx = int(np.mean(corner[:, 0]))
                    cy = int(np.mean(corner[:, 1]))
                    
                    # Get rotation and translation vectors
                    rvec = rvecs[i] if rvecs is not None else None
                    tvec = tvecs[i] if tvecs is not None else None
                    
                    # Calculate distance: prefer tvec, fallback to depth
                    if tvec is not None:
                        distance = float(np.linalg.norm(tvec))
                    else:
                        distance = self.get_distance_at_point(depth_frame, cx, cy)
                    
                    detections.append(ArucoDetection(
                        marker_id=int(marker_id),
                        corners=corner.astype(int),
                        center=(cx, cy),
                        distance=distance,
                        rvec=rvec,
                        tvec=tvec
                    ))
            
            return detections
        except Exception as e:
            logger.error(f"ArUco detection error: {e}")
            return []
    
    def detect_yolo(self, color_image: np.ndarray,
                   depth_frame) -> List[YoloDetection]:
        """
        Detect objects using YOLO
        
        Args:
            color_image: BGR image from camera
            depth_frame: Depth frame for distance calculation
            
        Returns:
            List of YoloDetection objects
        """
        with self._model_lock:
            if not self.model_ready or self.model is None:
                return []
        
        t0 = time.time()
        
        try:
            results = self.model(
                color_image,
                conf=self.config.YOLO_CONFIDENCE,
                verbose=False
            )
            self.last_infer_ms = (time.time() - t0) * 1000.0
            
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Filter by target class
                    if class_name != self.target_class.value:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    distance = self.get_distance_at_point(depth_frame, cx, cy)
                    
                    detections.append(YoloDetection(
                        class_name=class_name,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        center=(cx, cy),
                        distance=distance
                    ))
            
            return detections
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def is_in_fov(self, detection: Detection) -> bool:
        """Check if detection is within field of view margin"""
        cx = detection.center[0]
        center_x = self.config.FRAME_WIDTH // 2
        return abs(cx - center_x) < self.config.fov_margin
    
    def get_navigation_command(self, target: Detection,
                              is_aruco: bool = False,
                              marker_id: int = None) -> str:
        """
        Unified command generation for both ArUco and YOLO
        
        Args:
            target: Detection object
            is_aruco: Whether this is an ArUco marker
            marker_id: Marker ID (for special handling of marker 9)
            
        Returns:
            Command string
        """
        if not target or not target.distance:
            return 'none'
        
        cx, _ = target.center
        distance = target.distance
        center_x = self.config.FRAME_WIDTH // 2
        
        # Check centering
        offset = cx - center_x
        if abs(offset) > self.config.center_threshold:
            return 'left' if offset < 0 else 'right'
        
        # Determine target distance based on marker type
        if is_aruco:
            if marker_id == 9:
                target_dist = self.config.MARKER_9_DISTANCE
            else:
                target_dist = self.config.TARGET_DISTANCE
        else:
            target_dist = self.config.YOLO_TARGET_DISTANCE
        
        # Check distance
        if distance > target_dist:
            if distance > self.config.BURST_FORWARD_DIST:
                return 'forward_burst'
            return 'forward'
        
        # At target distance
        if is_aruco and isinstance(target, ArucoDetection):
            if marker_id == 9:
                # Marker 9: Stop (no turn)
                return 'marker_9_reached'
            else:
                # Markers 1-8: Turn based on odd/even
                return ('action_turn_left' if target.marker_id % 2 != 0
                       else 'action_turn_right')
        
        return 'stop'
    
    def process_aruco_mode(self, detections: List[ArucoDetection],
                          current_time: float) -> Tuple[Optional[Detection], str]:
        """
        Process ArUco navigation mode
        
        Mission flow:
        1. Navigate markers 1-8 (odd->left, even->right)
        2. When marker 9 is reached (NO TURN) -> Switch to YOLO
        
        Returns:
            (target_detection, command)
        """
        # Navigation state machine
        if self.nav_state == NavigationState.SEARCHING:
            return self._handle_searching_state(detections)
        
        elif self.nav_state == NavigationState.APPROACHING:
            return self._handle_approaching_state(detections)
        
        elif self.nav_state == NavigationState.TURNING:
            return self._handle_turning_state(current_time)
        
        elif self.nav_state == NavigationState.MOVING_AFTER_TURN:
            return self._handle_moving_after_turn_state(current_time)
        
        return None, 'none'
    
    def _handle_searching_state(self, detections: List[ArucoDetection]
                               ) -> Tuple[Optional[Detection], str]:
        """Handle SEARCHING state"""
        valid_markers = []
        
        for d in detections:
            # Check valid distance
            if not d.distance or d.distance >= self.config.MAX_DISTANCE:
                continue
            
            # Initial scan: only look for markers 1 and 2
            if not self.initial_scan_done and d.marker_id not in [1, 2]:
                continue
            
            valid_markers.append(d)
        
        if valid_markers:
            self.initial_scan_done = True
            # Find closest marker
            target = min(valid_markers, key=lambda d: d.distance)
            self.current_marker_id = target.marker_id
            self.nav_state = NavigationState.APPROACHING
            logger.info(f"Found marker {self.current_marker_id}. Approaching.")
            command = self.get_navigation_command(target, is_aruco=True, marker_id=target.marker_id)
            return target, command
        
        return None, 'left'  # Keep scanning
    
    def _handle_approaching_state(self, detections: List[ArucoDetection]
                                 ) -> Tuple[Optional[Detection], str]:
        """Handle APPROACHING state"""
        target = next(
            (d for d in detections if d.marker_id == self.current_marker_id),
            None
        )
        
        if target:
            command = self.get_navigation_command(target, is_aruco=True, marker_id=target.marker_id)
            
            # Check if marker 9 reached
            if command == 'marker_9_reached':
                logger.info("=" * 60)
                logger.info("MARKER 9 REACHED! (NO TURN)")
                logger.info("Switching to YOLO mode for logo detection")
                logger.info(f"Target logo: {self.target_class.value}")
                logger.info("=" * 60)
                
                with self._model_lock:
                    if not self.model_ready:
                        logger.warning("YOLO model not ready yet. Waiting...")
                        return target, 'stop'
                
                # Switch to YOLO mode immediately
                self.visited_markers.add(self.current_marker_id)
                self.mode = NavigationMode.YOLO
                self.nav_state = NavigationState.TRACKING
                return target, 'stop'
            
            # Normal markers: check for turn command
            if command.startswith('action_turn'):
                self.nav_state = NavigationState.TURNING
                self.state_timer = time.time()
                self.visited_markers.add(self.current_marker_id)
                logger.info(f"Reached marker {self.current_marker_id}. "
                          f"Executing {command}")
            
            return target, command
        else:
            logger.warning("Lost marker during approach. Resuming search.")
            self.nav_state = NavigationState.SEARCHING
            return None, 'stop'
    
    def _handle_turning_state(self, current_time: float
                             ) -> Tuple[Optional[Detection], str]:
        """Handle TURNING state (only for markers 1-8, not marker 9)"""
        elapsed = current_time - self.state_timer
        
        if elapsed < self.config.TURN_DURATION:
            # Continue turning
            command = ('action_turn_left' if self.current_marker_id % 2 != 0
                      else 'action_turn_right')
            return None, command
        else:
            # Turn complete - move forward
            self.nav_state = NavigationState.MOVING_AFTER_TURN
            self.state_timer = current_time
            logger.info("Turn complete. Moving forward.")
            return None, 'forward'
    
    def _handle_moving_after_turn_state(self, current_time: float
                                       ) -> Tuple[Optional[Detection], str]:
        """Handle MOVING_AFTER_TURN state"""
        elapsed = current_time - self.state_timer
        
        if elapsed < self.config.MOVE_AFTER_TURN_DURATION:
            return None, 'forward'
        else:
            logger.info("Move complete. Resuming search.")
            self.nav_state = NavigationState.SEARCHING
            self.current_marker_id = None
            return None, 'stop'
    
    def process_yolo_mode(self, detections: List[YoloDetection]
                         ) -> Tuple[Optional[Detection], str]:
        """
        Process YOLO tracking mode
        
        Track target logo and navigate to it
        
        Returns:
            (target_detection, command)
        """
        if self.nav_state == NavigationState.MISSION_COMPLETE:
            return None, 'stop'
        
        if not detections:
            logger.debug("No target logo detected. Scanning...")
            return None, 'left'  # Slow rotation to find logo
        
        # Filter valid detections with distance
        valid_objs = [d for d in detections if d.distance is not None]
        
        if valid_objs:
            # Track closest object
            target = min(valid_objs, key=lambda d: d.distance)
            command = self.get_navigation_command(target, is_aruco=False)
            
            # Check if mission complete
            if (command == 'stop' and target.distance and 
                target.distance <= self.config.YOLO_TARGET_DISTANCE):
                logger.info("=" * 60)
                logger.info("MISSION COMPLETE!")
                logger.info(f"Reached target logo: {self.target_class.value}")
                logger.info(f"Final distance: {target.distance:.2f}m")
                logger.info("=" * 60)
                self.nav_state = NavigationState.MISSION_COMPLETE
                self.mode = NavigationMode.COMPLETED
                return target, 'stop'
            
            return target, command
        
        return None, 'left'
    
    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[Detection],
                                     List[Detection], str, Optional[float]]:
        """
        Main processing function
        
        Returns:
            (color_image, target, all_detections, command, center_depth)
        """
        color_frame, depth_frame = self.get_frame()
        if color_frame is None:
            return None, None, [], 'none', None
        
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get center depth for visualization
        center_depth = self.get_distance_at_point(
            depth_frame,
            self.config.FRAME_WIDTH // 2,
            self.config.FRAME_HEIGHT // 2
        )
        
        target = None
        command = 'none'
        current_time = time.time()
        
        if self.mode == NavigationMode.ARUCO:
            detections = self.detect_aruco(color_image, depth_frame)
            target, command = self.process_aruco_mode(detections, current_time)
        
        elif self.mode == NavigationMode.YOLO:
            detections = self.detect_yolo(color_image, depth_frame)
            target, command = self.process_yolo_mode(detections)
        
        elif self.mode == NavigationMode.COMPLETED:
            detections = []
            command = 'stop'
        
        else:
            detections = []
        
        # Convert to dict format for visualization compatibility
        detection_dicts = [d.to_dict() for d in detections]
        target_dict = target.to_dict() if target else None
        
        return color_image, target_dict, detection_dicts, command, center_depth


class RobotControllerWrapper:
    """Wrapper for robot control with timing and state management"""
    
    def __init__(self, robot, config: NavigationConfig = None):
        self.robot = robot
        self.config = config or NavigationConfig()
        self.last_command_time = 0
        self.current_command = None
        self.is_moving = False
        
        logger.info("RobotControllerWrapper initialized")
    
    def execute_command(self, command: str) -> bool:
        """
        Execute a movement command
        
        Args:
            command: Command string
            
        Returns:
            True if command was executed, False otherwise
        """
        if self.robot is None:
            return False
        
        if self.is_moving:
            return False
        
        current_time = time.time()
        
        # Cooldown check (except for stop)
        if (command != 'stop' and 
            current_time - self.last_command_time < self.config.CONTROL_COOLDOWN):
            return False
        
        # Avoid redundant 'none' commands
        if command == self.current_command and command == 'none':
            return False
        
        try:
            logger.debug(f"Executing command: {command}")
            self.is_moving = True
            
            if command == 'forward':
                self.robot.move_forward(fast=True)
            
            elif command == 'forward_burst':
                for _ in range(self.config.BURST_REPEAT):
                    self.robot.move_forward(fast=True)
                    time.sleep(0.1)
            
            elif command == 'left':
                self.robot.turn_left()
            
            elif command == 'right':
                self.robot.turn_right()
            
            elif command == 'action_turn_left':
                for _ in range(self.config.BURST_REPEAT):
                    self.robot.turn_left()
                    time.sleep(0.1)
            
            elif command == 'action_turn_right':
                for _ in range(self.config.BURST_REPEAT):
                    self.robot.turn_right()
                    time.sleep(0.1)
            
            elif command == 'stop' or command == 'marker_9_reached':
                pass  # Robot naturally stops
            
            self.last_command_time = current_time
            self.current_command = command
            return True
        
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            return False
        
        finally:
            self.is_moving = False


def visualize(frame: np.ndarray, target: Optional[Dict],
             detections: List[Dict], command: str, fps: float,
             mode: NavigationMode, target_class: TargetClass,
             center_depth: Optional[float], config: NavigationConfig,
             visited_markers: set,
             camera_matrix: Optional[np.ndarray] = None,
             dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Visualize detection results with comprehensive overlay
    
    Args:
        frame: Input BGR image
        target: Target detection dict or None
        detections: List of all detection dicts
        command: Current navigation command
        fps: Current FPS
        mode: Current navigation mode
        target_class: Target class for YOLO
        center_depth: Depth at image center
        config: Navigation configuration
        visited_markers: Set of visited marker IDs
        camera_matrix: Camera intrinsics for axis drawing
        dist_coeffs: Distortion coefficients
        
    Returns:
        Annotated image
    """
    disp = frame.copy()
    h, w = disp.shape[:2]
    cx0, cy0 = w // 2, h // 2
    
    # Draw center crosshair
    cv2.line(disp, (cx0 - 15, cy0), (cx0 + 15, cy0), (0, 0, 255), 2)
    cv2.line(disp, (cx0, cy0 - 15), (cx0, cy0 + 15), (0, 0, 255), 2)
    
    # Center depth display
    if center_depth:
        cv2.putText(disp, f"Center: {center_depth:.2f}m",
                   (cx0 + 20, cy0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Draw guide lines
    cv2.line(disp, (cx0, 0), (cx0, h), (255, 255, 255), 1)
    
    # Center threshold lines
    ct = config.center_threshold
    cv2.line(disp, (cx0 - ct, 0), (cx0 - ct, h), (0, 255, 0), 1)
    cv2.line(disp, (cx0 + ct, 0), (cx0 + ct, h), (0, 255, 0), 1)
    
    # FOV margin lines
    fov = config.fov_margin
    cv2.line(disp, (cx0 - fov, 0), (cx0 - fov, h), (255, 0, 0), 1)
    cv2.line(disp, (cx0 + fov, 0), (cx0 + fov, h), (255, 0, 0), 1)
    
    # Draw all detections
    for d in detections:
        is_target = (target is not None and d == target)
        color = (0, 255, 0) if is_target else (0, 165, 255)
        
        if d['type'] == 'aruco':
            marker_id = d['id']
            is_visited = marker_id in visited_markers
            
            # Special color for marker 9
            if marker_id == 9:
                color = (255, 0, 255) if not is_target else (0, 255, 0)  # Magenta for marker 9
            elif is_visited:
                color = (128, 128, 128) if not is_target else (0, 255, 0)
            
            # Draw marker boundary
            corners = d['corners'].reshape((-1, 1, 2))
            cv2.polylines(disp, [corners], True, color, 2)
            
            # Draw coordinate axes if pose is available
            if (camera_matrix is not None and dist_coeffs is not None and
                d.get('rvec') is not None and d.get('tvec') is not None):
                try:
                    axis_len = config.MARKER_LENGTH * 0.75
                    cv2.drawFrameAxes(disp, camera_matrix, dist_coeffs,
                                     d['rvec'], d['tvec'], axis_len)
                except Exception as e:
                    logger.debug(f"Could not draw axes: {e}")
            
            # Draw ID and distance on marker
            mcx, mcy = d['center']
            dist = d.get('distance')
            
            if dist is None:
                text = f"ID {marker_id} | N/A"
            else:
                text = f"ID {marker_id} | {dist:.2f}m"
            
            if is_visited:
                text += " ✓"
            if marker_id == 9:
                text += " [FINAL]"
            
            # Text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            
            tx = int(mcx - tw / 2)
            ty = int(mcy + th / 2)
            
            pad = 4
            cv2.rectangle(disp,
                         (tx - pad, ty - th - pad),
                         (tx + tw + pad, ty + baseline + pad),
                         (0, 0, 0), -1)
            cv2.putText(disp, text, (tx, ty), font, scale,
                       (255, 255, 255), thickness, cv2.LINE_AA)
        
        elif d['type'] == 'yolo':
            # Draw bounding box
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            dist = d.get('distance')
            label = f"{d['class']} | {dist:.2f}m" if dist else d['class']
            cv2.putText(disp, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # Draw HUD (status overlay)
    y_offset = h - 150
    
    # Mode display
    mode_color = (255, 255, 0) if mode == NavigationMode.ARUCO else (0, 255, 255)
    if mode == NavigationMode.COMPLETED:
        mode_color = (0, 255, 0)
    cv2.putText(disp, f"MODE: {mode.value}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)
    
    y_offset += 30
    cv2.putText(disp, f"CMD: {command.upper()}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    y_offset += 30
    if mode == NavigationMode.YOLO or mode == NavigationMode.COMPLETED:
        cv2.putText(disp, f"TARGET: {target_class.value}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    y_offset += 30
    # Visited markers
    if visited_markers:
        visited_str = f"Visited: {sorted(visited_markers)}"
        cv2.putText(disp, visited_str, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # FPS display
    cv2.putText(disp, f"FPS: {fps:.1f}", (w - 150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return disp


def main():
    """Main execution function"""
    print("=" * 70)
    print(" Improved Hybrid Navigation Robot")
    print("=" * 70)
    print(" Mission Flow:")
    print("  1. ARUCO MODE: Navigate maze using markers 1-8")
    print("     - Odd markers (1,3,5,7) → Turn LEFT")
    print("     - Even markers (2,4,6,8) → Turn RIGHT")
    print("  2. MARKER 9: Final ArUco marker → just switch to YOLO")
    print("  3. YOLO MODE: Detect and approach target logo")
    print("=" * 70)
    
    # Select target class
    target_class = None
    while target_class is None:
        print("\nSelect Target Logo for YOLO Phase:")
        print(" 1: AI")
        print(" 2: AWEAR")
        print(" 3: IMR")
        choice = input("Enter choice (1/2/3): ").strip()
        target_class = TargetClass.from_input(choice)
        if not target_class:
            print("Invalid choice. Please try again.")
    
    logger.info(f"Selected target logo: {target_class.value}")
    
    # Initialize configuration
    config = NavigationConfig()
    
    # Initialize robot
    robot = None
    if ROBOT_AVAILABLE:
        logger.info("Connecting to robot...")
        try:
            ctrl = DynamixelController(DEVICENAME, BAUDRATE)
            if not ctrl.connect():
                logger.error("Failed to connect to Dynamixel")
            else:
                robot = QuadrupedRobot(ctrl, LEG_IDS)
                robot.enable_all_torque()
                robot.initialize_pose()
                time.sleep(1)
                robot.stand_pose()
                logger.info("Robot connected and initialized")
        except Exception as e:
            logger.error(f"Robot initialization failed: {e}")
    else:
        logger.warning("Running in visualization-only mode")
    
    # Initialize tracker
    tracker = HybridTracker(target_class, config)
    if not tracker.start():
        logger.error("Failed to start tracker")
        sys.exit(1)
    
    # Initialize robot controller
    robot_ctrl = RobotControllerWrapper(robot, config)
    
    frame_count = 0
    robot_enabled = True
    
    try:
        logger.info("Starting main loop. Press 'q' to quit, 's' to toggle robot control.")
        
        while True:
            frame_count += 1
            
            # Process frame
            img, target, detections, command, center_depth = tracker.process_frame()
            if img is None:
                continue
            
            # Execute robot command (throttled by frame interval)
            if robot and robot_enabled and (frame_count % config.CONTROL_FRAME_INTERVAL == 0):
                robot_ctrl.execute_command(command)
            
            # Visualize
            disp = visualize(
                img, target, detections, command,
                tracker.current_fps, tracker.mode, tracker.target_class,
                center_depth, config, tracker.visited_markers,
                tracker.camera_matrix, tracker.dist_coeffs
            )
            
            # Robot status overlay
            status_text = "ENABLED" if robot_enabled else "DISABLED"
            status_color = (0, 255, 0) if robot_enabled else (0, 0, 255)
            cv2.putText(disp, f"ROBOT: {status_text}",
                       (disp.shape[1] - 220, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            
            # Mission complete banner
            if tracker.mode == NavigationMode.COMPLETED:
                overlay = disp.copy()
                cv2.rectangle(overlay, (50, disp.shape[0]//2 - 50),
                            (disp.shape[1] - 50, disp.shape[0]//2 + 50),
                            (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
                cv2.putText(disp, "MISSION COMPLETE!", 
                          (disp.shape[1]//2 - 200, disp.shape[0]//2),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Display
            cv2.imshow("Hybrid Navigation Robot", disp)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit command received")
                break
            elif key == ord('s'):
                robot_enabled = not robot_enabled
                logger.info(f"Robot control: {'ENABLED' if robot_enabled else 'DISABLED'}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    finally:
        logger.info("Cleaning up...")
        tracker.stop()
        cv2.destroyAllWindows()
        
        if robot:
            try:
                robot.disable_all_torque()
                ctrl.disconnect()
                logger.info("Robot disconnected")
            except Exception as e:
                logger.error(f"Error during robot cleanup: {e}")
        
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
