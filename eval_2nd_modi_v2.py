#!/usr/bin/env python3
"""
Quadruped Robot Controller - Improved Version
Dynamixel SDK를 사용한 4족 로봇 제어 프로그램
"""

import time
import sys
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from dynamixel_sdk import *

# --- PLATFORM-SPECIFIC IMPORTS ---
try:
    import msvcrt
    _WINDOWS = True
except ImportError:
    import termios
    import tty
    _WINDOWS = False


# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- CONSTANTS ---

# Communication Settings
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

# Motor Address (Protocol 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4

# Motor Configuration
MOTORS_PER_LEG = 3
LEG_IDS = {
    1: [0, 1, 2],    # Front-Right
    2: [3, 4, 5],    # Back-Right
    3: [6, 7, 8],    # Back-Left
    4: [9, 10, 11]   # Front-Left
}
ALL_MOTOR_IDS = [id for ids in LEG_IDS.values() for id in ids]

# Position Limits (safety bounds)
POSITION_MIN = 0
POSITION_MAX = 4095

# Interpolation Settings
INTERPOLATION_STEPS = 25

# Gait Timing (seconds)
NORMAL_POWER_DURATION = 0.3
NORMAL_RETURN_DURATION = 0.45  # 1.5x power duration
FAST_POWER_DURATION = 0.24     # 0.8x normal
FAST_RETURN_DURATION = 0.36    # 0.8x normal


# --- ENUMS ---

class RobotState(Enum):
    """로봇의 현재 상태"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    STANDING = "standing"
    WALKING = "walking"
    TURNING = "turning"
    ERROR = "error"


class GaitSpeed(Enum):
    """보행 속도"""
    NORMAL = "normal"
    FAST = "fast"


# --- DATA CLASSES ---

@dataclass
class Position:
    """3D 위치를 나타내는 데이터 클래스"""
    joint1: int
    joint2: int
    joint3: int
    
    def to_list(self) -> List[int]:
        return [self.joint1, self.joint2, self.joint3]
    
    @classmethod
    def from_list(cls, pos_list: List[int]) -> 'Position':
        if len(pos_list) != 3:
            raise ValueError(f"Position requires 3 values, got {len(pos_list)}")
        return cls(pos_list[0], pos_list[1], pos_list[2])
    
    def validate(self) -> bool:
        """위치가 안전 범위 내에 있는지 확인"""
        return all(
            POSITION_MIN <= val <= POSITION_MAX 
            for val in [self.joint1, self.joint2, self.joint3]
        )


class Poses:
    """모든 정적 자세 및 보행 시퀀스 위치 정의"""
    
    # Static Poses
    INITIAL = Position(2048, 2048, 2048)
    START = Position(2048, 2141, 2228)
    
    # Forward Gait Positions
    FORWARD_BASE = Position(2048, 2141, 2228)
    FORWARD_INTERM_A = Position(2086, 1681, 2202)
    FORWARD_INTERM_B = Position(2009, 1681, 2202)
    
    FORWARD_POS_11 = Position(2418, 2432, 1409)
    FORWARD_POS_12 = Position(1754, 2057, 2413)
    FORWARD_POS_13 = Position(1677, 2432, 1409)
    FORWARD_POS_14 = Position(2341, 2057, 2413)
    
    BACKWARD_POS_11 = Position(1754, 2057, 2413)
    BACKWARD_POS_12 = Position(2418, 2432, 1409)
    BACKWARD_POS_13 = Position(2341, 2057, 2413)
    BACKWARD_POS_14 = Position(1677, 2432, 1409)
    
    # Turn Positions
    TURN_NEUTRAL = Position(2048, 2141, 2228)
    
    # Left Turn
    LEFT_STEP = Position(2200, 2141, 2228)
    LEFT_INTERM = Position(2100, 1804, 2094)
    
    # Right Turn
    RIGHT_STEP = Position(1895, 2141, 2228)
    RIGHT_INTERM = Position(1995, 1804, 2094)
    
    # Walk Left Positions (왼쪽 다리 4,3을 앞다리로 취급)
    # move_forward 매핑: 1→4, 4→3, 2→1, 3→2
    WALK_LEFT_INTERM_A = Position(2086, 1681, 2202)  # 다리 4,3 들기용
    WALK_LEFT_INTERM_B = Position(2009, 1681, 2202)  # 다리 1,2 들기용
    
    # Power stroke positions (forward에서 회전)
    WALK_LEFT_POS_4 = Position(1677, 2432, 1409)  # 다리4 = forward의 다리1 
    WALK_LEFT_POS_3 = Position(1754, 2057, 2413)  # 다리3 = forward의 다리4
    WALK_LEFT_POS_1 = Position(2418, 2432, 1409)  # 다리1 = forward의 다리2
    WALK_LEFT_POS_2 = Position(2341, 2057, 2413)  # 다리2 = forward의 다리3
    
    # Return stroke positions (forward에서 회전)
    WALK_LEFT_BACK_4 = Position(2341, 2057, 2413)  # 다리4 = forward의 다리1 backward
    WALK_LEFT_BACK_3 = Position(2418, 2432, 1409)  # 다리3 = forward의 다리4 backward
    WALK_LEFT_BACK_1 = Position(1754, 2057, 2413)  # 다리1 = forward의 다리2 backward
    WALK_LEFT_BACK_2 = Position(1677, 2432, 1409)  # 다리2 = forward의 다리3 backward
    
    # Walk Right Positions (오른쪽 다리 1,2를 앞다리로 취급)
    # move_forward 매핑: forward의 4→1, 3→2, 1→4, 2→3 (반시계 회전)
    WALK_RIGHT_INTERM_A = Position(2086, 1681, 2202)  # 다리 1,2 들기용
    WALK_RIGHT_INTERM_B = Position(2009, 1681, 2202)  # 다리 4,3 들기용
    
    # Power stroke positions (forward에서 회전)
    WALK_RIGHT_POS_1 = Position(1754, 2057, 2413)  # 다리1 = forward의 다리4
    WALK_RIGHT_POS_2 = Position(2341, 2057, 2413)  # 다리2 = forward의 다리3
    WALK_RIGHT_POS_4 = Position(1677, 2432, 1409)  # 다리4 = forward의 다리1
    WALK_RIGHT_POS_3 = Position(2418, 2432, 1409)  # 다리3 = forward의 다리2
    
    # Return stroke positions (forward에서 회전)
    WALK_RIGHT_BACK_1 = Position(2418, 2432, 1409)  # 다리1 = forward의 다리4 backward
    WALK_RIGHT_BACK_2 = Position(1677, 2432, 1409)  # 다리2 = forward의 다리3 backward
    WALK_RIGHT_BACK_4 = Position(2341, 2057, 2413)  # 다리4 = forward의 다리1 backward
    WALK_RIGHT_BACK_3 = Position(1754, 2057, 2413)  # 다리3 = forward의 다리2 backward


@dataclass
class GaitStep:
    """보행 단계를 나타내는 데이터 클래스"""
    movements: Dict[int, Position]  # {leg_num: target_position}
    duration_type: str  # 'power' or 'return'
    description: str = ""


# --- GAIT PATTERNS ---

class GaitPatterns:
    """보행 패턴 정의"""
    
    @staticmethod
    def forward_gait() -> List[GaitStep]:
        """전진 보행 패턴"""
        return [
            GaitStep(
                movements={
                    2: Poses.FORWARD_INTERM_A,
                    4: Poses.FORWARD_INTERM_A,
                },
                duration_type='return',
                description="Lift legs 2,4"
            ),
            GaitStep(
                movements={
                    1: Poses.FORWARD_POS_13,
                    2: Poses.FORWARD_POS_11,
                    3: Poses.FORWARD_POS_14,
                    4: Poses.FORWARD_POS_12,
                },
                duration_type='power',
                description="Power stroke"
            ),
            GaitStep(
                movements={
                    1: Poses.FORWARD_INTERM_B,
                    3: Poses.FORWARD_INTERM_B,
                },
                duration_type='return',
                description="Lift legs 1,3"
            ),
            GaitStep(
                movements={
                    1: Poses.BACKWARD_POS_13,
                    2: Poses.BACKWARD_POS_11,
                    3: Poses.BACKWARD_POS_14,
                    4: Poses.BACKWARD_POS_12,
                },
                duration_type='power',
                description="Return stroke"
            ),
        ]
    
    @staticmethod
    def turn_left_gait() -> List[GaitStep]:
        """좌회전 패턴 (Trot)"""
        return [
            GaitStep(
                movements={
                    2: Poses.LEFT_INTERM,
                    4: Poses.LEFT_INTERM,
                    1: Poses.TURN_NEUTRAL,
                    3: Poses.TURN_NEUTRAL,
                },
                duration_type='return',
                description="Lift pair 2,4"
            ),
            GaitStep(
                movements={
                    2: Poses.LEFT_STEP,
                    4: Poses.LEFT_STEP,
                },
                duration_type='power',
                description="Place pair 2,4"
            ),
            GaitStep(
                movements={
                    1: Poses.LEFT_INTERM,
                    3: Poses.LEFT_INTERM,
                    2: Poses.TURN_NEUTRAL,
                    4: Poses.TURN_NEUTRAL,
                },
                duration_type='return',
                description="Lift pair 1,3"
            ),
            GaitStep(
                movements={
                    1: Poses.LEFT_STEP,
                    3: Poses.LEFT_STEP,
                },
                duration_type='power',
                description="Place pair 1,3"
            ),
        ]
    
    @staticmethod
    def turn_right_gait() -> List[GaitStep]:
        """우회전 패턴 (Trot)"""
        return [
            GaitStep(
                movements={
                    2: Poses.RIGHT_INTERM,
                    4: Poses.RIGHT_INTERM,
                    1: Poses.TURN_NEUTRAL,
                    3: Poses.TURN_NEUTRAL,
                },
                duration_type='return',
                description="Lift pair 2,4"
            ),
            GaitStep(
                movements={
                    2: Poses.RIGHT_STEP,
                    4: Poses.RIGHT_STEP,
                },
                duration_type='power',
                description="Place pair 2,4"
            ),
            GaitStep(
                movements={
                    1: Poses.RIGHT_INTERM,
                    3: Poses.RIGHT_INTERM,
                    2: Poses.TURN_NEUTRAL,
                    4: Poses.TURN_NEUTRAL,
                },
                duration_type='return',
                description="Lift pair 1,3"
            ),
            GaitStep(
                movements={
                    1: Poses.RIGHT_STEP,
                    3: Poses.RIGHT_STEP,
                },
                duration_type='power',
                description="Place pair 1,3"
            ),
        ]
    
    @staticmethod
    def sidestep_left_gait() -> List[GaitStep]:
        """
        왼쪽으로 걷기 - 왼쪽 다리(4,3)을 앞다리로 취급한 보행
        
        로봇 배치:     4(FL)---1(FR)  (앞)
                      |           |
                      3(BL)---2(BR)  (뒤)
        
        왼쪽 이동 시:  4,3(앞) --- 1,2(뒤)
        move_forward의 Trot 패턴 유지 (대각선으로 들기)
        
        forward: 2,4 들기 / 1,3 들기
        left:    1,3 들기 / 2,4 들기  (역순)
        """
        return [
            # Step 1: 다리 1,3 들기 (대각선 - forward의 2,4에 해당)
            GaitStep(
                movements={
                    1: Poses.WALK_LEFT_INTERM_A,  # Front-Right
                    3: Poses.WALK_LEFT_INTERM_A,  # Back-Left
                },
                duration_type='return',
                description="Lift legs 1,3 (diagonal pair)"
            ),
            
            # Step 2: Power stroke - 모든 다리 동시 이동
            GaitStep(
                movements={
                    4: Poses.WALK_LEFT_POS_4,  # 다리4 (forward의 다리1 역할)
                    3: Poses.WALK_LEFT_POS_3,  # 다리3 (forward의 다리4 역할)
                    1: Poses.WALK_LEFT_POS_1,  # 다리1 (forward의 다리2 역할)
                    2: Poses.WALK_LEFT_POS_2,  # 다리2 (forward의 다리3 역할)
                },
                duration_type='power',
                description="Power stroke - all legs push left"
            ),
            
            # Step 3: 다리 2,4 들기 (대각선 - forward의 1,3에 해당)
            GaitStep(
                movements={
                    2: Poses.WALK_LEFT_INTERM_B,  # Back-Right
                    4: Poses.WALK_LEFT_INTERM_B,  # Front-Left
                },
                duration_type='return',
                description="Lift legs 2,4 (diagonal pair)"
            ),
            
            # Step 4: Return stroke - 모든 다리 복귀
            GaitStep(
                movements={
                    4: Poses.WALK_LEFT_BACK_4,  # 다리4
                    3: Poses.WALK_LEFT_BACK_3,  # 다리3
                    1: Poses.WALK_LEFT_BACK_1,  # 다리1
                    2: Poses.WALK_LEFT_BACK_2,  # 다리2
                },
                duration_type='power',
                description="Return stroke - complete cycle"
            ),
        ]
    
    @staticmethod
    def sidestep_right_gait() -> List[GaitStep]:
        """
        오른쪽으로 걷기 - 오른쪽 다리(1,2)를 앞다리로 취급한 보행
        
        로봇 배치:     4(FL)---1(FR)  (앞)
                      |           |
                      3(BL)---2(BR)  (뒤)
        
        오른쪽 이동 시: 1,2(앞) --- 4,3(뒤)
        move_forward를 반시계 회전: 다리번호 1→4, 2→3, 3→2, 4→1
        
        forward: 2,4 들기 / 1,3 들기
        right:   3,1 들기 / 4,2 들기 (회전된 대각선)
        """
        return [
            # Step 1: 다리 3,1 들기 (대각선 - 뒤3 + 앞1)
            GaitStep(
                movements={
                    3: Poses.WALK_RIGHT_INTERM_A,  # Back-Left (뒷다리)
                    1: Poses.WALK_RIGHT_INTERM_A,  # Front-Right (앞다리)
                },
                duration_type='return',
                description="Lift legs 3,1 (diagonal pair)"
            ),
            
            # Step 2: Power stroke
            GaitStep(
                movements={
                    1: Poses.WALK_RIGHT_POS_1,  # 다리1 = forward의 다리4 역할
                    2: Poses.WALK_RIGHT_POS_2,  # 다리2 = forward의 다리3 역할
                    4: Poses.WALK_RIGHT_POS_4,  # 다리4 = forward의 다리1 역할
                    3: Poses.WALK_RIGHT_POS_3,  # 다리3 = forward의 다리2 역할
                },
                duration_type='power',
                description="Power stroke - all legs push right"
            ),
            
            # Step 3: 다리 4,2 들기 (대각선 - 뒤4 + 앞2)
            GaitStep(
                movements={
                    4: Poses.WALK_RIGHT_INTERM_B,  # Front-Left (뒷다리)
                    2: Poses.WALK_RIGHT_INTERM_B,  # Back-Right (앞다리)
                },
                duration_type='return',
                description="Lift legs 4,2 (diagonal pair)"
            ),
            
            # Step 4: Return stroke
            GaitStep(
                movements={
                    1: Poses.WALK_RIGHT_BACK_1,  # 다리1
                    2: Poses.WALK_RIGHT_BACK_2,  # 다리2
                    4: Poses.WALK_RIGHT_BACK_4,  # 다리4
                    3: Poses.WALK_RIGHT_BACK_3,  # 다리3
                },
                duration_type='power',
                description="Return stroke - complete cycle"
            ),
        ]


# --- UTILITY FUNCTIONS ---

def get_keypress() -> Optional[str]:
    """
    키 입력 감지 (Windows / POSIX 모두 지원)
    
    Returns:
        입력된 1글자 또는 None (특수키의 경우)
    """
    if _WINDOWS:
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            _ = msvcrt.getch()
            return None
        try:
            return ch.decode('utf-8')
        except UnicodeDecodeError:
            return None
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def print_controls():
    """조작 키 안내 출력"""
    print("\n" + "="*60)
    print("Quadruped Robot Controller - Improved Version")
    print("="*60)
    print(f"  [i] Initialize Pose      -> {Poses.INITIAL.to_list()}")
    print(f"  [s] Stand/Start Pose     -> {Poses.START.to_list()}")
    print("  [w] Move Forward         -> Normal Speed (다리 1,4 앞)")
    print("  [W] Move Forward (Fast)  -> High Speed")
    print("  [a] Turn Left            -> Trot Pattern (제자리 회전)")
    print("  [d] Turn Right           -> Trot Pattern (제자리 회전)")
    print("  [z] Walk Left            -> 다리 4,3을 앞다리로 (왼쪽 이동)")
    print("  [c] Walk Right           -> 다리 1,2를 앞다리로 (오른쪽 이동)")
    print("  [t] Disable All Torque   -> Safety Release")
    print("  [q] Quit Program         -> Exit and Disconnect")
    print("="*60 + "\n")


# --- DYNAMIXEL CONTROLLER ---

class DynamixelController:
    """Dynamixel SDK와의 저수준 통신 관리"""
    
    def __init__(self, device_name: str, baudrate: int):
        self.device_name = device_name
        self.baudrate = baudrate
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.group_sync_write = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )
        self.is_connected = False
    
    def connect(self) -> bool:
        """포트 연결 및 보드레이트 설정"""
        try:
            if not self.port_handler.openPort():
                logger.error(f"Failed to open port: {self.device_name}")
                return False
            
            if not self.port_handler.setBaudRate(self.baudrate):
                logger.error(f"Failed to set baudrate: {self.baudrate}")
                return False
            
            self.is_connected = True
            logger.info(f"Connected to {self.device_name} at {self.baudrate} baud")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """포트 연결 해제"""
        if self.is_connected:
            self.port_handler.closePort()
            self.is_connected = False
            logger.info("Dynamixel port disconnected")
    
    def _write_1byte_txrx(self, motor_id: int, address: int, value: int) -> bool:
        """1바이트 쓰기 명령"""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, address, value
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            logger.error(
                f"Motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
            return False
        elif dxl_error != 0:
            logger.error(
                f"Motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}"
            )
            return False
        return True
    
    def enable_torque(self, motor_ids: List[int]) -> bool:
        """모터 토크 활성화"""
        success = True
        for motor_id in motor_ids:
            if not self._write_1byte_txrx(motor_id, ADDR_TORQUE_ENABLE, 1):
                success = False
        return success
    
    def disable_torque(self, motor_ids: List[int]) -> bool:
        """모터 토크 비활성화"""
        success = True
        for motor_id in motor_ids:
            if not self._write_1byte_txrx(motor_id, ADDR_TORQUE_ENABLE, 0):
                success = False
        return success
    
    def sync_write_goal_position(self, motor_positions: Dict[int, int]) -> bool:
        """
        여러 모터에 목표 위치 동시 전송
        
        Args:
            motor_positions: {motor_id: position} 딕셔너리
            
        Returns:
            성공 여부
        """
        self.group_sync_write.clearParam()
        
        for motor_id, position in motor_positions.items():
            # 안전 범위 체크
            if not (POSITION_MIN <= position <= POSITION_MAX):
                logger.warning(
                    f"Motor {motor_id} position {position} out of range "
                    f"[{POSITION_MIN}, {POSITION_MAX}]"
                )
                continue
            
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position)),
                DXL_HIBYTE(DXL_LOWORD(position)),
                DXL_LOBYTE(DXL_HIWORD(position)),
                DXL_HIBYTE(DXL_HIWORD(position)),
            ]
            
            if not self.group_sync_write.addParam(motor_id, param_goal_position):
                logger.error(f"Motor {motor_id}: addParam failed")
                return False
        
        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            logger.error(
                f"Sync write failed: "
                f"{self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
            return False
        
        return True


# --- QUADRUPED ROBOT ---

class QuadrupedRobot:
    """4족 로봇의 상태와 고수준 동작 관리"""
    
    def __init__(
        self, 
        controller: DynamixelController, 
        leg_ids: Dict[int, List[int]],
        interpolation_steps: int = INTERPOLATION_STEPS
    ):
        self.controller = controller
        self.leg_ids = leg_ids
        self.all_motor_ids = [id for ids in leg_ids.values() for id in ids]
        self.interpolation_steps = interpolation_steps
        self.state = RobotState.IDLE
        
        # 현재 각 다리의 위치
        self.current_positions: Dict[int, Position] = {
            leg_num: Poses.INITIAL for leg_num in self.leg_ids.keys()
        }
    
    def enable_all_torque(self) -> bool:
        """모든 모터 토크 활성화"""
        logger.info("Enabling torque for all motors...")
        success = self.controller.enable_torque(self.all_motor_ids)
        if success:
            logger.info("All motors enabled")
        else:
            logger.error("Failed to enable some motors")
            self.state = RobotState.ERROR
        return success
    
    def disable_all_torque(self) -> bool:
        """모든 모터 토크 비활성화"""
        logger.info("Disabling torque for all motors...")
        success = self.controller.disable_torque(self.all_motor_ids)
        if success:
            logger.info("All motors disabled")
            self.state = RobotState.IDLE
        return success
    
    def _get_duration(self, duration_type: str, speed: GaitSpeed) -> float:
        """보행 타입과 속도에 따른 duration 반환"""
        if speed == GaitSpeed.FAST:
            return (
                FAST_POWER_DURATION if duration_type == 'power' 
                else FAST_RETURN_DURATION
            )
        else:
            return (
                NORMAL_POWER_DURATION if duration_type == 'power' 
                else NORMAL_RETURN_DURATION
            )
    
    def move_to_pose_immediate(self, target_pos: Position) -> bool:
        """
        모든 다리를 즉시 target_pos로 이동 (보간 없음)
        
        Args:
            target_pos: 목표 위치
            
        Returns:
            성공 여부
        """
        if not target_pos.validate():
            logger.error(f"Invalid target position: {target_pos}")
            return False
        
        motor_positions = {}
        for leg_num, motor_ids in self.leg_ids.items():
            pos_list = target_pos.to_list()
            for i in range(MOTORS_PER_LEG):
                motor_positions[motor_ids[i]] = pos_list[i]
        
        if self.controller.sync_write_goal_position(motor_positions):
            # 상태 업데이트
            for leg_num in self.leg_ids.keys():
                self.current_positions[leg_num] = target_pos
            return True
        
        return False
    
    def _move_legs_interpolated(
        self, 
        movements: Dict[int, Position], 
        duration: float
    ) -> bool:
        """
        지정된 다리들을 보간하여 부드럽게 이동
        
        Args:
            movements: {leg_num: target_position} 딕셔너리
            duration: 이동 시간 (초)
            
        Returns:
            성공 여부
        """
        # 입력 검증
        for leg_num, target_pos in movements.items():
            if leg_num not in self.leg_ids:
                logger.error(f"Invalid leg number: {leg_num}")
                return False
            if not target_pos.validate():
                logger.error(f"Invalid position for leg {leg_num}: {target_pos}")
                return False
        
        # 시작 위치 저장
        start_positions = {
            leg_num: self.current_positions[leg_num]
            for leg_num in movements.keys()
        }
        
        steps = self.interpolation_steps
        step_time = duration / steps
        
        for step in range(1, steps + 1):
            start_time = time.time()
            motor_positions = {}
            
            for leg_num, target_pos in movements.items():
                leg_motor_ids = self.leg_ids[leg_num]
                start_pos = start_positions[leg_num].to_list()
                target_pos_list = target_pos.to_list()
                
                for i in range(MOTORS_PER_LEG):
                    start = start_pos[i]
                    end = target_pos_list[i]
                    current_pos = int(start + (end - start) * (step / steps))
                    motor_id = leg_motor_ids[i]
                    motor_positions[motor_id] = current_pos
            
            if not self.controller.sync_write_goal_position(motor_positions):
                logger.error(f"Failed at interpolation step {step}/{steps}")
                return False
            
            # 타이밍 조절
            elapsed = time.time() - start_time
            if elapsed < step_time:
                time.sleep(step_time - elapsed)
            else:
                logger.debug(f"Step {step} took {elapsed:.3f}s (target: {step_time:.3f}s)")
        
        # 최종 상태 업데이트
        for leg_num, target_pos in movements.items():
            self.current_positions[leg_num] = target_pos
        
        return True
    
    def _execute_gait_pattern(
        self, 
        gait_steps: List[GaitStep], 
        speed: GaitSpeed,
        pattern_name: str
    ) -> bool:
        """
        보행 패턴 실행
        
        Args:
            gait_steps: 보행 단계 리스트
            speed: 보행 속도
            pattern_name: 패턴 이름 (로깅용)
            
        Returns:
            성공 여부
        """
        logger.info(f"Executing {pattern_name} ({speed.value} speed)")
        
        for idx, step in enumerate(gait_steps, 1):
            duration = self._get_duration(step.duration_type, speed)
            logger.debug(f"Step {idx}/{len(gait_steps)}: {step.description}")
            
            if not self._move_legs_interpolated(step.movements, duration):
                logger.error(f"Failed at step {idx}: {step.description}")
                self.state = RobotState.ERROR
                return False
        
        logger.info(f"{pattern_name} completed successfully")
        return True
    
    def initialize_pose(self) -> bool:
        """초기 자세로 이동"""
        self.state = RobotState.INITIALIZING
        logger.info(f"Moving to Initial Pose: {Poses.INITIAL.to_list()}")
        success = self.move_to_pose_immediate(Poses.INITIAL)
        if success:
            self.state = RobotState.IDLE
            logger.info("Initialization complete")
        else:
            self.state = RobotState.ERROR
        return success
    
    def stand_pose(self) -> bool:
        """서기 자세로 이동"""
        self.state = RobotState.STANDING
        logger.info(f"Moving to Stand Pose: {Poses.START.to_list()}")
        success = self.move_to_pose_immediate(Poses.START)
        if success:
            self.state = RobotState.IDLE
            logger.info("Standing complete")
        else:
            self.state = RobotState.ERROR
        return success
    
    def move_forward(self, speed: GaitSpeed = GaitSpeed.NORMAL) -> bool:
        """전진 보행 실행"""
        self.state = RobotState.WALKING
        gait = GaitPatterns.forward_gait()
        success = self._execute_gait_pattern(gait, speed, "Forward Gait")
        self.state = RobotState.IDLE if success else RobotState.ERROR
        return success
    
    def turn_left(self) -> bool:
        """좌회전 실행"""
        self.state = RobotState.TURNING
        gait = GaitPatterns.turn_left_gait()
        # 회전은 항상 고정 속도 (0.15s)
        # 패턴 내부에서 duration을 사용하므로 NORMAL 속도 기준 수정 필요
        # 간단히 하기 위해 여기서는 짧은 duration 사용
        success = True
        for step in gait:
            duration = 0.15  # 고정 회전 속도
            if not self._move_legs_interpolated(step.movements, duration):
                success = False
                break
        
        self.state = RobotState.IDLE if success else RobotState.ERROR
        if success:
            logger.info("Turn left completed")
        return success
    
    def turn_right(self) -> bool:
        """우회전 실행"""
        self.state = RobotState.TURNING
        gait = GaitPatterns.turn_right_gait()
        success = True
        for step in gait:
            duration = 0.15
            if not self._move_legs_interpolated(step.movements, duration):
                success = False
                break
        
        self.state = RobotState.IDLE if success else RobotState.ERROR
        if success:
            logger.info("Turn right completed")
        return success
    
    def sidestep_left(self) -> bool:
        """왼쪽으로 걷기 (다리 4,3을 앞다리처럼 사용)"""
        self.state = RobotState.WALKING
        logger.info("Walking left - legs 4,3 as leading legs")
        gait = GaitPatterns.sidestep_left_gait()
        success = self._execute_gait_pattern(gait, GaitSpeed.NORMAL, "Walk Left")
        self.state = RobotState.IDLE if success else RobotState.ERROR
        return success
    
    def sidestep_right(self) -> bool:
        """오른쪽으로 걷기 (다리 1,2를 앞다리처럼 사용)"""
        self.state = RobotState.WALKING
        logger.info("Walking right - legs 1,2 as leading legs")
        gait = GaitPatterns.sidestep_right_gait()
        success = self._execute_gait_pattern(gait, GaitSpeed.NORMAL, "Walk Right")
        self.state = RobotState.IDLE if success else RobotState.ERROR
        return success


# --- MAIN PROGRAM ---

def main():
    """메인 프로그램"""
    # Controller 초기화
    controller = DynamixelController(DEVICENAME, BAUDRATE)
    
    if not controller.connect():
        logger.error("Failed to connect to Dynamixel. Exiting.")
        sys.exit(1)
    
    try:
        # Robot 초기화
        robot = QuadrupedRobot(controller, LEG_IDS)
        
        if not robot.enable_all_torque():
            logger.error("Failed to enable torque. Exiting.")
            return
        
        if not robot.initialize_pose():
            logger.error("Failed to initialize pose. Exiting.")
            return
        
        # 조작법 출력
        print_controls()
        
        # 메인 루프
        logger.info("Entering main control loop. Press keys to control robot.")
        
        while True:
            key = get_keypress()
            
            if key is None:
                time.sleep(0.01)
                continue
            
            # 명령 처리
            if key == 'i':
                robot.initialize_pose()
                
            elif key == 's':
                robot.stand_pose()
                
            elif key == 'w':
                robot.move_forward(GaitSpeed.NORMAL)
                
            elif key == 'W':
                robot.move_forward(GaitSpeed.FAST)
                
            elif key == 'a':
                robot.turn_left()
                
            elif key == 'd':
                robot.turn_right()
            
            elif key == 'z':
                robot.sidestep_left()
            
            elif key == 'c':
                robot.sidestep_right()
                
            elif key == 't':
                robot.disable_all_torque()
                
            elif key == 'q':
                logger.info("Quit command received")
                robot.disable_all_torque()
                break
            
            else:
                logger.debug(f"Unknown key: {key}")
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        robot.disable_all_torque()
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        robot.disable_all_torque()
    
    finally:
        controller.disconnect()
        logger.info("Program terminated")


if __name__ == "__main__":
    main()
