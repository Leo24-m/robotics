#!/usr/bin/env python3
import time
import sys
import serial  # still unused but kept if you plan to use it later

from dynamixel_sdk import *  # Dynamixel SDK

# --- PLATFORM-SPECIFIC IMPORTS -----------------------------------
try:
    import msvcrt  # Windows
    _WINDOWS = True
except ImportError:
    import termios
    import tty
    _WINDOWS = False

# --- 상수 정의 ---

# 통신 설정
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'  # e.g. 'COM5' on Windows, '/dev/ttyUSB0' on Linux
PROTOCOL_VERSION = 2.0

# 모터 주소 (Protocol 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
LEN_GOAL_POSITION = 4  # 4 bytes
LEN_PRESENT_POSITION = 4  # 4 bytes

# 모터 ID 정의
LEG_IDS = {
    1: [0, 1, 2],   # Front-Right
    2: [3, 4, 5],   # Front-Left
    3: [6, 7, 8],   # Back-Left
    4: [9, 10, 11]  # Back-Right
}
ALL_MOTOR_IDS = [id for ids in LEG_IDS.values() for id in ids]

# 보간 설정

STEP_DURATION = 0.3   # (Normal) Power Stroke 시간
RETURN_STEP_DURATION = STEP_DURATION * 1.5  # (Normal) Return Stroke 시간
INTERPOLATION_STEPS = 25  # 보간 단계 수

# 고속 보행 설정
FAST_STEP_DURATION = STEP_DURATION * 0.8        # (Fast) Power Stroke 시간
FAST_RETURN_STEP_DURATION = RETURN_STEP_DURATION * 0.8  # (Fast) Return Stroke 시간


class Poses:
    """
    모든 정적 자세 및 보행 시퀀스 위치를 정의합니다.
    """
    # --- 정적 자세 ---
    INITIAL = [2048, 2048, 2048]  # 'i' (Initialize)용
    START = [2048, 2141, 2228]    # 's' (Start/Stand)용

    # --- 전진 보행에 사용되는 위치 ---
    FORWARD_POS_0 = [2048, 2141, 2228]  # 시작/복귀 위치
    FORWARD_POS_1 = [2048, 1679, 1896]  # move_forward에서 다리2가 사용
    FORWARD_POS_3 = [2861, 2141, 2228]  # move_forward에서 다리3이 사용
    FORWARD_POS_5 = [1248, 2141, 2228]  # move_forward에서 다리1이 사용
    FORWARD_POS_10 = [2048, 2382, 1467] # move_forward에서 다리4가 사용

    FORWARD_POS_11 = [2418, 2432, 1409] # move_forward에서 다리2가 사용
    FORWARD_POS_12 = [1754, 2057, 2413] # move_forward에서 다리4가 사용
    INTERM_POS = [2086,1681,2202]
    BACKWARD_POS_11 = [1754, 2057, 2413] # move_forward에서 다리2가 사용
    BACKWARD_POS_12 = [2418, 2432, 1409] # move_forward에서 다리4가 사용

    FORWARD_POS_13 = [1677, 2432, 1409] # move_forward에서 다리1가 사용
    FORWARD_POS_14 = [2341, 2057, 2413] # move_forward에서 다리3가 사용
    INTERM_POS = [2009,1681,2202]
    BACKWARD_POS_13 = [2341, 2057, 2413] # move_forward에서 다리1가 사용
    BACKWARD_POS_14 = [1677, 2432, 1409] # move_forward에서 다리3가 사용

    # --- 좌회전에 사용되는 위치 ---
    TURN_LEFT_POS_1 = [2200, 2141, 2228]  # = pos3
    LEFT_INTERM_POS = [2100,1804,2094]
    TURN_LEFT_POS_2 = [2861, 1679, 1896]  # = pos2
    TURN_LEFT_POS_3 = [2048, 1679, 1896]  # = pos1
    TURN_LEFT_POS_4 = [2048, 2141, 2228]  # = pos0, pos4 (복귀)

    # --- 우회전에 사용되는 위치 ---
    TURN_RIGHT_POS_3 = [2048, 1679, 1896]  # = pos1
    RIGHT_INTERM_POS = [1995,1804,2094]
    TURN_RIGHT_POS_4 = [2048, 2141, 2228]  # = pos0, pos4 (복귀)
    TURN_RIGHT_POS_5 = [1895, 2141, 2228]  # = pos5
    TURN_RIGHT_POS_7 = [1248, 1679, 1896]  # = pos12


# --- 유틸리티 함수 ---

def get_keypress():
    """
    키 입력을 감지 (Windows / POSIX 모두 지원)

    Returns:
        str or None: 입력된 1글자. 특수키(방향키 등)는 None.
    """
    if _WINDOWS:
        ch = msvcrt.getch()
        # 방향키, 기능키 등은 2바이트 시퀀스 (첫 바이트가 0x00 또는 0xE0)
        if ch in (b'\x00', b'\xe0'):
            _ = msvcrt.getch()  # 두 번째 바이트 소비
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
    print("------------------------------------------------")
    print("Dynamixel Gait Control (Refactored)")
    print(f"Press 'i' -> Initialize Pose {Poses.INITIAL}")
    print(f"Press 's' -> Stand/Start Pose {Poses.START}")
    print("Press 'w' -> Move Forward (Normal Speed)")
    print("Press 'W' (Shift+w) -> Move Forward (FAST Speed)")
    print("Press 'a' -> Turn Left")
    print("Press 'd' -> Turn Right")
    print("Press 't' -> Disable torque for ALL motors")
    print("Press 'q' -> Quit and close port")
    print("------------------------------------------------")


# --- Dynamixel 제어 클래스 ---

class DynamixelController:
    """
    Dynamixel SDK와의 저수준 통신을 관리합니다.
    """
    def __init__(self, device_name, baudrate):
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.group_sync_write = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )
        self.baudrate = baudrate

    def connect(self):
        """포트를 열고 보드레이트를 설정합니다."""
        if not self.port_handler.openPort():
            print("Failed to open the port.")
            return False
        if not self.port_handler.setBaudRate(self.baudrate):
            print("Failed to set the baudrate.")
            return False
        print("Dynamixel port connected.")
        return True

    def disconnect(self):
        """포트를 닫습니다."""
        self.port_handler.closePort()
        print("Dynamixel port disconnected.")

    def _write_1byte_txrx(self, motor_id, address, value):
        """1바이트 쓰기 명령을 전송합니다."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, address, value
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"{self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"{self.packet_handler.getRxPacketError(dxl_error)}")

    def enable_torque(self, motor_ids):
        """선택된 모터들의 토크를 활성화합니다."""
        for motor_id in motor_ids:
            self._write_1byte_txrx(motor_id, ADDR_TORQUE_ENABLE, 1)

    def disable_torque(self, motor_ids):
        """선택된 모터들의 토크를 비활성화합니다."""
        for motor_id in motor_ids:
            self._write_1byte_txrx(motor_id, ADDR_TORQUE_ENABLE, 0)

    def sync_write_goal_position(self, motor_positions: dict):
        """
        여러 모터에 목표 위치를 동시에 씁니다.
        :param motor_positions: {motor_id: position} 형태의 딕셔너리
        """
        self.group_sync_write.clearParam()

        for motor_id, position in motor_positions.items():
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position)),
                DXL_HIBYTE(DXL_LOWORD(position)),
                DXL_LOBYTE(DXL_HIWORD(position)),
                DXL_HIBYTE(DXL_HIWORD(position)),
            ]
            add_param_result = self.group_sync_write.addParam(
                motor_id, param_goal_position
            )
            if not add_param_result:
                print(f"[ID:{motor_id}] groupSyncWrite addParam failed")
                return False

        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(
                "sync_write_goal_position txPacket failed: "
                f"{self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
            return False
        return True


# --- 로봇 클래스 ---

class QuadrupedRobot:
    """
    4족 로봇의 상태와 고수준 동작을 관리합니다.
    """
    def __init__(self, controller: DynamixelController, leg_ids: dict):
        self.controller = controller
        self.leg_ids = leg_ids
        self.all_motor_ids = [id for ids in leg_ids.values() for id in ids]
        # 현재 모든 다리의 위치를 INITIAL_POS로 초기화
        self.current_positions = {
            leg_num: list(Poses.INITIAL) for leg_num in self.leg_ids.keys()
        }

    def enable_all_torque(self):
        """모든 모터의 토크를 켭니다."""
        print("Enabling torque for all motors...")
        self.controller.enable_torque(self.all_motor_ids)

    def disable_all_torque(self):
        """모든 모터의 토크를 끕니다."""
        print("Disabling torque for all motors...")
        self.controller.disable_torque(self.all_motor_ids)

    def _get_durations(self, fast=False):
        """보행 속도에 맞는 duration을 반환합니다."""
        if fast:
            return FAST_STEP_DURATION, FAST_RETURN_STEP_DURATION
        else:
            return STEP_DURATION, RETURN_STEP_DURATION
    

    def move_to_pose_immediate(self, target_pos: list):
        """[보간 없음] 모든 다리를 동시에 target_pos로 이동시킵니다."""
        motor_positions = {}
        for leg_num, motor_ids in self.leg_ids.items():
            for i in range(3):  # 3개 모터
                motor_positions[motor_ids[i]] = target_pos[i]

        if self.controller.sync_write_goal_position(motor_positions):
            # 상태 업데이트
            for leg_num in self.leg_ids.keys():
                self.current_positions[leg_num] = list(target_pos)

    def _move_legs_interpolated(self, movements: dict, duration: float):
        """
        [보간 적용] 지정된 다리들을 목표 위치로 부드럽게 이동시킵니다.
        :param movements: {leg_num: [target_pos]} 형태의 딕셔너리
        :param duration: 이동에 걸리는 총 시간
        """
        start_positions = {
            leg_num: list(self.current_positions[leg_num])
            for leg_num in movements.keys()
        }
        steps = INTERPOLATION_STEPS
        step_time = duration / steps

        for n in range(1, steps + 1):  # 1부터 steps까지
            start_time = time.time()
            motor_positions_to_write = {}

            for leg_num, target_pos in movements.items():
                leg_motor_ids = self.leg_ids[leg_num]
                start_pos = start_positions[leg_num]

                for i in range(3):  # 3개 모터
                    start = start_pos[i]
                    end = target_pos[i]
                    current_step_pos = int(start + (end - start) * (n / steps))
                    motor_id = leg_motor_ids[i]
                    motor_positions_to_write[motor_id] = current_step_pos

            if not self.controller.sync_write_goal_position(motor_positions_to_write):
                return False  # 전송 실패 시 중단
            elapsed_time = time.time() - start_time
            if elapsed_time < step_time:
                time.sleep(step_time - elapsed_time)
            else:
                print("Warning: Step processing took longer than step_time.")

        # 최종 상태 업데이트
        for leg_num, target_pos in movements.items():
            self.current_positions[leg_num] = list(target_pos)

        return True

    def initialize_pose(self):
        """초기 자세(i)로 이동합니다."""
        print(f"Moving all legs to Initial Pose {Poses.INITIAL}...")
        self.move_to_pose_immediate(Poses.INITIAL)
        print("Done.")

    def stand_pose(self):
        """시작/서기 자세(s)로 이동합니다."""
        print(f"Moving all legs to Start Pose {Poses.START}...")
        self.move_to_pose_immediate(Poses.START)
        print("Done.")

    def move_forward(self, fast=True):
        """
        전진 보행 (순서: 3 -> 1 -> 4 -> 2)
        """
        power_duration, return_duration = self._get_durations(fast)
        speed_str = "FAST" if fast else "Normal"
        print(f"--- Moving Forward ({speed_str}) ---")

        # 다리별로 순차적으로 이동
        #if not self._move_legs_interpolated({3: Poses.FORWARD_POS_3}, power_duration):
        #    return
        #if not self._move_legs_interpolated({1: Poses.FORWARD_POS_5}, power_duration):
        #    return

        if not self._move_legs_interpolated(
            {
                2: Poses.INTERM_POS, 
                4: Poses.INTERM_POS,
            }, 
            return_duration
        ):
            return
        
        if not self._move_legs_interpolated(
            {
                1: Poses.FORWARD_POS_13,
                2: Poses.FORWARD_POS_11, 
                3: Poses.FORWARD_POS_14,
                4: Poses.FORWARD_POS_12,
            }, 
            power_duration
        ):
            return
        
        if not self._move_legs_interpolated(
            {
                1: Poses.INTERM_POS, 
                3: Poses.INTERM_POS,
            }, 
            return_duration
        ):
            return

        if not self._move_legs_interpolated(
            {
                1: Poses.BACKWARD_POS_13,
                2: Poses.BACKWARD_POS_11, 
                3: Poses.BACKWARD_POS_14,
                4: Poses.BACKWARD_POS_12,
            }, 
            power_duration
        ):
            return
        

        # 모든 다리 원위치
        
        print("--- Forward Step Finished ---")



    def turn_left(self):
        """
        좌회전 (순서: 3->2->1->4)
        """
        print("--- Turning Left ---")
        power_duration = 0.2

        # 각 다리를 순차적으로 이동
        if not self._move_legs_interpolated(
            {
                1: Poses.TURN_LEFT_POS_4,
                2: Poses.LEFT_INTERM_POS,
                3: Poses.TURN_LEFT_POS_4,
                4: Poses.LEFT_INTERM_POS,                
            }, power_duration
        ):
            return
        
        if not self._move_legs_interpolated(
            {
                2: Poses.TURN_LEFT_POS_1,
                4: Poses.TURN_LEFT_POS_1
            }, power_duration
        ):
            return
        
        if not self._move_legs_interpolated(
            {
                1: Poses.LEFT_INTERM_POS,
                2: Poses.TURN_LEFT_POS_4,
                3: Poses.LEFT_INTERM_POS,                
                4: Poses.TURN_LEFT_POS_4
            }, power_duration
        ):
            return

        if not self._move_legs_interpolated(
            {
                1: Poses.TURN_LEFT_POS_1,
                3: Poses.TURN_LEFT_POS_1
            }, power_duration
        ):
            return

        print("--- Turn Left Finished ---")

    def turn_right(self):
        """
        우회전 (순서: 3->2->1->4)
        """
        print("--- Turning Right ---")
        power_duration = 0.2

        # 각 다리를 순차적으로 이동
        if not self._move_legs_interpolated(
            {
                1: Poses.RIGHT_INTERM_POS,
                2: Poses.TURN_RIGHT_POS_4,
                3: Poses.RIGHT_INTERM_POS,                
                4: Poses.TURN_RIGHT_POS_4
            }, power_duration
        ):
            return

        if not self._move_legs_interpolated(
            {
                1: Poses.TURN_RIGHT_POS_5,
                3: Poses.TURN_RIGHT_POS_5
            }, power_duration
        ):
            return
        
        if not self._move_legs_interpolated(
            {
                1: Poses.TURN_RIGHT_POS_4,
                2: Poses.RIGHT_INTERM_POS,
                3: Poses.TURN_RIGHT_POS_4,
                4: Poses.RIGHT_INTERM_POS,                
            }, power_duration
        ):
            return
        
        if not self._move_legs_interpolated(
            {
                2: Poses.TURN_RIGHT_POS_5,
                4: Poses.TURN_RIGHT_POS_5
            }, power_duration
        ):
            return


        print("--- Turn Right Finished ---")
# --- 메인 실행 ---

def main():
    # 1. 컨트롤러 초기화 및 연결
    controller = DynamixelController(DEVICENAME, BAUDRATE)
    if not controller.connect():
        sys.exit(1)

    # 2. 로봇 객체 생성
    robot = QuadrupedRobot(controller, LEG_IDS)

    # 3. 로봇 초기 설정
    robot.enable_all_torque()
    robot.initialize_pose()

    # 4. 조작법 안내
    print_controls()

    # 5. 메인 루프
    while True:
        key = get_keypress()
        if key is None:
            # 방향키 같은 특수키는 무시
            time.sleep(0.01)
            continue

        if key == 'i':
            robot.initialize_pose()

        elif key == 's':
            robot.stand_pose()

        elif key == 'w':
            robot.move_forward(fast=False)

        elif key == 'W':  # 대문자 W
            robot.move_forward(fast=True)

        elif key == 'a':
            robot.turn_left()

        elif key == 'd':
            robot.turn_right()

        elif key == 't':
            robot.disable_all_torque()

        elif key == 'q':
            print("Exiting...")
            robot.disable_all_torque()
            break

        time.sleep(0.01)

    # 6. 연결 종료
    controller.disconnect()


if __name__ == "__main__":
    main()
