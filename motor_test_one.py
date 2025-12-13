import os
import time

# 운영체제에 따라 getch (키보드 입력 감지)를 위한 라이브러리를 임포트합니다.
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# Dynamixel SDK의 필요한 클래스들을 임포트합니다.
from dynamixel_sdk import *

# ############################# 설정 값 #############################
# 포트 및 통신 속도 설정
BAUDRATE                = 57600
DEVICENAME              = '/dev/ttyUSB0'

# 모터 ID 설정
# DXL_ID_P1               = [0, 6]  # 프로토콜 1.0을 사용하는 모터 ID 리스트 (MX-28)
DXL_ID_P1               = [0]  # 프로토콜 1.0을 사용하는 모터 ID 리스트 (MX-28)
DXL_ID_P2               = [] # 프로토콜 2.0을 사용하는 모터 ID 리스트 (MX-64)
# DXL_ID_P2               = [1, 2, 3, 4, 5, 7, 8]
ALL_DXL_IDS             = DXL_ID_P1 + DXL_ID_P2

# 프로토콜 1.0 제어 테이블 주소 (MX-28)
ADDR_P1_TORQUE_ENABLE      = 24
ADDR_P1_GOAL_POSITION      = 30
ADDR_P1_PRESENT_POSITION   = 36

# 프로토콜 2.0 제어 테이블 주소 (MX-64 v2.0)
ADDR_P2_TORQUE_ENABLE      = 64
ADDR_P2_GOAL_POSITION      = 116
ADDR_P2_PRESENT_POSITION   = 132

# 기타 설정
TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0
DXL_MOVING_STATUS_THRESHOLD = 20

# ###################################################################

# --- 헬퍼 함수 정의 ---
def angle_to_position(angle):
    """0-360도 각도를 0-4095 위치 값으로 변환합니다."""
    return int((angle / 360.0) * 4095.0)

def set_torque(dxl_id, status):
    """모터 ID를 확인하여 올바른 프로토콜로 토크를 설정합니다."""
    if dxl_id in DXL_ID_P1:
        # 프로토콜 1.0으로 토크 설정
        dxl_comm_result, dxl_error = packetHandler1.write1ByteTxRx(portHandler, dxl_id, ADDR_P1_TORQUE_ENABLE, status)
    elif dxl_id in DXL_ID_P2:
        # 프로토콜 2.0으로 토크 설정
        dxl_comm_result, dxl_error = packetHandler2.write1ByteTxRx(portHandler, dxl_id, ADDR_P2_TORQUE_ENABLE, status)
    else:
        return # ID가 목록에 없으면 아무것도 안 함

    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id}] Torque Set Fail: {packetHandler1.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id}] Torque Set Fail: {packetHandler1.getRxPacketError(dxl_error)}")
    else:
        state = "ON" if status == TORQUE_ENABLE else "OFF"
        print(f"Dynamixel ID:{dxl_id} Torque is {state}.")

def move_to_angle(dxl_id, angle):
    """모터 ID를 확인하여 올바른 프로토콜로 목표 각도로 이동시킵니다."""
    position = angle_to_position(angle)
    print(f"  -> Moving ID:{dxl_id} to Angle:{angle} (Position:{position})")
    if dxl_id in DXL_ID_P1:
        # 프로토콜 1.0으로 위치 쓰기 (2바이트)
        dxl_comm_result, dxl_error = packetHandler1.write2ByteTxRx(portHandler, dxl_id, ADDR_P1_GOAL_POSITION, position)
    elif dxl_id in DXL_ID_P2:
        # 프로토콜 2.0으로 위치 쓰기 (4바이트)
        dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, dxl_id, ADDR_P2_GOAL_POSITION, position)
    else:
        return

    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id}] Move Fail: {packetHandler1.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id}] Move Fail: {packetHandler1.getRxPacketError(dxl_error)}")

# --- 메인 코드 ---
# 포트 핸들러와 **두 개의** 패킷 핸들러 인스턴스 생성
portHandler = PortHandler(DEVICENAME)
packetHandler1 = PacketHandler(1.0) # 프로토콜 1.0용
packetHandler2 = PacketHandler(2.0) # 프로토콜 2.0용

# 1. 포트 열기
if not portHandler.openPort():
    print("Failed to open the port. Press any key to terminate...")
    getch()
    quit()
print("Succeeded to open the port")

# 2. 포트 통신 속도(Baudrate) 설정
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate. Press any key to terminate...")
    getch()
    quit()
print("Succeeded to change the baudrate")

# 3. 모든 모터 토크 켜기
for dxl_id in ALL_DXL_IDS:
    set_torque(dxl_id, TORQUE_ENABLE)

# 메인 루프
while 1:
    try:
        # 사용자로부터 목표 각도 입력받기
        target_angle_str = input("\nEnter a target angle (0-360), or type 'q' to quit: ")

        if target_angle_str.lower() == 'q':
            break

        target_angle = float(target_angle_str)
        if not (0 <= target_angle <= 360):
            print("  -> Angle out of range. Please enter a value between 0 and 360.")
            continue

        # 모든 모터를 목표 각도로 이동
        for dxl_id in ALL_DXL_IDS:
            move_to_angle(dxl_id, target_angle)
        
        # 이동 완료까지 잠시 대기 (간단한 구현)
        print("  -> Waiting for motors to arrive...")
        time.sleep(2) # 이동 시간에 맞춰 조절 필요

    except ValueError:
        print("  -> Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        break


# 6. 모든 모터 토크 끄기
print("\nDisabling torque on all motors...")
for dxl_id in ALL_DXL_IDS:
    set_torque(dxl_id, TORQUE_DISABLE)

# 7. 포트 닫기
portHandler.closePort()
print("Succeeded to close the port.")

