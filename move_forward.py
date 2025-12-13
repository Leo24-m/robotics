import time
import sys
import tty
import termios
from dynamixel_sdk import * # Dynamixel SDK

# --- 통신 설정 ---
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'

# --- 모터 주소 및 데이터 길이 (Protocol 2.0) ---
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
LEN_GOAL_POSITION = 4 # 4 bytes
LEN_PRESENT_POSITION = 4 # 4 bytes

# --- 모터 ID 정의 ---
# 전체 모터 ID
DXL_ID_P2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 
# 다리별 모터 ID (가정: 0,1,2=Leg1 / 3,4,5=Leg2 / 6,7,8=Leg3 / 9,10,11=Leg4)
LEG1_IDS = [0, 1, 2]
LEG2_IDS = [3, 4, 5]
LEG3_IDS = [6, 7, 8]
LEG4_IDS = [9, 10, 11]

# --- 정적 자세 위치 (0-4095) ---
INITIAL_POS = [2048, 2048, 2048] # 'i' (Initialize)용
START_POS = [2048, 2141, 2228]   # 's' (Start/Stand)용

# --- 보행 시퀀스 (0-4095 원시 위치 값) ---
# 2번, 3번 다리용
GAIT_SEQ_1 = [
    [2048, 1679, 1896],
    [2561, 1679, 1896],
    [2561, 2141, 2228],
    [2048, 2141, 2228]
]
# 1번, 4번 다리용
GAIT_SEQ_2 = [
    [2048, 1679, 1896],
    [1512, 1679, 1896],
    [1512, 2141, 2228],
    [2048, 2141, 2228]
]

# --- 보간 설정 ---
STEP_DURATION = 1.0  # 한 보행 스텝에 걸리는 총 시간 (초)
INTERPOLATION_STEPS = 50 # 보간 단계 수

# --- Dynamixel SDK 설정 ---
port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(2.0)

# --- GroupSyncWrite 인스턴스 초기화 (목표 위치용) ---
group_sync_write = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

# --- 포트 열기 ---
if not port_handler.openPort():
    print("Failed to open the port.")
    sys.exit()
if not port_handler.setBaudRate(BAUDRATE):
    print("Failed to set the baudrate.")
    sys.exit()

# --- [NEW] 현재 다리 위치 상태 저장 ---
# 프로그램이 각 다리의 현재 위치를 추적하기 위함 (보간의 시작점으로 사용)
current_leg_positions = {
    1: list(INITIAL_POS), # Leg 1
    2: list(INITIAL_POS), # Leg 2
    3: list(INITIAL_POS), # Leg 3
    4: list(INITIAL_POS)  # Leg 4
}

# --- 기본 함수 ---

def unlock_torque(motor_ids):
    """선택된 모터들의 토크를 비활성화"""
    for motor_id in motor_ids:
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        print(f"Torque disabled for motor {motor_id}")

def enable_torque(motor_ids):
    """선택된 모터들의 토크를 활성화"""
    for motor_id in motor_ids:
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        print(f"Torque enabled for motor {motor_id}")

def get_keypress():
    """키 입력을 감지"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# --- [NEW] 보행 및 자세 함수 ---

def move_all_legs_sync(target_pos):
    """
    [보간 없음] 모든 다리를 동시에 target_pos로 이동시킵니다.
    'i' (초기화) 및 's' (시작) 자세에 사용됩니다.
    """
    global current_leg_positions
    group_sync_write.clearParam()
    
    all_leg_ids = [LEG1_IDS, LEG2_IDS, LEG3_IDS, LEG4_IDS]
    
    for leg_motors in all_leg_ids:
        for i in range(3): # 다리 모터 3개
            motor_id = leg_motors[i]
            position = target_pos[i]
            
            # 4바이트 위치 데이터 준비
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(position)), 
                                 DXL_HIBYTE(DXL_LOWORD(position)), 
                                 DXL_LOBYTE(DXL_HIWORD(position)), 
                                 DXL_HIBYTE(DXL_HIWORD(position))]
            
            add_param_result = group_sync_write.addParam(motor_id, param_goal_position)
            if not add_param_result:
                print(f"[ID:{motor_id}] groupSyncWrite addParam failed")

    # 동기화 패킷 전송
    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"move_all_legs_sync txPacket failed: {packet_handler.getTxRxResult(dxl_comm_result)}")
    else:
        # [중요] 모든 다리의 현재 위치 상태 업데이트
        for i in range(1, 5):
            current_leg_positions[i] = list(target_pos)

def move_leg_interpolated(leg_motor_ids, leg_id_num, target_pos):
    """
    [보간 적용] 한쪽 다리를 현재 위치에서 target_pos까지 부드럽게 이동시킵니다.
    'w' (보행)에 사용됩니다.
    """
    global current_leg_positions
    
    start_pos = current_leg_positions[leg_id_num] # 현재 상태에서 시작
    duration = STEP_DURATION
    steps = INTERPOLATION_STEPS
    step_time = duration / steps

    for n in range(1, steps + 1): # 1부터 steps까지
        group_sync_write.clearParam()
        current_step_pos_list = []

        for i in range(3): # 3개 모터에 대해
            start = start_pos[i]
            end = target_pos[i]
            
            # 현재 스텝의 위치 계산
            current_step_pos = int(start + (end - start) * (n / steps))
            current_step_pos_list.append(current_step_pos) # (상태 업데이트용)
            
            motor_id = leg_motor_ids[i]
            
            # 4바이트 위치 데이터 준비
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(current_step_pos)), 
                                 DXL_HIBYTE(DXL_LOWORD(current_step_pos)), 
                                 DXL_LOBYTE(DXL_HIWORD(current_step_pos)), 
                                 DXL_HIBYTE(DXL_HIWORD(current_step_pos))]
            
            add_param_result = group_sync_write.addParam(motor_id, param_goal_position)
            if not add_param_result:
                print(f"[ID:{motor_id}] groupSyncWrite addParam failed")
                return False

        # 이 스텝의 동기화 패킷 전송
        dxl_comm_result = group_sync_write.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(f"move_leg_interpolated txPacket failed: {packet_handler.getTxRxResult(dxl_comm_result)}")
            return False
        
        time.sleep(step_time)

    # [중요] 보간 완료 후, 다리의 최종 위치를 상태에 업데이트
    current_leg_positions[leg_id_num] = list(target_pos)
    return True

def move_forward():
    """
    정의된 시퀀스와 순서(3 -> 2 -> 1 -> 4)대로 
    다리 움직임을 한 사이클 실행합니다. (보간 적용)
    """
    print("--- Moving Forward (Cycle: 3 -> 2 -> 1 -> 4) ---")
    
    # 1. 3번 다리 (GAIT_SEQ_1)
    print("Moving Leg 3...")
    for step in GAIT_SEQ_1:
        if not move_leg_interpolated(LEG3_IDS, 3, step):
            print("Failed to move Leg 3. Aborting cycle.")
            return
    print("Leg 3 sequence complete.")

    # 2. 2번 다리 (GAIT_SEQ_1)
    print("Moving Leg 2...")
    for step in GAIT_SEQ_1:
        if not move_leg_interpolated(LEG2_IDS, 2, step):
            print("Failed to move Leg 2. Aborting cycle.")
            return
    print("Leg 2 sequence complete.")

    # 3. 1번 다리 (GAIT_SEQ_2) - 순서 변경
    print("Moving Leg 1...")
    for step in GAIT_SEQ_2:
        if not move_leg_interpolated(LEG1_IDS, 1, step):
            print("Failed to move Leg 1. Aborting cycle.")
            return
    print("Leg 1 sequence complete.")

    # 4. 4번 다리 (GAIT_SEQ_2) - 순서 변경
    print("Moving Leg 4...")
    for step in GAIT_SEQ_2:
        if not move_leg_interpolated(LEG4_IDS, 4, step):
            print("Failed to move Leg 4. Aborting cycle.")
            return
    print("Leg 4 sequence complete.")
    
    print("--- Forward Step Finished ---")


# --- 메인 루프 ---
def main():
    global current_leg_positions
    
    # 시작 시 모든 모터 토크 활성화
    enable_torque(DXL_ID_P2)
    
    # [NEW] 프로그램을 시작할 때 로봇을 INITIAL_POS로 이동
    # 이래야 current_leg_positions 상태와 실제 로봇 상태가 일치함
    print(f"Initializing robot to {INITIAL_POS}...")
    move_all_legs_sync(INITIAL_POS)
    print("Robot Initialized. Ready for commands.")

    print("------------------------------------------------")
    print("Dynamixel Gait Control")
    print(f"Press 'i' -> Initialize Pose {INITIAL_POS}")
    print(f"Press 's' -> Stand/Start Pose {START_POS}")
    print("Press 'w' -> Move Forward (cycle: 3 -> 2 -> 1 -> 4)")
    print("Press 't' -> Disable torque for ALL motors")
    print("Press 'q' -> Quit and close port")
    print("------------------------------------------------")

    while True:
        key = get_keypress()

        if key == 'i':
            # 초기 위치로 이동 (보간 없음)
            print(f"Moving all legs to Initial Pose {INITIAL_POS}...")
            move_all_legs_sync(INITIAL_POS)
            print("Done.")
        
        elif key == 's':
            # 시작 위치(일어서기)로 이동 (보간 없음)
            print(f"Moving all legs to Start Pose {START_POS}...")
            move_all_legs_sync(START_POS)
            print("Done.")

        elif key == 'w':
            # 보행 사이클 1회 실행 (보간 있음)
            move_forward()

        elif key == 't':
            # 모든 모터 토크 비활성화
            unlock_torque(DXL_ID_P2)
        
        elif key == 'q':
            print("Disabling torque on all motors and exiting...")
            unlock_torque(DXL_ID_P2) # 종료 전 토크 풀기
            break
        
        time.sleep(0.01) # 키 입력 대기

    # 포트 닫기
    port_handler.closePort()

if __name__ == "__main__":
    main()

