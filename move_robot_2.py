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

# --- [UPDATE] 모터 ID 정의 ---
# 전체 모터 ID
DXL_ID_P2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 
# 다리별 모터 ID (가정: 0,1,2=Leg1 / 3,4,5=Leg2 / 6,7,8=Leg3 / 9,10,11=Leg4)
LEG1_IDS = [0, 1, 2] # Front-Left
LEG2_IDS = [3, 4, 5] # Front-Right
LEG3_IDS = [6, 7, 8] # Back-Right
LEG4_IDS = [9, 10, 11] # Back-Left

# --- [UPDATE] 정적 자세 위치 (0-4095) ---
INITIAL_POS = [2048, 2048, 2048] # 'i' (Initialize)용
# [UPDATE] 's' (Start/Stand)용 (새로운 Pos 0). 로직을 위해 Pos 0 값이어야 합니다.
START_POS = [2048, 2462, 1926]

# --- [UPDATE] 새로운 9단계 보행 시퀀스 ---
# 2번, 3번 다리용 (Right Side)
GAIT_SEQ_1 = [
    [2048, 1430, 1792], # Pos 0
    [2048, 1430, 1792], # Pos 1
    [2561, 1430, 1792], # Pos 2
    [2561, 2462, 1926], # Pos 3
    [2048, 2462, 1926], # Pos 4
    [1512, 2462, 1926], # Pos 5
    [1512, 1430, 1792], # Pos 6
    [2048, 1430, 1792], # Pos 7
    [2048, 2462, 1926]  # Pos 8
]
# 1번, 4번 다리용 (Left Side)
GAIT_SEQ_2 = [
    [2048, 1430, 1792], # Pos 0
    [2048, 1430, 1792], # Pos 1
    [1512, 1430, 1792], # Pos 2
    [1512, 2462, 1926], # Pos 3
    [2048, 2462, 1926], # Pos 4 
    [2561, 2462, 1926], # Pos 5
    [2561, 1430, 1792], # Pos 6
    [2048, 1430, 1792], # Pos 7
    [2048, 2462, 1926]  # Pos 8 
]

# --- [NEW] 전진/후진 시퀀스 분리 ---
# FWD: Pos 0~8
GAIT_SEQ_1_FWD = GAIT_SEQ_1[0:8] 
GAIT_SEQ_2_FWD = GAIT_SEQ_2[0:8] 


# --- [UPDATE] 보간 설정 ---
STEP_DURATION = 0.5  # (Normal) Power Stroke 시간 (Pos 0->1, 1->2, 2->3 용)
RETURN_STEP_DURATION = STEP_DURATION * 5 # (Normal) Return Stroke 시간 (Pos 3->4 용)
INTERPOLATION_STEPS = 50 # 보간 단계 수
POSE_CHANGE_DURATION = 1.5 # [NEW] 'i', 's' 키를 위한 느린 자세 변경 시간

# --- [UPDATE] 고속 보행 설정 (현실적인 값) ---
FAST_STEP_DURATION = STEP_DURATION * 0.6 # (Fast) Power Stroke 시간
FAST_RETURN_STEP_DURATION = RETURN_STEP_DURATION * 0.6 # (Fast) Return Stroke 시간

# --- Dynamixel SDK 설정 ---
port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(2.0)
group_sync_write = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

# --- 포트 열기 ---
if not port_handler.openPort():
    print("Failed to open the port.")
    sys.exit()
if not port_handler.setBaudRate(BAUDRATE):
    print("Failed to set the baudrate.")
    sys.exit()

# --- 현재 다리 위치 상태 저장 ---
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

def enable_torque(motor_ids):
    """선택된 모터들의 토크를 활성화"""
    for motor_id in motor_ids:
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)

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

# --- 보행 및 자세 함수 ---

# [REMOVED] move_all_legs_sync 함수 제거됨 (move_legs_interpolated_sync로 대체)

def move_legs_interpolated_sync(movements, duration=STEP_DURATION):
    """[보간 적용] 하나 또는 여러 개의 다리를 동시에 부드럽게 이동"""
    global current_leg_positions
    
    start_positions = {}
    for move in movements:
        leg_num = move['leg_num']
        start_positions[leg_num] = list(current_leg_positions[leg_num]) 

    steps = INTERPOLATION_STEPS
    step_time = duration / steps

    for n in range(1, steps + 1): # 1부터 steps까지
        group_sync_write.clearParam()
        
        for move in movements:
            leg_num = move['leg_num']
            leg_motor_ids = move['ids']
            target_pos = move['target_pos']
            start_pos = start_positions[leg_num]

            for i in range(3): # 3개 모터에 대해
                start = start_pos[i]
                end = target_pos[i]
                current_step_pos = int(start + (end - start) * (n / steps))
                motor_id = leg_motor_ids[i]
                param_goal_position = [DXL_LOBYTE(DXL_LOWORD(current_step_pos)), 
                                     DXL_HIBYTE(DXL_LOWORD(current_step_pos)), 
                                     DXL_LOBYTE(DXL_HIWORD(current_step_pos)), 
                                     DXL_HIBYTE(DXL_HIWORD(current_step_pos))]
                add_param_result = group_sync_write.addParam(motor_id, param_goal_position)
                if not add_param_result:
                    print(f"[ID:{motor_id}] groupSyncWrite addParam failed")
                    return False

        dxl_comm_result = group_sync_write.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(f"move_legs_interpolated_sync txPacket failed: {packet_handler.getTxRxResult(dxl_comm_result)}")
            return False
        
        time.sleep(step_time)

    for move in movements:
        current_leg_positions[move['leg_num']] = list(move['target_pos'])
    
    return True

def move_leg_interpolated(leg_motor_ids, leg_id_num, target_pos, duration=STEP_DURATION):
    """[보간 적용] 한쪽 다리를 부드럽게 이동 (래퍼 함수)"""
    movement_data = [{
        'ids': leg_motor_ids, 
        'leg_num': leg_id_num, 
        'target_pos': target_pos
    }]
    return move_legs_interpolated_sync(movement_data, duration=duration)


# --- [UPDATE] 전진 (w, W) (다리 순서: 3->2->4->1) ---
def move_forward(fast=False):
    """
    [UPDATE]
    - Leg 1,4 (Left)는 GAIT_SEQ_2_FWD 사용
    - Leg 2,3 (Right)는 GAIT_SEQ_1_FWD 사용
    - 다리 순서: 3 -> 2 -> 4 -> 1
    """
    
    # 속도 결정
    if fast:
        power_duration = FAST_STEP_DURATION
        return_duration = FAST_RETURN_STEP_DURATION
        print("--- Moving Forward (FAST) ---")
    else:
        power_duration = STEP_DURATION
        return_duration = RETURN_STEP_DURATION
        print("--- Moving Forward (Cycle: 3->2->4->1) ---") # [UPDATE] 순서

    
    # --- 1. 3번 다리 (Right) (Power Stroke) ---
    print("Moving Leg 3 (Right) (Power)...")
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_FWD[1], duration=power_duration): return
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_FWD[2], duration=power_duration): return
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_FWD[3], duration=power_duration): return

    # --- 2. 2번 다리 (Right) (Power 0->1) + 3번 다리 (Right) (Return 3->4) ---
    print("Moving Leg 2 (Right) (Power 0->1) + Leg 3 (Right) (Return 3->4)...")
    if not move_legs_interpolated_sync([
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_FWD[1]}, # Power 0->1
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_FWD[4]}  # [ASYNC] Return 3->4
    ], duration=return_duration): return
    
    # Leg 2 나머지 Power Stroke
    if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_FWD[2], duration=power_duration): return
    if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_FWD[3], duration=power_duration): return

    # --- [UPDATE] 3. 4번 다리 (Left) (Power 0->1) + 2번 다리 (Right) (Return 3->4) ---
    print("Moving Leg 4 (Left) (Power 0->1) + Leg 2 (Right) (Return 3->4)...")
    if not move_legs_interpolated_sync([
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_2_FWD[1]}, # Power 0->1
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_FWD[4]}  # [ASYNC] Return 3->4
    ], duration=return_duration): return
    
    # Leg 4 나머지 Power Stroke
    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2_FWD[2], duration=power_duration): return
    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2_FWD[3], duration=power_duration): return
    
    # --- [UPDATE] 4. 1번 다리 (Left) (Power 0->1) + 4번 다리 (Left) (Return 3->4) ---
    print("Moving Leg 1 (Left) (Power 0->1) + Leg 4 (Left) (Return 3->4)...")
    if not move_legs_interpolated_sync([
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2_FWD[1]}, # Power 0->1
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_2_FWD[4]}  # [ASYNC] Return 3->4
    ], duration=return_duration): return
    
    # Leg 1 나머지 Power Stroke
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_FWD[2], duration=power_duration): return
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_FWD[3], duration=power_duration): return

    # --- [UPDATE] 5. Cleanup: 1번 다리 (Left) (Return 3->4) ---
    print("Finishing cycle: Leg 1 returning...")
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_FWD[4], duration=return_duration): return
    
    print("--- Forward Step Finished ---")


# # --- [UPDATE] 좌회전 (a) (다리 순서: 3->2->4->1) ---
# def turn_left():
#     """
#     [NEW] 좌회전 (순서: 3->2->4->1)
#     - Left Legs (1, 4): REVERSE (_REV)
#     - Right Legs (2, 3): FORWARD (_FWD)
#     """
#     power_duration = STEP_DURATION
#     return_duration = RETURN_STEP_DURATION
#     print("--- Turning Left (Cycle: 3->2->4->1) ---") # [UPDATE] 순서
    
#     # --- 1. 3번 다리 (Right) (FORWARD Power) ---
#     print("Moving Leg 3 (Right) (FWD Power)...")
#     if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_FWD[1], duration=power_duration): return
#     if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_FWD[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_FWD[3], duration=power_duration): return

#     # --- 2. 2번 다리 (Right) (FWD Power 0->1) + 3번 다리 (Right) (FWD Return 3->4) ---
#     print("Moving Leg 2 (Right) (FWD Power 0->1) + Leg 3 (Right) (FWD Return 3->4)...")
#     if not move_legs_interpolated_sync([
#         {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_FWD[1]}, # FWD Power
#         {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_FWD[4]}  # [ASYNC] FWD Return
#     ], duration=return_duration): return
    
#     # Leg 2 나머지 FWD Power
#     if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_FWD[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_FWD[3], duration=power_duration): return

#     # --- [UPDATE] 3. 4번 다리 (Left) (REV Power 0->1) + 2번 다리 (Right) (FWD Return 3->4) ---
#     print("Moving Leg 4 (Left) (REV Power 0->1) + Leg 2 (Right) (FWD Return 3->4)...")
#     if not move_legs_interpolated_sync([
#         {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_2_REV[1]}, # [REV] Power
#         {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_FWD[4]}  # [ASYNC] FWD Return
#     ], duration=return_duration): return
    
#     # Leg 4 나머지 REV Power
#     if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2_REV[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2_REV[3], duration=power_duration): return
    
#     # --- [UPDATE] 4. 1번 다리 (Left) (REV Power 0->1) + 4번 다리 (Left) (REV Return 3->4) ---
#     print("Moving Leg 1 (Left) (REV Power 0->1) + Leg 4 (Left) (REV Return 3->4)...")
#     if not move_legs_interpolated_sync([
#         {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2_REV[1]}, # [REV] Power
#         {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_2_REV[4]}  # [ASYNC] [REV] Return
#     ], duration=return_duration): return
    
#     # Leg 1 나머지 REV Power
#     if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_REV[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_REV[3], duration=power_duration): return

#     # --- [UPDATE] 5. Cleanup: 1번 다리 (Left) (REV Return 3->4) ---
#     print("Finishing cycle: Leg 1 returning...")
#     if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_REV[4], duration=return_duration): return
    
#     print("--- Turn Left Step Finished ---")

# # --- [UPDATE] 우회전 (d) (다리 순서: 3->2->4->1) ---
# def turn_right():
#     """
#     [NEW] 우회전 (순서: 3->2->4->1)
#     - Left Legs (1, 4): FORWARD (_FWD)
#     - Right Legs (2, 3): REVERSE (_REV)
#     """
#     power_duration = STEP_DURATION
#     return_duration = RETURN_STEP_DURATION
#     print("--- Turning Right (Cycle: 3->2->4->1) ---") # [UPDATE] 순서
    
#     # --- 1. 3번 다리 (Right) (REVERSE Power) ---
#     print("Moving Leg 3 (Right) (REV Power)...")
#     if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REV[1], duration=power_duration): return
#     if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REV[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REV[3], duration=power_duration): return

#     # --- 2. 2번 다리 (Right) (REV Power 0->1) + 3번 다리 (Right) (REV Return 3->4) ---
#     print("Moving Leg 2 (Right) (REV Power 0->1) + Leg 3 (Right) (REV Return 3->4)...")
#     if not move_legs_interpolated_sync([
#         {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REV[1]}, # [REV] Power
#         {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_REV[4]}  # [ASYNC] [REV] Return
#     ], duration=return_duration): return
    
#     # Leg 2 나머지 REV Power
#     if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REV[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REV[3], duration=power_duration): return

#     # --- [UPDATE] 3. 4번 다리 (Left) (FWD Power 0->1) + 2번 다리 (Right) (REV Return 3->4) ---
#     print("Moving Leg 4 (Left) (FWD Power 0->1) + Leg 2 (Right) (REV Return 3->4)...")
#     if not move_legs_interpolated_sync([
#         {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_2_FWD[1]}, # FWD Power
#         {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REV[4]}  # [ASYNC] [REV] Return
#     ], duration=return_duration): return
    
#     # Leg 4 나머지 FWD Power
#     if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2_FWD[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2_FWD[3], duration=power_duration): return
    
#     # --- [UPDATE] 4. 1번 다리 (Left) (FWD Power 0->1) + 4번 다리 (Left) (FWD Return 3->4) ---
#     print("Moving Leg 1 (Left) (FWD Power 0->1) + Leg 4 (Left) (FWD Return 3->4)...")
#     if not move_legs_interpolated_sync([
#         {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2_FWD[1]}, # FWD Power
#         {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_2_FWD[4]}  # [ASYNC] FWD Return
#     ], duration=return_duration): return
    
#     # Leg 1 나머지 FWD Power
#     if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_FWD[2], duration=power_duration): return
#     if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_FWD[3], duration=power_duration): return

#     # --- [UPDATE] 5. Cleanup: 1번 다리 (Left) (FWD Return 3->4) ---
#     print("Finishing cycle: Leg 1 returning...")
#     if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2_FWD[4], duration=return_duration): return
    
#     print("--- Turn Right Step Finished ---")


# --- 메인 루프 ---
def main():
    global current_leg_positions
    
    enable_torque(DXL_ID_P2)
    
    # [UPDATE] 'i', 's'를 위한 느린 동작 적용
    print(f"Initializing robot slowly to {INITIAL_POS}...")
    init_movements = [
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': INITIAL_POS},
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': INITIAL_POS},
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': INITIAL_POS},
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': INITIAL_POS}
    ]
    move_legs_interpolated_sync(init_movements, duration=POSE_CHANGE_DURATION)
    print("Robot Initialized. Ready for commands.")

    print("------------------------------------------------")
    print("Dynamixel Gait Control (New Gait Seq)")
    print(f"Leg 1: Front-Left, Leg 2: Front-Right")
    print(f"Leg 3: Back-Right,  Leg 4: Back-Left")
    print("------------------------------------------------")
    print(f"Press 'i' -> Initialize Pose {INITIAL_POS} (Slowly)")
    print(f"Press 's' -> Stand/Start Pose {START_POS} (Slowly)")
    print("Press 'w' -> Move Forward (Normal Speed)")
    print("Press 'W' (Shift+w) -> Move Forward (FAST Speed)")
    print("Press 'a' -> Turn Left")
    print("Press 'd' -> Turn Right")
    print("Press 't' -> Disable torque for ALL motors")
    print("Press 'q' -> Quit and close port")
    print("------------------------------------------------")

    while True:
        key = get_keypress()

        if key == 'i':
            # [UPDATE] 'i' 느린 동작
            print(f"Moving all legs slowly to Initial Pose {INITIAL_POS}...")
            movements = [
                {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': INITIAL_POS},
                {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': INITIAL_POS},
                {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': INITIAL_POS},
                {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': INITIAL_POS}
            ]
            move_legs_interpolated_sync(movements, duration=POSE_CHANGE_DURATION)
            print("Done.")
        
        elif key == 's':
            # [UPDATE] 's' 느린 동작
            print(f"Moving all legs slowly to Start Pose {START_POS}...")
            movements = [
                {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': START_POS},
                {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': START_POS},
                {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': START_POS},
                {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': START_POS}
            ]
            move_legs_interpolated_sync(movements, duration=POSE_CHANGE_DURATION)
            print("Done.")

        elif key == 'w':
            move_forward(fast=False)
            
        elif key == 'W':
            move_forward(fast=True)

        # elif key == 'a':
        #     turn_left()
            
        # elif key == 'd':
        #     turn_right()

        elif key == 't':
            print("Disabling torque for all motors...")
            unlock_torque(DXL_ID_P2)
            print("Done.")
        
        elif key == 'q':
            print("Disabling torque on all motors and exiting...")
            unlock_torque(DXL_ID_P2)
            break
        
        time.sleep(0.01)

    port_handler.closePort()

if __name__ == "__main__":
    main()

