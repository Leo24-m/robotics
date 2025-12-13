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
LEG1_IDS = [0, 1, 2] # Front-Right
LEG2_IDS = [3, 4, 5] # Front-Left
LEG3_IDS = [6, 7, 8] # Back-Left
LEG4_IDS = [9, 10, 11] # Back-Right

# --- 정적 자세 위치 (0-4095) ---
INITIAL_POS = [2048, 2048, 2048] # 'i' (Initialize)용
START_POS = [2048, 2141, 2228]   # 's' (Start/Stand)용 (Pos 0)

# --- [UPDATE] 전진 보행 시퀀스 ---
# 2번, 3번 다리용 (Left Side)
GAIT_SEQ_1 = [
    [2048, 2141, 2228], # Pos 0 (Start/End)
    [2048, 1679, 1896], # Pos 1
    [2861, 1679, 1896], # Pos 2
    [2861, 2141, 2228], # Pos 3
    [2048, 2141, 2228],  # Pos 4 (Return to Pos 0)
    [1248, 2141, 2228],  # Pos 5
    [2060, 1897, 2362],  # Pos 6pull
    [1748, 1991, 1797],  # Pos 7push
    [2348, 2141, 2228],  # Pos 8little front 
    [1748, 2141, 2228],   # Pos 9little backward
    [2048, 2382, 1467], # Pos 10
    [2861, 2141, 2228],  # Pos 11
    [1248, 1679, 1896]

]
# 1번, 4번 다리용 (Right Side)
GAIT_SEQ_2 = [
    [2048, 2141, 2228], # Pos 0 (Start/End)
    [2048, 1679, 1896], # Pos 1
    [1512, 1679, 1896], # Pos 2
    [1512, 2141, 2228], # Pos 3
    [2048, 2141, 2228]  # Pos 4 (Return to Pos 0)
]

# --- [NEW] 후진 (회전)용 보행 시퀀스 ---
# 파워 스트로크(1->2->3)를 역순(3->2->1)으로 수행
GAIT_SEQ_1_REVERSE = [
    GAIT_SEQ_1[0], # Pos 0
    GAIT_SEQ_1[3], # Pos 1 (Reverse)
    GAIT_SEQ_1[2], # Pos 2 (Reverse)
    GAIT_SEQ_1[1], # Pos 3 (Reverse)
    GAIT_SEQ_1[4],  # Pos 4 (Return to Pos 0)
    GAIT_SEQ_1[5],
    GAIT_SEQ_1[11],
    GAIT_SEQ_1[12]
]
GAIT_SEQ_2_REVERSE = [
    GAIT_SEQ_2[0], # Pos 0
    GAIT_SEQ_2[3], # Pos 1 (Reverse)
    GAIT_SEQ_2[2], # Pos 2 (Reverse)
    GAIT_SEQ_2[1], # Pos 3 (Reverse)
    GAIT_SEQ_2[4]  # Pos 4 (Return to Pos 0)
]

# --- 보간 설정 ---
STEP_DURATION = 0.1  # (Normal) Power Stroke 시간 (Pos 0->1, 1->2, 2->3 용)
RETURN_STEP_DURATION = STEP_DURATION * 5 # (Normal) Return Stroke 시간 (Pos 3->4 용)
INTERPOLATION_STEPS = 25 # 보간 단계 수

# --- [NEW] 고속 보행 설정 ---
FAST_STEP_DURATION = STEP_DURATION * 0.8 # (Fast) Power Stroke 시간
FAST_RETURN_STEP_DURATION = RETURN_STEP_DURATION * 0.8 # (Fast) Return Stroke 시간

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

def move_all_legs_sync(target_pos):
    """[보간 없음] 모든 다리를 동시에 target_pos로 이동"""
    global current_leg_positions
    group_sync_write.clearParam()
    
    all_leg_ids = [LEG1_IDS, LEG2_IDS, LEG3_IDS, LEG4_IDS]
    
    for leg_motors in all_leg_ids:
        for i in range(3): # 다리 모터 3개
            motor_id = leg_motors[i]
            position = target_pos[i]
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(position)), 
                                 DXL_HIBYTE(DXL_LOWORD(position)), 
                                 DXL_LOBYTE(DXL_HIWORD(position)), 
                                 DXL_HIBYTE(DXL_HIWORD(position))]
            add_param_result = group_sync_write.addParam(motor_id, param_goal_position)
            if not add_param_result:
                print(f"[ID:{motor_id}] groupSyncWrite addParam failed")

    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"move_all_legs_sync txPacket failed: {packet_handler.getTxRxResult(dxl_comm_result)}")
    else:
        for i in range(1, 5):
            current_leg_positions[i] = list(target_pos)

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


# --- [UPDATE] 전진 (w, W) ---
def move_forward(fast=False):
    """
    [UPDATE] fast 파라미터 추가
    - 순서: 3 -> 2 -> 1 -> 4
    - Power (Pos 0->1, 1->2, 2->3): power_duration
    - Return (Pos 3->4): return_duration
    """
    
    # [NEW] 속도 결정
    if fast:
        power_duration = FAST_STEP_DURATION
        return_duration = FAST_RETURN_STEP_DURATION
        print("--- Moving Forward (FAST) ---")
    else:
        power_duration = STEP_DURATION
        return_duration = RETURN_STEP_DURATION
        print("--- Moving Forward (Cycle: 3->2->1->4) ---")

    
    print("Moving Leg 3 (Power)...")
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1[3], duration=power_duration): return
    # time.sleep(0.5)
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_1[5], duration=power_duration): return
    # time.sleep(0.5)
    
    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_1[10], duration=power_duration): return
    # time.sleep(0.5)
    if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1[1], duration=power_duration): return

    
    
    if not move_legs_interpolated_sync([
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1[0]}, # Power 0->1
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_1[0]}, # Power 0->1
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1[0]}, # Power 0->1
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_1[0]} # Power 0->1

    ], duration=return_duration): return # [UPDATE] 1/5 속도 (return_duration)
    # time.sleep(0.5)

    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_1[0], duration=power_duration): return
    # time.sleep(0.5)

    print("--- Forward Step Finished ---")


# --- [NEW] 좌회전 (a) ---
def turn_left():
    """
    [NEW] 좌회전
    - 순서: 3 -> 2 -> 1 -> 4
    - Left Legs (2, 3): REVERSE (뒤로)
    - Right Legs (1, 4): NORMAL (앞으로)
    """
    power_duration = STEP_DURATION
    return_duration = RETURN_STEP_DURATION
    print("--- Turning Left (Cycle: 3->2->1->4) ---")
    
    # --- 1. 3번 다리 (REVERSE Power) ---
    print("Moving Leg 3 (REVERSE Power)...")
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REVERSE[1], duration=power_duration): return
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REVERSE[6], duration=power_duration): return
    
    print("Moving Leg 2 (REVERSE Power 0->1) + Leg 3 (REVERSE Return 3->4)...")

    if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[1], duration=power_duration): return
    if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[6], duration=power_duration): return
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_1_REVERSE[1], duration=power_duration): return
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_1_REVERSE[6], duration=power_duration): return
    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_1_REVERSE[1], duration=power_duration): return
    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_1_REVERSE[6], duration=power_duration): return
    
    # Leg 2 나머지 REVERSE Power
    # if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[2], duration=power_duration): return
    if not move_legs_interpolated_sync([
        # {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2[1]}, # NORMAL Power
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_1_REVERSE[2]},  # [ASYNC] REVERSE Return
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REVERSE[2]},  # [ASYNC] REVERSE Return
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_REVERSE[2]},
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_1_REVERSE[2]}
    ], duration=return_duration): return
    # if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REVERSE[2], duration=power_duration): return
    


    # if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[3], duration=power_duration): return

    # --- 3. 1번 다리 (NORMAL Power 0->1) + 2번 다리 (REVERSE Return 3->4) ---
    print("Moving Leg 1 (NORMAL Power 0->1) + Leg 2 (REVERSE Return 3->4)...")
    if not move_legs_interpolated_sync([
        # {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2[1]}, # NORMAL Power
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_1_REVERSE[3]},  # [ASYNC] REVERSE Return
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REVERSE[3]},  # [ASYNC] REVERSE Return
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_REVERSE[3]},
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_1_REVERSE[3]}
    ], duration=return_duration): return
    
    if not move_legs_interpolated_sync([
        # {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2[1]}, # NORMAL Power
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_1_REVERSE[4]},  # [ASYNC] REVERSE Return
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REVERSE[4]},  # [ASYNC] REVERSE Return
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_REVERSE[4]},
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_1_REVERSE[4]}
    ], duration=return_duration): return

    # # Leg 1 나머지 NORMAL Power
    # if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2[2], duration=power_duration): return
    # if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_2[3], duration=power_duration): return
    
    # # --- 4. 4번 다리 (NORMAL Power 0->1) + 1번 다리 (NORMAL Return 3->4) ---
    # print("Moving Leg 4 (NORMAL Power 0->1) + Leg 1 (NORMAL Return 3->4)...")
    # if not move_legs_interpolated_sync([
    #     {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_2[1]}, # NORMAL Power
    #     {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2[4]}  # [ASYNC] NORMAL Return
    # ], duration=return_duration): return
    
    # # Leg 4 나머지 NORMAL Power
    # if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2[2], duration=power_duration): return
    # if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2[3], duration=power_duration): return

    # # --- 5. Cleanup: 4번 다리 (NORMAL Return 3->4) ---
    # print("Finishing cycle: Leg 4 returning...")
    # if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_2[4], duration=return_duration): return
    
    # print("--- Turn Left Step Finished ---")

# --- [NEW] 우회전 (d) ---
def turn_right():
    """
    [NEW] 우회전
    - 순서: 3 -> 2 -> 1 -> 4
    - Left Legs (2, 3): NORMAL (앞으로)
    - Right Legs (1, 4): REVERSE (뒤로)
    """
    power_duration = STEP_DURATION
    return_duration = RETURN_STEP_DURATION
    print("--- Turning Left (Cycle: 3->2->1->4) ---")
    
    # --- 1. 3번 다리 (REVERSE Power) ---
    print("Moving Leg 3 (REVERSE Power)...")
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return
    if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return
    
    print("Moving Leg 2 (REVERSE Power 0->1) + Leg 3 (REVERSE Return 3->4)...")
    
    if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return
    if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return
    
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return
    if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return

    # if not move_leg_interpolated(LEG1_IDS, 1, GAIT_SEQ_1_REVERSE[0], duration=power_duration): return
    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return
    if not move_leg_interpolated(LEG4_IDS, 4, GAIT_SEQ_1_REVERSE[5], duration=power_duration): return

    # Leg 2 나머지 REVERSE Power
    # if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[2], duration=power_duration): return
    if not move_legs_interpolated_sync([
        # {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2[1]}, # NORMAL Power
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_1_REVERSE[7]},  # [ASYNC] REVERSE Return
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REVERSE[7]},  # [ASYNC] REVERSE Return
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_REVERSE[7]},
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_1_REVERSE[7]}
    ], duration=return_duration): return
    # if not move_leg_interpolated(LEG3_IDS, 3, GAIT_SEQ_1_REVERSE[2], duration=power_duration): return
    
    # if not move_leg_interpolated(LEG2_IDS, 2, GAIT_SEQ_1_REVERSE[3], duration=power_duration): return

    # --- 3. 1번 다리 (NORMAL Power 0->1) + 2번 다리 (REVERSE Return 3->4) ---
    print("Moving Leg 1 (NORMAL Power 0->1) + Leg 2 (REVERSE Return 3->4)...")
    if not move_legs_interpolated_sync([
        # {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2[1]}, # NORMAL Power
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_1_REVERSE[3]},  # [ASYNC] REVERSE Return
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REVERSE[3]},  # [ASYNC] REVERSE Return
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_REVERSE[3]},
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_1_REVERSE[3]}
    ], duration=return_duration): return
    
    if not move_legs_interpolated_sync([
        # {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_2[1]}, # NORMAL Power
        {'ids': LEG1_IDS, 'leg_num': 1, 'target_pos': GAIT_SEQ_1_REVERSE[4]},  # [ASYNC] REVERSE Return
        {'ids': LEG2_IDS, 'leg_num': 2, 'target_pos': GAIT_SEQ_1_REVERSE[4]},  # [ASYNC] REVERSE Return
        {'ids': LEG3_IDS, 'leg_num': 3, 'target_pos': GAIT_SEQ_1_REVERSE[4]},
        {'ids': LEG4_IDS, 'leg_num': 4, 'target_pos': GAIT_SEQ_1_REVERSE[4]}
    ], duration=return_duration): return


# --- 메인 루프 ---
def main():
    global current_leg_positions
    
    enable_torque(DXL_ID_P2)
    
    print(f"Initializing robot to {INITIAL_POS}...")
    move_all_legs_sync(INITIAL_POS) # 현재 위치를 INITIAL_POS로 설정
    print("Robot Initialized. Ready for commands.")

    # [UPDATE]
    print("------------------------------------------------")
    print("Dynamixel Gait Control")
    print(f"Press 'i' -> Initialize Pose {INITIAL_POS}")
    print(f"Press 's' -> Stand/Start Pose {START_POS}")
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
            print(f"Moving all legs to Initial Pose {INITIAL_POS}...")
            move_all_legs_sync(INITIAL_POS)
            print("Done.")
        
        elif key == 's':
            print(f"Moving all legs to Start Pose {START_POS}...")
            move_all_legs_sync(START_POS)
            print("Done.")

        elif key == 'w':
            move_forward(fast=False) # [UPDATE]
            
        elif key == 'W':
            move_forward(fast=True) # [NEW]

        elif key == 'a':
            turn_left() # [NEW]
            
        elif key == 'd':
            turn_right() # [NEW]

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
