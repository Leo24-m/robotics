import os
import sys
import time
import collections 
import matplotlib.pyplot as plt 
import numpy as np
from dynamixel_sdk import *

# --- 1. 설정 (🚨 사용자 환경에 맞게 수정) ---
# ... (설정 부분은 변경 없음) ...
DEVICENAME              = '/dev/ttyUSB0' 
BAUDRATE                = 57600
READ_INTERVAL_SEC       = 0.05          
MAX_DATA_POINTS         = 100           

DXL_ID_1_0              = [6, 8]              
DXL_IDS_2_0             = [7]                 
ALL_DXL_IDS             = DXL_ID_1_0 + DXL_IDS_2_0 

ADDR_P1_PRESENT_POSITION = 36
LEN_P1_PRESENT_POSITION  = 2  
ADDR_P2_PRESENT_POSITION = 132
LEN_P2_PRESENT_POSITION  = 4  
# ---------------------------------------------------

# --- 2. 핸들러 초기화 ---
portHandler = PortHandler(DEVICENAME)
packetHandler_1_0 = PacketHandler(1.0)
packetHandler_2_0 = PacketHandler(2.0)

# --- 3. 데이터 저장소 및 그래프 초기화 ---
time_data = collections.deque([0.0] * MAX_DATA_POINTS, maxlen=MAX_DATA_POINTS)
angle_data = {
    i: collections.deque([0.0] * MAX_DATA_POINTS, maxlen=MAX_DATA_POINTS)
    for i in ALL_DXL_IDS 
}
plt.ion() 
fig, ax = plt.subplots(figsize=(10, 6))
lines = {
    i: ax.plot(list(time_data), list(angle_data[i]), label=f'ID {i}')[0]
    for i in ALL_DXL_IDS
}

ax.set_title("Real-time Motor Position (deg)")
ax.set_xlabel("Relative Time (s)")
ax.set_ylabel("Angle (deg)")
ax.legend()
ax.grid(True)
start_time_global = time.monotonic()


# --- 4. 헬퍼 함수 ---
def setup_port():
    if not portHandler.openPort():
        print(f"Failed to open the port {DEVICENAME}"); return False
    if not portHandler.setBaudRate(BAUDRATE):
        print(f"Failed to change the baudrate to {BAVICHRATE}"); return False
    print("Succeeded to open port and set baudrate.")
    return True

def cleanup():
    portHandler.closePort()
    plt.close(fig) 
    print("\nPort closed.")

def read_position_1_0(dxl_id):
    """Protocol 1.0 모터의 현재 위치를 읽고 각도(degree)로 변환합니다."""
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler_1_0.read2ByteTxRx(
        portHandler, dxl_id, ADDR_P1_PRESENT_POSITION
    )
    if dxl_comm_result != COMM_SUCCESS:
        # ⭐ P1.0 통신 실패 시 터미널에 출력하여 문제 진단에 도움을 줍니다.
        # print(f"[WARN] ID {dxl_id} (P1.0) Comm Failure: {packetHandler_1_0.getTxRxResult(dxl_comm_result)}")
        return None
    elif dxl_error != 0:
        # print(f"[WARN] ID {dxl_id} (P1.0) Packet Error: {packetHandler_1_0.getRxPacketError(dxl_error)}")
        return None
        
    angle_deg = round((dxl_present_position / 1023.0) * 300.0, 2)
    return angle_deg


# --- 5. 그래프 업데이트 함수 ---
def update_plot(current_angles):
    """데이터를 업데이트하고 그래프를 다시 그립니다."""
    global start_time_global
    current_time = time.monotonic() - start_time_global
    
    # 1. 데이터 업데이트
    time_data.append(current_time)
    for i in ALL_DXL_IDS:
        # ⭐ 수정: angle_data[i]가 비어있지 않다는 것을 보장하므로,
        # 읽기 실패 시 이전 값 (angle_data[i][-1])을 사용하도록 명시합니다.
        # 초기화 시 0으로 채워졌으므로 .append(0.0) 대신 이전 값을 사용합니다.
        
        # 기본값으로 이전 값을 사용하고, angle_data가 완전히 비어있을 경우 (발생할 일 없음) 0.0을 사용
        last_value = angle_data[i][-1] if angle_data[i] else 0.0
        angle_data[i].append(current_angles.get(i, last_value)) 

    # 2. 그래프 데이터 설정
    x = list(time_data)
    for i in ALL_DXL_IDS:
        lines[i].set_data(x, list(angle_data[i]))
        
    # 3. 축 범위 동적 업데이트 (Y축 동적 설정)
    ax.set_xlim(x[0], x[-1] + 0.1)
    
    # 모든 모터 데이터 결합
    all_y = np.concatenate([list(angle_data[i]) for i in ALL_DXL_IDS])
    
    if all_y.size > 0 and current_time > READ_INTERVAL_SEC: 
        y_min_data = all_y.min()
        y_max_data = all_y.max()
        
        y_min_plot = y_min_data - 10
        y_max_plot = y_max_data + 10
        
        if y_max_plot - y_min_plot < 20: 
            center = (y_min_data + y_max_data) / 2
            y_min_plot = center - 10
            y_max_plot = center + 10

        ax.set_ylim(max(0, y_min_plot), min(360, y_max_plot)) 
    
    # 4. 그래프 다시 그리기
    fig.canvas.draw()
    fig.canvas.flush_events()


# --- 6. 메인 루프 ---
def main():
    if not setup_port():
        return

    print(f"\n--- Realtime Motor Angle Monitoring (Interval: {READ_INTERVAL_SEC}s) ---")
    print("Graph window opened. Press Ctrl+C to stop.")
    
    # GroupSyncRead 초기화 (P2.0 모터 ID 7)
    groupSyncRead = GroupSyncRead(portHandler, packetHandler_2_0, 
                                  ADDR_P2_PRESENT_POSITION, LEN_P2_PRESENT_POSITION)
    for dxl_id in DXL_IDS_2_0: 
        groupSyncRead.addParam(dxl_id)
        
    global start_time_global
    start_time_global = time.monotonic()


    try:
        while True:
            loop_start_time = time.time()
            current_angles = {}
            
            # 1. Protocol 1.0 (ID 6, 8) 위치 읽기 (순차적 읽기)
            for dxl_id in DXL_ID_1_0: 
                angle = read_position_1_0(dxl_id)
                if angle is not None:
                    # ID 6, 8 데이터 저장 (이전 수정 반영됨)
                    current_angles[dxl_id] = angle 

            # 2. Protocol 2.0 (ID 7) 위치 동시 읽기 (GroupSyncRead)
            if groupSyncRead.txRxPacket() == COMM_SUCCESS:
                
                # ID 7 위치 가져오기
                if groupSyncRead.isAvailable(DXL_IDS_2_0[0], ADDR_P2_PRESENT_POSITION, LEN_P2_PRESENT_POSITION):
                    position_7 = groupSyncRead.getData(DXL_IDS_2_0[0], ADDR_P2_PRESENT_POSITION, LEN_P2_PRESENT_POSITION)
                    current_angles[DXL_IDS_2_0[0]] = round((position_7 / 4095.0) * 360.0, 2)
                # else:
                #     # ⭐ ID 7 통신 실패 시 메시지 출력 (필요 시 주석 해제)
                #     print(f"[WARN] ID 7 (P2.0) GroupSyncRead Failed to get data.")
            
            # 3. 그래프 업데이트
            update_plot(current_angles)
            
            # 4. 시간 지연
            elapsed_time = time.time() - loop_start_time
            sleep_time = READ_INTERVAL_SEC - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")
    finally:
        cleanup()

if __name__ == "__main__":
    main()