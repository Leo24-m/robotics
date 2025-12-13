import os
import sys
import time
import pandas as pd
import numpy as np # ⭐ 보간을 위해 numpy 추가
from dynamixel_sdk import *

# --- 1. 설정 (🚨 사용자 환경에 맞게 수정) ---
# 🚨 재생할 CSV 파일 경로
CSV_FILE                = './logs/2_leg.csv' 

# 🚨 포트 및 통신 속도
DEVICENAME              = '/dev/ttyUSB0' 
BAUDRATE                = 57600
PROTOCOL_VERSION        = 2.0 

# 🚨 대상 모터 ID
DXL_IDS                 = [3, 4, 5]

# P2.0 제어 테이블 주소
ADDR_P2_TORQUE_ENABLE      = 64
ADDR_P2_OPERATING_MODE     = 11
ADDR_P2_GOAL_POSITION      = 116
LEN_P2_GOAL_POSITION       = 4

# 모드 값
P2_MODE_POSITION           = 3

# ⭐ 새롭게 추가된 설정: 보간 간격 (초)
# 0.01초(100Hz) 간격으로 데이터를 생성하여 부드러운 움직임을 유도합니다.
INTERPOLATION_INTERVAL = 0.01 
# ---------------------------------------------------

# --- 2. 다이나믹셀 핸들러 초기화 ---
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# 동시 제어를 위한 GroupSyncWrite 초기화
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, 
                                ADDR_P2_GOAL_POSITION, LEN_P2_GOAL_POSITION)

# --- 3. 헬퍼 함수 ---
def set_torque(dxl_id, status):
    """P2.0 모터의 토크를 설정합니다."""
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
        portHandler, dxl_id, ADDR_P2_TORQUE_ENABLE, status
    )
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id}] Torque Set Fail: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id}] Torque Set Error: {packetHandler.getRxPacketError(dxl_error)}")

def set_operating_mode(dxl_id, mode):
    """P2.0 모터의 작동 모드를 설정합니다. (토크가 꺼진 상태에서 해야 함)"""
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
        portHandler, dxl_id, ADDR_P2_OPERATING_MODE, mode
    )
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id}] Mode Set Fail: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id}] Mode Set Error: {packetHandler.getRxPacketError(dxl_error)}")

def cleanup():
    """종료 시 모터 토크를 끄고 포트를 닫습니다."""
    print("\nDisabling torque...")
    for dxl_id in DXL_IDS:
        set_torque(dxl_id, 0)
    portHandler.closePort()
    print("Port closed.")

def pack_ticks_to_4bytes(ticks):
    """GroupSyncWrite에 필요한 4바이트 파라미터로 변환합니다."""
    # 보간된 값은 float일 수 있으므로 정수로 변환해야 합니다.
    ticks = int(round(ticks))
    return [
        DXL_LOBYTE(DXL_LOWORD(ticks)),
        DXL_HIBYTE(DXL_LOWORD(ticks)),
        DXL_LOBYTE(DXL_HIWORD(ticks)),
        DXL_HIBYTE(DXL_HIWORD(ticks))
    ]
# ---------------------------------------------------

# --- 4. 메인 코드 ---
def main():
    # 1. 포트 열기 및 통신 속도 설정
    if not portHandler.openPort():
        print(f"Failed to open the port {DEVICENAME}"); return
    if not portHandler.setBaudRate(BAUDRATE):
        print(f"Failed to change the baudrate to {BAUDRATE}"); return
    print("Succeeded to open port and set baudrate.")

    # 2. 로그 파일 로드 및 **선형 보간** 처리
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV 파일 '{CSV_FILE}'을 찾을 수 없습니다."); return
        
    df['time_iso'] = pd.to_datetime(df['time_iso'])
    df = df.sort_values(by='time_iso')
    
    # 0초부터 시작하는 상대 시간(초) 컬럼 생성
    t0 = df['time_iso'].iloc[0]
    df['t_sec'] = (df['time_iso'] - t0).dt.total_seconds()

    # --- 선형 보간 로직 ---
    # 1. 새로운 시간 축 생성
    t_end = df['t_sec'].iloc[-1]
    new_t_sec = pd.Series(
        data=np.arange(0.0, t_end + INTERPOLATION_INTERVAL, INTERPOLATION_INTERVAL),
        name='t_sec'
    )
    
    # 2. t_sec을 인덱스로 설정
    df_indexed = df.set_index('t_sec')
    
    # 3. 새로운 시간 축에 맞춰 위치 데이터(ticks) 선형 보간
    df_resampled = df_indexed.reindex(df_indexed.index.union(new_t_sec.values))
    df_resampled = df_resampled[['id3_ticks', 'id4_ticks', 'id5_ticks']].interpolate(method='linear')
    
    # 4. 새로 생성된 시간 축으로만 필터링 및 인덱스 리셋
    df_interpolated = df_resampled.loc[new_t_sec.values].reset_index()
    
    print(f"Log file loaded. Original steps: {len(df)}")
    print(f"Interpolated steps ({INTERPOLATION_INTERVAL}s interval): {len(df_interpolated)}")
    print(f"Total duration: {t_end:.2f} seconds.")

    # 앞으로 'df_interpolated'를 재생 데이터로 사용합니다.
    df = df_interpolated 
    # -----------------------

    # 3. 모터 초기화 (토크 끄기 -> 모드 3 설정 -> 토크 켜기)
    print("Initializing motors (Setting to Position Mode)...")
    for dxl_id in DXL_IDS:
        set_torque(dxl_id, 0)
        set_operating_mode(dxl_id, P2_MODE_POSITION)
        set_torque(dxl_id, 1)

    # 4. 시작 위치로 이동
    first_row = df.iloc[0]
    print(f"Moving to start position (ID3:{first_row['id3_ticks']:.0f}, ID4:{first_row['id4_ticks']:.0f}, ID5:{first_row['id5_ticks']:.0f})...")
    
    param3 = pack_ticks_to_4bytes(first_row['id3_ticks'])
    param4 = pack_ticks_to_4bytes(first_row['id4_ticks'])
    param5 = pack_ticks_to_4bytes(first_row['id5_ticks'])
    
    groupSyncWrite.addParam(3, param3)
    groupSyncWrite.addParam(4, param4)
    groupSyncWrite.addParam(5, param5)
    
    groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()
    
    time.sleep(2.0)
    
    try:
        input("\nMotors are at start pose. Press Enter to begin playback...")
    except KeyboardInterrupt:
        cleanup(); return

    # 5. 재생(Playback) 루프
    print(f"--- Starting Playback (Total {len(df)-1} steps) ---")
    
    # 재생 시작 시간 (타이밍 보정을 위해)
    t_playback_start = time.monotonic() 
    
    # 0번(첫 번째) 행은 이미 보냈으므로 1번부터 시작
    for row in df.iloc[1:].itertuples():
        
        # 1. 이번 스텝이 실행되어야 할 *절대 시간* 계산
        t_target = t_playback_start + row.t_sec
        
        # 2. 목표 시간까지 대기 (타이밍 보정)
        t_now = time.monotonic()
        sleep_duration = t_target - t_now
        
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        elif sleep_duration < -0.05: # 50ms 이상 지연되면 경고
            print(f"WARN: Loop overrun at t={row.t_sec:.2f}s (late by {-sleep_duration*1000:.1f} ms)")
            
        # 3. GroupSyncWrite 파라미터 준비 및 전송
        param3 = pack_ticks_to_4bytes(row.id3_ticks)
        param4 = pack_ticks_to_4bytes(row.id4_ticks)
        param5 = pack_ticks_to_4bytes(row.id5_ticks)
        
        groupSyncWrite.addParam(3, param3)
        groupSyncWrite.addParam(4, param4)
        groupSyncWrite.addParam(5, param5)
        
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()
        
        # 터미널에 진행 상황 표시 
        if row.Index % 100 == 0:
             sys.stdout.write(f"\r  -> Progress: {row.t_sec:.2f}s / {df['t_sec'].iloc[-1]:.2f}s")
             sys.stdout.flush()

    print("\n--- Playback Finished ---")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        cleanup()