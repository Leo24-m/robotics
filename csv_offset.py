import pandas as pd
import numpy as np
import os
import sys

# --- 1. 설정 및 초기 값 정의 ---
CSV_FILE = './logs/2_leg.csv'
OUTPUT_FILE = './logs/modified_log_6_7_8.csv'

# 모터 ID 매핑 및 초기 각도
INITIAL_ANGLES = {
    0: 0.0,    1: 0.0,    2: 180.0,
    3: 165.0,  4: 180.0,  5: 180.0,
    6: 180.0,  7: 180.0,  8: 240.0,
    9: 180.0, 10: 180.0, 11: 310.0,
}

# 변환 대상 모터 ID 정의 (원본 -> 새 ID)
MAPPING = {
    3: 6,
    4: 7,
    5: 8,
}

# 다이나믹셀 변환 상수 (Protocol 2.0 가정: 4096 틱 = 360도)
MAX_TICK = 4095.0 
MAX_DEGREE = 360.0
TICK_PER_DEG = MAX_TICK / MAX_DEGREE

# --- 2. 오프셋 계산 (새 ID - 원본 ID) ---
offsets = {}
for original_id, new_id in MAPPING.items():
    if original_id in INITIAL_ANGLES and new_id in INITIAL_ANGLES:
        # 오프셋 = (새 모터의 초기각) - (원본 모터의 초기각)
        offset_deg = INITIAL_ANGLES[new_id] - INITIAL_ANGLES[original_id]
        offsets[new_id] = offset_deg
        print(f"[INFO] ID {original_id} -> ID {new_id}: 오프셋 = {offset_deg:.2f} deg")
    else:
        print(f"[ERROR] ID {original_id} 또는 {new_id}의 초기 각도를 찾을 수 없습니다.")
        sys.exit()

# --- 3. CSV 파일 로드 ---
try:
    df = pd.read_csv(CSV_FILE)
    print(f"\n[INFO] 파일 로드 성공: {CSV_FILE}, 총 {len(df)} 행")
except FileNotFoundError:
    print(f"[ERROR] 파일을 찾을 수 없습니다: {CSV_FILE}")
    sys.exit()

# --- 4. 틱 데이터 변환 및 오프셋 적용 ---
for original_id, new_id in MAPPING.items():
    tick_col = f'id{original_id}_ticks'
    new_tick_col = f'id{new_id}_ticks'
    offset_deg = offsets[new_id]
    
    if tick_col in df.columns:
        # 1. 틱 -> 각도 (Degree) 변환
        original_deg = df[tick_col] / TICK_PER_DEG
        
        # 2. 오프셋 적용 (새 각도)
        new_deg = original_deg + offset_deg
        
        # 3. 새 각도 -> 틱 변환
        new_ticks = new_deg * TICK_PER_DEG
        
        # 4. 물리적 범위(0 ~ 4095)를 벗어나지 않도록 클리핑
        new_ticks = np.clip(new_ticks, 0, MAX_TICK).round().astype(int)
        
        # 5. 새 열로 데이터프레임에 추가
        df[new_tick_col] = new_ticks
        
        print(f"[INFO] ID {original_id} 데이터 -> {new_id} 데이터 변환 완료.")
    else:
        print(f"[WARN] 원본 틱 열 '{tick_col}'이 파일에 존재하지 않아 건너뜁니다.")

# --- 5. 불필요한 열 제거 및 저장 ---
# time_iso, t_sec, 그리고 새로 생성된 6, 7, 8번 틱 열만 남깁니다.
columns_to_keep = ['time_iso', 't_sec'] + [f'id{i}_ticks' for i in MAPPING.values()]
final_df = df[[col for col in columns_to_keep if col in df.columns]]

# 파일 저장
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ 성공적으로 로그 파일을 생성했습니다: {OUTPUT_FILE}")
print(f"저장된 열: {final_df.columns.tolist()}")