import pandas as pd

# --- 설정 (🚨 사용자 환경에 맞게 수정) ---
file_path = './logs/dxl_3_4_5_stream_20251025_173941.csv'
output_path = './logs/2_leg.csv' 

TIME_COLUMN_NAME = 'time_iso'     # 시간 기준 열 (ISO 형식 문자열 가정)

# 모터 각도 열 이름 (🚨 실제 CSV 파일의 열 이름과 일치하는지 확인하세요!)
COLUMNS_TO_KEEP = [
    TIME_COLUMN_NAME, 
    'id3_ticks', 
    'id4_ticks', 
    'id5_ticks'
]

START_TIME = 2.0  # 시작 시간 (2초)
END_TIME = 14.0   # 종료 시간 (14초)
# ----------------------------------------

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")
    exit()

if TIME_COLUMN_NAME in df.columns:
    
    # 1. 'time_iso'를 datetime 객체로 변환
    try:
        # errors='coerce'를 사용해 파싱 실패 시 NaT(Not a Time)로 변환
        df[TIME_COLUMN_NAME] = pd.to_datetime(df[TIME_COLUMN_NAME], errors='coerce')
    except Exception as e:
        print(f"❌ 오류: 시간 열 '{TIME_COLUMN_NAME}'을 datetime으로 변환 실패. 원인: {e}")
        exit()
        
    # NaT 값 제거 (파싱 불가능한 행 제거)
    df = df.dropna(subset=[TIME_COLUMN_NAME])
    
    # 데이터프레임이 비어있는지 다시 확인
    if df.empty:
        print(f"❌ 오류: 유효한 시간 데이터가 없어 필터링할 데이터가 없습니다.")
        exit()

    # 2. 상대 시간('t_sec') 열 생성 (필터링의 기준이 됩니다)
    t0 = df[TIME_COLUMN_NAME].iloc[0]
    df['t_sec'] = (df[TIME_COLUMN_NAME] - t0).dt.total_seconds()
    
    # 3. 상대 시간 기준으로 데이터 필터링
    filtered_df = df[
        (df['t_sec'] >= START_TIME) & 
        (df['t_sec'] <= END_TIME)
    ]
    
    # 4. 원하는 열만 선택하여 저장 (시간 열과 모터 각도 열)
    # COLUMNS_TO_KEEP에 't_sec' 열을 추가하여 함께 저장하는 것이 유용합니다.
    final_columns = COLUMNS_TO_KEEP + ['t_sec']
    
    # 실제 데이터프레임에 존재하는 열만 선택
    columns_to_save = [col for col in final_columns if col in filtered_df.columns]
    
    final_df = filtered_df[columns_to_save]
    
    # 5. 새로운 CSV 파일로 저장
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ 성공적으로 데이터를 필터링하고 '{output_path}'에 저장했습니다.")
    print(f"원본 데이터 행 수: {len(pd.read_csv(file_path))}")
    print(f"저장된 데이터 행 수: {len(final_df)}")
    
else:
    print(f"❌ 오류: CSV 파일에 지정된 시간 열 '{TIME_COLUMN_NAME}'이 없습니다.")