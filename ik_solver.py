import time, math
from math import sin, cos, pi, atan2, sqrt
# from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite  # 실제 환경에서 주석 해제

DT = 0.004        # 250 Hz
T  = 0.8          # 보행 주기 [s]
S  = 0.12         # 보폭 [m]
H  = 0.05         # 발 스윙 높이 [m]
BETA = 0.6        # 듀티팩터(스탠스 비율)
BODY_H = 0.18     # 몸-발 기본 높이 [m]

# 각 다리의 힙 위치(몸체 좌표계)와 위상 (예: Trot)
LEGS = {
  "FL": {"hip":[+0.18, +0.10, 0.0], "phase": 0.0,    "ids": (1,2,3)},   # (HAA,HFE,KFE)
  "FR": {"hip":[+0.18, -0.10, 0.0], "phase": pi,     "ids": (4,5,6)},
  "RL": {"hip":[-0.18, +0.10, 0.0], "phase": pi,     "ids": (7,8,9)},
  "RR": {"hip":[-0.18, -0.10, 0.0], "phase": 0.0,    "ids": (10,11,12)},
}

# 링크 길이 (예시)
L1 = 0.08  # hip->knee
L2 = 0.10  # knee->foot

def foot_trajectory(phase, t):
    """주기 T, 듀티팩터 BETA 기준으로 발끝 (x,z) 생성. x: 전진(+), z: 위(+). y는 HAA에서 처리."""
    tau = ((t / T) + phase/(2*pi)) % 1.0
    if tau < BETA:
        # Stance: 지면 미끄러지듯 뒤로 이동 (보행 진행 반대)
        s = tau / BETA
        x = ( +S/2 ) - s * (S)     # +S/2 → -S/2
        z = 0.0
    else:
        # Swing: -S/2 → +S/2 로 이동, 높이 H 사이클로이드
        s = (tau - BETA) / (1.0 - BETA)
        x = (-S/2) + s * (S)
        z = H * (1 - cos(pi * s)) / 2.0 * 2.0  # 부드럽게 오르내리기
    return x, z

def ik_3dof(hip_pos, foot_local):
    """
    입력:
      hip_pos: 몸체 좌표계 힙 위치 [x,y,z]
      foot_local: 힙 기준 발끝 위치 [x,y,z]  (힙 프레임)
    출력: (q_haa, q_hfe, q_kfe)
    """
    x, y, z = foot_local
    # HAA: y 방향을 관절 각도로(단순화: y를 측면 회전으로 변환)
    q_haa = atan2(y, abs(hip_pos[0])+1e-6)  # 로봇별 정의에 맞게 조정 필요

    # HFE/KFE: y를 제거한 2D 평면(힙면)에서 계산
    R = sqrt(x**2 + z**2)  # 평면 거리
    # 코사인 법칙
    cK = (L1**2 + L2**2 - R**2) / (2*L1*L2)
    cK = max(min(cK, 1.0), -1.0)
    q_kfe = pi - math.acos(cK)

    # 허벅지 각
    cH = (L1**2 + R**2 - L2**2) / (2*L1*R + 1e-9)
    cH = max(min(cH, 1.0), -1.0)
    q_hfe_mid = math.acos(cH)
    q_hfe = atan2(-z, x) - q_hfe_mid  # 좌표 정의에 맞게 부호 조정

    return q_haa, q_hfe, q_kfe

def body_to_hip(hip, foot_world):
    """몸체 기준 힙 위치 hip와 ‘원하는 발끝(몸 기준)’에서 힙 로컬로 변환(롤/피치 0 가정)."""
    return [foot_world[0] - hip[0], foot_world[1] - hip[1], foot_world[2] - hip[2]]

def clamp(a, lo, hi): return max(lo, min(hi, a))

# ---- Dynamixel 초기화/SyncWrite 준비 (여기선 스텁) ----
def send_positions_rad(id_list, q_list):
    """라디안 → 모터 유닛 변환 후 그룹 동기 전송. 실제에선 모델/해상도에 맞게 변환."""
    # 예: X-series 0~4095 = 0~2π → pos = int((q%(2*pi)) * 4095/(2*pi))
    pass

# ---- 메인 루프 ----
t0 = time.time()
while True:
    t = time.time() - t0

    motor_ids = []
    motor_q   = []

    for name, leg in LEGS.items():
        # 1) 발끝 목표 생성 (몸 기준 전진 x, 위 z; 기본 높이 BODY_H만큼 아래로)
        x_rel, z_rel = foot_trajectory(leg["phase"], t)
        foot_world = [x_rel, 0.0, -BODY_H + z_rel]

        # 2) 힙 로컬 좌표로 변환
        foot_local = body_to_hip(leg["hip"], foot_world)

        # 3) IK
        q_haa, q_hfe, q_kfe = ik_3dof(leg["hip"], foot_local)

        # (필요 시 오프셋/초기각 보정값 더해주기)
        # q_haa += offset_haa[name]; q_hfe += offset_hfe[name]; q_kfe += offset_kfe[name]

        # 4) 한계/속도 보호
        q_haa = clamp(q_haa, -0.8, 0.8)
        q_hfe = clamp(q_hfe, -1.5, 1.0)
        q_kfe = clamp(q_kfe, -2.4, 0.0)

        # 5) 모터 리스트에 누적
        ids = leg["ids"]
        motor_ids += list(ids)
        motor_q   += [q_haa, q_hfe, q_kfe]

    # 6) 동기 전송
    send_positions_rad(motor_ids, motor_q)

    time.sleep(DT)
