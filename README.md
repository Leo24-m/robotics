# 4족 로봇 자율 주행 평가 시스템

Dynamixel 모터 기반 4족 로봇의 자율 주행 성능 평가를 위한 코드베이스입니다.  
RealSense 카메라와 ArUco 마커, YOLO 객체 인식을 활용하여 다양한 네비게이션 시나리오를 평가합니다.

---

## 하드웨어 요구사항

- **로봇**: Dynamixel XL-330 모터 12개 (4다리 × 3관절)
- **카메라**: Intel RealSense D435 (Color + Depth)
- **마커**: ArUco 4x4 Dictionary (ID 1-9)
- **컴퓨터**: USB 연결 (Dynamixel U2D2)

---

## 평가 단계별 코드

### 1차 평가: 수동 보행 제어
**파일**: [`eval_1st_modi.py`](file:///mnt/nvme/workspace/robotics/eval_1st_modi.py)

키보드 입력을 통한 로봇의 기본 보행 동작 테스트

| 키 | 동작 |
|---|---|
| `i` | 초기 자세 (Initial Pose) |
| `s` | 서기 자세 (Stand Pose) |
| `w` | 전진 (Normal Speed) |
| `W` | 전진 (Fast Speed) |
| `a` | 좌회전 |
| `d` | 우회전 |
| `t` | 토크 비활성화 |
| `q` | 종료 |

```bash
python eval_1st_modi.py
```

---

### 2차 평가: YOLO 객체 추적 네비게이션
**파일**:
- [`eval_2nd_ai.py`](file:///mnt/nvme/workspace/robotics/eval_2nd_ai.py) - AI 타겟
- [`eval_2nd_awear.py`](file:///mnt/nvme/workspace/robotics/eval_2nd_awear.py) - AWEAR 타겟
- [`eval_2nd_imr.py`](file:///mnt/nvme/workspace/robotics/eval_2nd_imr.py) - IMR 타겟

YOLO 모델을 사용하여 특정 객체(AI, AWEAR, IMR)를 인식하고 자동으로 추적하여 목표 거리(30cm)까지 접근

**주요 기능**:
- RealSense 깊이 센서로 거리 측정
- 화면 중심 정렬 (±80px 허용)
- 목표 거리 도달 시 자동 정지
- 실시간 시각화 (FPS, 추론 시간 등)

```bash
python eval_2nd_ai.py      # AI 타겟 추적
python eval_2nd_awear.py   # AWEAR 타겟 추적
python eval_2nd_imr.py     # IMR 타겟 추적
```

| 키 | 동작 |
|---|---|
| `1` | AI 타겟 선택 |
| `2` | AWEAR 타겟 선택 |
| `3` | IMR 타겟 선택 |
| `0` | 가장 가까운 객체 추적 |
| `s` | 로봇 제어 활성화/비활성화 |
| `q` | 종료 |

---

### 3차 평가: 하이브리드 네비게이션 (ArUco + YOLO)
**파일**: [`eval_3rd_sidestep.py`](file:///mnt/nvme/workspace/robotics/eval_3rd_sidestep.py)

ArUco 마커 기반 경로 탐색과 YOLO 객체 추적을 결합한 복합 네비게이션 시스템

**네비게이션 로직**:
1. **SEARCHING**: 좌회전하며 마커 1 또는 2 탐색
2. **APPROACHING**: 발견한 마커에 60cm까지 접근
3. **ACTION_SIDESTEP**: 마커 ID에 따라 옆걸음 (홀수→좌, 짝수→우)
4. **M9_CENTERING**: 마커 9 발견 시 1m까지 접근 및 중앙 정렬
5. **YOLO_TRACKING**: YOLO로 최종 타겟(AI/AWEAR/IMR) 추적

**주요 기능**:
- 5프레임 ArUco 버퍼링 (노이즈 감소)
- 2m 이내 마커만 인식 (오탐 방지)
- 비동기 YOLO 모델 로딩
- 옆걸음(Sidestep) 보행 지원

```bash
python eval_3rd_sidestep.py
# 실행 시 타겟 선택: 1=AI, 2=AWEAR, 3=IMR
```

| 키 | 동작 |
|---|---|
| `s` | 로봇 활성화/비활성화 |
| `q` | 종료 |

---

## 의존성

```bash
pip install numpy opencv-python pyrealsense2 dynamixel-sdk ultralytics
```

---

## 프로젝트 구조

```
robotics/
├── eval_1st_modi.py        # 1차 평가: 키보드 수동 제어
├── eval_2nd_ai.py          # 2차 평가: AI 객체 추적
├── eval_2nd_awear.py       # 2차 평가: AWEAR 객체 추적
├── eval_2nd_imr.py         # 2차 평가: IMR 객체 추적
├── eval_2nd_modi.py        # 2차 평가용 로봇 제어 모듈
├── eval_2nd_modi_v2.py     # 3차 평가용 로봇 제어 모듈 (Sidestep 지원)
├── eval_3rd_sidestep.py    # 3차 평가: 하이브리드 네비게이션
├── yolov8n.pt              # YOLO 모델 가중치
└── logs/                   # 실행 로그
```

---

## 클래스 구조

### `DynamixelController`
Dynamixel SDK를 통한 저수준 모터 통신 관리
- `connect()` / `disconnect()`: 포트 연결 관리
- `enable_torque()` / `disable_torque()`: 토크 제어
- `sync_write_goal_position()`: 동기화 위치 전송

### `QuadrupedRobot`
4족 로봇 고수준 동작 제어
- `initialize_pose()` / `stand_pose()`: 자세 설정
- `move_forward()`: 전진 보행
- `turn_left()` / `turn_right()`: 회전
- `sidestep_left()` / `sidestep_right()`: 옆걸음 (v2)

### `YOLOObjectTracker` (2차 평가)
YOLO 기반 객체 추적
- `detect_objects()`: 객체 감지
- `get_distance_at_point()`: 깊이 측정
- `get_control_command()`: 제어 명령 결정

### `HybridEvaluator` (3차 평가)
ArUco + YOLO 하이브리드 네비게이션
- `detect_aruco()`: ArUco 마커 감지 및 3D 위치 추정
- `detect_yolo()`: YOLO 객체 감지
- `process()`: 상태 머신 기반 네비게이션 처리
