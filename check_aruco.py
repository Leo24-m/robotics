#!/usr/bin/env python3
import time
import numpy as np
import pyrealsense2 as rs
import cv2

# -----------------------------
# Config
# -----------------------------
MARKER_DICT_TYPE = cv2.aruco.DICT_4X4_100
MARKER_LENGTH = 0.0268   # meters (6 cm)
WINDOW_NAME = "Aruco Distance + Axis (RealSense)"

FRAME_W, FRAME_H, FPS = 640, 480, 30

# 텍스트 표시 설정
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_PAD = 4

# 축 길이(시각화용). 마커 크기 기준으로 적당히.
AXIS_LEN = MARKER_LENGTH * 0.75


# -----------------------------
# Utilities
# -----------------------------
def get_distance_at_point(depth_frame, x, y, radius=3, max_dist=10.0):
    """
    depth_frame.get_distance(x,y)를 주변 radius 영역에서 median으로 안정화
    """
    h, w = depth_frame.get_height(), depth_frame.get_width()
    if x < radius or y < radius or x >= w - radius or y >= h - radius:
        return None

    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = depth_frame.get_distance(x + dx, y + dy)
            if d > 0 and d < max_dist:
                vals.append(d)

    if not vals:
        return None
    return float(np.median(vals))


def draw_text_box(img, text, center_xy, bg=(0, 0, 0), fg=(255, 255, 255)):
    """
    center_xy를 기준으로 텍스트를 '가운데 정렬'로 그리기 + 배경 박스
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, TEXT_SCALE, TEXT_THICKNESS)

    cx, cy = center_xy
    tx = int(cx - tw / 2)
    ty = int(cy + th / 2)

    # background rectangle
    x1 = tx - TEXT_PAD
    y1 = ty - th - TEXT_PAD
    x2 = tx + tw + TEXT_PAD
    y2 = ty + baseline + TEXT_PAD
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)

    # text
    cv2.putText(img, text, (tx, ty), font, TEXT_SCALE, fg, TEXT_THICKNESS, cv2.LINE_AA)


# -----------------------------
# Main
# -----------------------------
def main():
    # RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FPS)

    align = rs.align(rs.stream.color)

    # Start
    profile = pipeline.start(config)

    # Intrinsics (color)
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1]
    ], dtype=np.float32)

    dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

    print("RealSense started.")
    print("Camera Matrix:\n", camera_matrix)
    print("Dist Coeffs:\n", dist_coeffs)

    # ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(MARKER_DICT_TYPE)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    last_time = time.time()
    fps_est = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            disp = color_img.copy()

            # FPS estimate
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt) if fps_est > 0 else (1.0 / dt)

            # 1) Detect markers
            corners, ids, _ = detector.detectMarkers(color_img)
            print(corners, ids)
            if ids is not None and len(ids) > 0:
                ids = ids.flatten()                             
                print(ids)
                # (선택) 마커 박스/ID 기본 그리기
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)

                # 2) Pose 추정 (각 마커 rvec/tvec)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH, camera_matrix, dist_coeffs
                )

                # 3) 각 마커별로 거리/축/텍스트 표시
                for i, marker_id in enumerate(ids):
                    c = corners[i][0]  # (4,2)
                    mcx = int(np.mean(c[:, 0]))
                    mcy = int(np.mean(c[:, 1]))

                    rvec = rvecs[i].reshape(3, 1)
                    tvec = tvecs[i].reshape(3, 1)

                    # 거리: 기본은 tvec norm (pose 기반)
                    dist_pose = float(np.linalg.norm(tvec))

                    # (선택) depth 기반도 같이 구해서 sanity 체크(원하면 사용)
                    dist_depth = get_distance_at_point(depth_frame, mcx, mcy, radius=3)

                    # 최종 거리 선택: pose가 튀는 경우 depth로 대체하고 싶으면 아래 로직 사용
                    dist_final = dist_pose
                    if dist_depth is not None:
                        # pose와 depth가 너무 다르면 depth를 채택 (임계값은 상황 따라 조절)
                        if abs(dist_pose - dist_depth) > 0.25:
                            dist_final = dist_depth

                    # 3) 축(axis) 그리기
                    try:
                        cv2.drawFrameAxes(disp, camera_matrix, dist_coeffs, rvec, tvec, AXIS_LEN)
                    except Exception:
                        pass

                    # 2) 거리 텍스트를 "마커 위"에 표시
                    text = f"ID {int(marker_id)} | {dist_final:.2f} m"
                    draw_text_box(disp, text, (mcx, mcy))

            # HUD
            cv2.putText(disp, f"FPS: {fps_est:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()
