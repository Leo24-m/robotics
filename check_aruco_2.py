#!/usr/bin/env python3
import time
import numpy as np
import pyrealsense2 as rs
import cv2

# -----------------------------
# Config
# -----------------------------
MARKER_DICT_TYPE = cv2.aruco.DICT_4X4_100
MARKER_LENGTH = 0.0268       # meters (마커 한 변 실제 길이)
GOAL_FORWARD_M = 0.18        # 마커 평면 기준 +Z로 18cm (여기만 바꾸면 됨)

FRAME_W, FRAME_H, FPS = 640, 480, 30
WINDOW_NAME = "Aruco +Z goal (depth-translation) | x,z,yaw,dist"

TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_PAD = 4
AXIS_LEN = MARKER_LENGTH * 0.75


# -----------------------------
# Utilities
# -----------------------------
def get_distance_at_point(depth_frame, x, y, radius=3, max_dist=10.0):
    """depth_frame.get_distance(x,y)를 주변 radius 영역에서 median으로 안정화"""
    h, w = depth_frame.get_height(), depth_frame.get_width()
    if x < radius or y < radius or x >= w - radius or y >= h - radius:
        return None

    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = depth_frame.get_distance(x + dx, y + dy)
            if d > 0 and d < max_dist:
                vals.append(d)

    return float(np.median(vals)) if vals else None


def draw_text_box(img, text, center_xy, bg=(0, 0, 0), fg=(255, 255, 255)):
    """center_xy를 기준으로 텍스트 가운데 정렬 + 배경 박스"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, TEXT_SCALE, TEXT_THICKNESS)

    cx, cy = center_xy
    tx = int(cx - tw / 2)
    ty = int(cy + th / 2)

    x1 = tx - TEXT_PAD
    y1 = ty - th - TEXT_PAD
    x2 = tx + tw + TEXT_PAD
    y2 = ty + baseline + TEXT_PAD
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(img, text, (tx, ty), font, TEXT_SCALE, fg, TEXT_THICKNESS, cv2.LINE_AA)


def make_aruco_detector(dict_type):
    """OpenCV 버전 차이를 최대한 피하는 ArUco detector 생성"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return aruco_dict, params, detector
    else:
        return aruco_dict, params, None


def detect_aruco_compat(bgr_img, aruco_dict, params, detector=None):
    """OpenCV 구/신 버전 호환 detectMarkers + grayscale 강제"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    return corners, ids, rejected


def compute_goal_from_marker_using_depth_rvec(rvec, tvec_depth, d_forward=0.18):
    """
    목표점 = (depth로 얻은 마커 중심 3D 위치) + R(rvec) @ [0,0,d_forward]
    - translation은 depth 기반(스케일 정확)
    - 방향(+Z)은 rvec 기반(마커 평면 법선 방향)

    return:
      xg, yg, zg : goal in camera frame (m)
      yaw(rad)   : atan2(xg,zg)
      dist_xz(m) : sqrt(x^2+z^2)
      dist_3d(m) : sqrt(x^2+y^2+z^2)
    """
    rvec = rvec.reshape(3, 1).astype(np.float32)
    tvec_depth = tvec_depth.reshape(3, 1).astype(np.float32)

    R, _ = cv2.Rodrigues(rvec)  # marker->camera rotation
    offset_marker = np.array([[0.0], [0.0], [float(d_forward)]], dtype=np.float32)

    p_goal = tvec_depth + (R @ offset_marker)

    xg = float(p_goal[0, 0])
    yg = float(p_goal[1, 0])
    zg = float(p_goal[2, 0])

    yaw = float(np.arctan2(xg, zg))
    dist_xz = float(np.sqrt(xg * xg + zg * zg))
    dist_3d = float(np.linalg.norm(p_goal))

    return xg, yg, zg, yaw, dist_xz, dist_3d


# -----------------------------
# Main
# -----------------------------
def main():
    print("OpenCV:", cv2.__version__)
    print("Aruco available:", hasattr(cv2, "aruco"))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FPS)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    # Intrinsics (color stream)
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

    print("RealSense started.")
    print("Camera Matrix:\n", camera_matrix)
    print("Dist Coeffs:\n", dist_coeffs)

    # ArUco detector
    aruco_dict, aruco_params, detector = make_aruco_detector(MARKER_DICT_TYPE)

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
            corners, ids, _ = detect_aruco_compat(color_img, aruco_dict, aruco_params, detector)

            if ids is not None and len(ids) > 0:
                ids = ids.flatten().astype(int)

                # Draw marker borders + ids
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)

                # 2) Pose estimation (for rvec / axis)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH, camera_matrix, dist_coeffs
                )

                for i, marker_id in enumerate(ids):
                    c = corners[i][0]
                    mcx = int(np.mean(c[:, 0]))
                    mcy = int(np.mean(c[:, 1]))

                    rvec = rvecs[i].reshape(3, 1)
                    tvec_pose = tvecs[i].reshape(3, 1)

                    # draw axis using pose (시각화용)
                    try:
                        cv2.drawFrameAxes(disp, camera_matrix, dist_coeffs, rvec, tvec_pose, AXIS_LEN)
                    except Exception:
                        pass

                    # distance from pose (참고용)
                    dist_pose = float(np.linalg.norm(tvec_pose))

                    # ---- depth 기반 marker center 3D (translation) ----
                    dist_depth = get_distance_at_point(depth_frame, mcx, mcy, radius=3)

                    # depth가 없으면 fallback으로 pose를 사용 (goal도 신뢰도 낮아짐)
                    if dist_depth is None:
                        tvec_depth = tvec_pose.copy()
                        dist_used = dist_pose
                        dist_src = "pose"
                    else:
                        # deproject pixel to 3D (camera frame, meters)
                        p = rs.rs2_deproject_pixel_to_point(
                            intr, [float(mcx), float(mcy)], float(dist_depth)
                        )
                        tvec_depth = np.array([[p[0]], [p[1]], [p[2]]], dtype=np.float32)
                        dist_used = dist_depth
                        dist_src = "depth"

                    # ---- 목표점: depth translation + rvec rotation ----
                    xg, yg, zg, yaw, dist_xz, dist_3d = compute_goal_from_marker_using_depth_rvec(
                        rvec, tvec_depth, d_forward=GOAL_FORWARD_M
                    )
                    yaw_deg = yaw * 180.0 / np.pi

                    # 표시 텍스트
                    line1 = f"ID {marker_id} | marker dist({dist_src}): {dist_used:.2f} m | pose: {dist_pose:.2f} m"
                    line2 = f"goal(+Z {GOAL_FORWARD_M*100:.0f}cm): x={xg:.2f} z={zg:.2f}  (y={yg:.2f})"
                    line3 = f"yaw=atan2(x,z)={yaw_deg:.1f} deg | go_dist_xz={dist_xz:.2f} m | go_dist_3d={dist_3d:.2f} m"

                    draw_text_box(disp, line1, (mcx, mcy))
                    draw_text_box(disp, line2, (mcx, mcy + 28))
                    draw_text_box(disp, line3, (mcx, mcy + 56))

            # HUD
            cv2.putText(disp, f"FPS: {fps_est:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(disp, "Press q / ESC to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

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
