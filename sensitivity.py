import cv2
import mediapipe as mp
import time
import json

# MediaPipe Pose 모델을 불러옵니다.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 비디오 캡처 객체를 생성합니다.
cap = cv2.VideoCapture("OMG_sml.mp4")

# FPS 계산을 위한 변수를 초기화합니다.
fps_start_time = time.time()
fps_frames = 0
wrist_movement_count = 0
fps = 0
cnt = 1

# 1초 동안 움직인 횟수를 저장할 변수를 초기화합니다.
move_count_1s = 0
start_time = time.time()

# mp_drawing 모듈을 로드합니다.
mp_drawing = mp.solutions.drawing_utils

# 관절과 그들을 잇는 선을 그리는 함수를 정의합니다.
# 왼쪽 손목 관절 랜드마크 ID를 가져옵니다.
def draw_pose_landmarks(image, landmarks):
    mp_drawing.draw_landmarks(
        image, landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )

# 왼쪽 손목 관절 랜드마크 ID를 가져옵니다.
RIGHT_WRIST_LANDMARK_ID = 16
sensitivity = {}
while True:
    # 비디오 프레임을 읽어옵니다.
    
    success, image = cap.read()

    if not success:
        break

    # 입력 이미지에서 포즈를 추정합니다.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 포즈 결과를 이용하여 왼쪽 손목 관절 위치를 가져옵니다.
    if results.pose_landmarks:
        # 모든 관절과 그들을 잇는 선을 표시합니다.
        draw_pose_landmarks(image, results.pose_landmarks)

        right_wrist = results.pose_landmarks.landmark[RIGHT_WRIST_LANDMARK_ID]

        if wrist_movement_count >= 5:
            # 왼쪽 손목 관절을 화면 상에 강조하여 표시합니다.
            cv2.drawMarker(image, (int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0])), (255, 0, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=8, line_type=cv2.LINE_AA)
        cv2.drawMarker(image, (int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0])), (255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=8, line_type=cv2.LINE_AA)

        # 왼쪽 손목 관절 민감도를 계산하여 표시합니다.
        # 이전 위치와 비교하여 움직임을 감지합니다.
        if 'last_right_wrist' in locals():
            dx = abs(right_wrist.x - last_right_wrist.x)
            dy = abs(right_wrist.y - last_right_wrist.y)
            distance_moved = (dx ** 2 + dy ** 2) ** 0.5
            if distance_moved > 0.01:
                wrist_movement_count += 1
        last_right_wrist = right_wrist

    # 1초마다 움직인 횟수를 계산합니다.
    if time.time() - start_time >= 1:
        wrist_movement_count = move_count_1s
        move_count_1s = 0
        start_time = time.time()
        cnt += 1
        
    sensitivity[cnt] = wrist_movement_count

    # FPS를 계산하고 화면 상단 좌측에 표시합니다.
    fps_frames += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_frames / (time.time() - fps_start_time)
        fps_frames = 0
        fps_start_time = time.time()
    cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 왼쪽 손목 관절 민감도를 화면 상단 좌측에 표시합니다.
    cv2.putText(image, "Right Wrist Sensitivity: {} mv/sec".format(wrist_movement_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 출력 이미지를 화면에 표시합니다.
    cv2.imshow("Output", image)

    # ESC 키를 누르면 프로그램을 종료합니다.
    if cv2.waitKey(1) == 27:
        break

with open("sensitivity.json", "a") as f:
    buf = 1
    for key in sensitivity:
        if key == buf:
            json.dump(sensitivity, f)
            f.write(",")
            buf = key
        else :
            json.dump(sensitivity, f)
            f.write(",")
            f.write("\n")
            buf = key   

# 비디오 캡처 객체와 창을 해제합니다.
cap.release()
cv2.destroyAllWindows()