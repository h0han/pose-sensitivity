import cv2
import mediapipe as mp
import time

# MediaPipe Pose 모델을 불러옴
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 비디오 캡처 객체를 생성
cap = cv2.VideoCapture("OMG_sml.mp4")

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 저장할 비디오 파일 이름 설정
out = cv2.VideoWriter('output_1.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# FPS 계산을 위한 변수를 초기화
fps_start_time = time.time()
fps_frames = 0
fps = 0

# 이전 오른쪽 손목 관절 위치를 저장하는 변수를 초기화
last_right_wrist = None

# 좌표를 저장할 리스트와 색상값을 초기화
points = []
colors = []

while True:
    
    # 비디오 프레임을 읽어옴
    success, image = cap.read()

    if not success:
        break

    # 입력 이미지에서 포즈를 추정함
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 포즈 결과를 이용하여 오른쪽 손목 관절 위치를 가져옴
    if results.pose_landmarks:

        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # 이전 위치와 비교하여 움직임을 감지함
        if last_right_wrist is not None:
            dx = abs(right_wrist.x - last_right_wrist.x)
            dy = abs(right_wrist.y - last_right_wrist.y)
            distance_moved = (dx ** 2 + dy ** 2) ** 0.5
            if distance_moved > 0.001:
                # 좌표와 색상값을 리스트에 추가함
                points.append((int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0])))
                colors.append((255, 0, 0))
                # 30 프레임 이상이면 리스트에서 가장 오래된 좌표와 색상값을 삭제함
                if len(points) > 30:
                    del points[0]
                    del colors[0]
        last_right_wrist = right_wrist

        # 이전 좌표들을 이용하여 연속적인 점을 그림
        for i, point in enumerate(points):
            cv2.circle(image, point, 5, colors[i], -1)

    # FPS를 계산하고 화면 상단 좌측에 표시함
    fps_frames += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_frames / (time.time() - fps_start_time)
        fps_frames = 0
        fps_start_time = time.time()
    cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 영상 저장하기
    out.write(image)

    # 화면에 표시함
    cv2.imshow("Output", image)

    # ESC 키를 누르면 프로그램을 종료함
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

