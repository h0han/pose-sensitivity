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
out = cv2.VideoWriter('output_2.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# FPS 계산을 위한 변수를 초기화
fps_start_time = time.time()
fps_frames = 0
fps = 0

# 이전 오른쪽 손목 관절 위치를 저장하는 변수를 초기화
last_right_wrist = None
last_left_wrist = None

# 좌표를 저장할 리스트와 색상값을 초기화
r_points = []
l_points = []
r_colors = []
l_colors = []

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
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # 이전 위치와 비교하여 움직임을 감지함
        if last_right_wrist is not None and last_left_wrist is not None:
            r_dx = abs(right_wrist.x - last_right_wrist.x)
            r_dy = abs(right_wrist.y - last_right_wrist.y)

            l_dx = abs(left_wrist.x - last_left_wrist.x)
            l_dy = abs(left_wrist.y - last_left_wrist.y)

            r_distance_moved = (r_dx ** 2 + r_dy ** 2) ** 0.5
            l_distance_moved = (l_dx ** 2 + l_dy ** 2) ** 0.5

            if r_distance_moved > 0.001 or l_distance_moved > 0.001:
                # 좌표와 색상값을 리스트에 추가함
                r_points.append((int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0])))
                l_points.append((int(left_wrist.x * image.shape[1]), int(left_wrist.y * image.shape[0])))
                
                r_colors.append((255, 0, 0))
                l_colors.append((0, 255, 255))

                # 30 프레임 이상이면 리스트에서 가장 오래된 좌표와 색상값을 삭제함
                if len(r_points) > 30:
                    del r_points[0]
                    del r_colors[0]
                
                if len(l_colors) > 30:    
                    del l_points[0]
                    del l_colors[0]

        last_right_wrist = right_wrist
        last_left_wrist = left_wrist

        # 이전 좌표들을 이용하여 연속적인 점을 그림
        for i, point in enumerate(r_points):
            cv2.circle(image, point, 5, r_colors[i], -1)

        for i, point in enumerate(l_points):
            cv2.circle(image, point, 5, l_colors[i], -1)
            

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

