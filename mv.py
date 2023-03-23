import cv2
import numpy as np
import time
import csv

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 저장할 비디오 파일 이름 설정
out = cv2.VideoWriter('mv_output.mp4', fourcc, 30.0, (720, 1280))

# FPS 계산을 위한 변수를 초기화
fps_start_time = time.time()
fps_frames = 0
fps = 0

# CSV 파일에서 좌표 정보를 읽어옴
points = []
with open('right_wrist_coordinates.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 첫 번째 줄을 생략
    for row in reader:
        points.append((float(row[1]) * 1000, float(row[2]) * 1000))

# 연속적인 점으로 움직임을 표현할 평면 생성
plane = np.zeros((720, 1280, 3), np.uint8)
color = (0, 255, 0)
thickness = 3

# FPS 계산을 위한 변수를 초기화
fps_start_time = time.time()
fps_frames = 0
fps = 0

# 이전 위치를 저장하는 변수를 초기화
last_pos = None

# 연속적인 움직임을 표시하기 위한 인덱스 변수를 초기화
index = 0

while index < len(points):

    # 현재 위치
    pos = (int(points[index][0]), int(points[index][1]))

    # 이전 위치가 있는 경우, 원을 그림
    if last_pos is not None:
        # 이전 위치에 원을 그림
        cv2.circle(plane, last_pos, thickness, (0, 255, 0), -1)
        # 현재 위치에 원을 그림
        cv2.circle(plane, pos, thickness, (0, 255, 0), -1)

    # 이전 위치를 현재 위치로 갱신
    last_pos = pos

    # FPS 계산
    fps_frames += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_frames / (time.time() - fps_start_time)
        fps_frames = 0
        fps_start_time = time.time()

    # FPS를 화면에 표시
    # cv2.putText(plane, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 이미지를 출력
    cv2.imshow("Output", plane)

    # 비디오에 이미지를 저장
    out.write(plane)

    # 다음 위치로 인덱스를 이동
    index += 1

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) == 27:
        break


# 'DONE' 이라는 글자를 출력
cv2.putText(plane, 'DONE', (int(plane.shape[1]/2)-50, int(plane.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

# 영상 저장하기
out.write(plane)

# 2초 대기 후, 프로그램 종료
cv2.imshow("Output", plane)
cv2.waitKey(2000)

# 비디오 캡처 객체와 창을 해제함
cv2.destroyAllWindows()
