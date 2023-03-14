import json

# JSON 파일 경로
json_file_path = "sensitivity.json"

# JSON 파일 열기
with open(json_file_path, "r") as f:
    data = json.load(f)

# key, value 출력하기
for key, value in data.items():
    print(key, value)
