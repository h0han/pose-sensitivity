# Sensitivity
## Features implemented
- Conducting pose estimation on video footage (using MediaPipe)
- Displaying Frames Per Second (FPS)
- Providing data on the number of times a specific joint exceeds the designated threshold (0.001 pixels) within a one-second timeframe [movement per second] (sensitivity)
- Calculating the velocity of a particular joint [pixels per second] (indicating how much it has moved compared to the preceding joint)
- Save coordinates of a particular joint for each frame (.csv format)

## Prerequisites
- A video to be used for demonstration (if you use the CPU, it's better to use shorter videos)
- Put your video path in `cap_video = cv2.VideoCapture('video.mp4')`

## Run
```bash
$ pyenv virtualenv 3.6 <the name of virtualenv>

$ pip3 install requirements.txt

$ python3 sensitivity.py
```

## Demo
<img src="demo/output_sml_rszd.gif" alt="OMG" width="40%"/>
