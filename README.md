# Football-Object-Detection

## Description
This project follows the tutorial by [Code In a Jiffy](https://www.youtube.com/@codeinajiffy) in their video [Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python](https://youtu.be/neBZ6huolkg?si=Efj-YhWzZoyq3m3p). The goal is to build a football analysis system with YOLO and understand more on how to use OpenCV and how to utilize the data obtained. I also attempted to fine-tune the model using data from [here] (https://universe.roboflow.com/alina-wang-pa0sr/fball). However, since most of the data were specifically just for football rather than the ball in a official match, it had low impact on fine-tuning the result.

## Future Improvements
The estimation of the distance from camera to the field is calculated specifically for the video. If other videos are used, it could get inaccurate results. This could be further improved by replacing it with a Depth Estimation model to help calculate the depth estimation rather than using cv2.getPerspectiveTransform.
In addition, due to hardware limitation, I chose a smaller YOLO 5 model to use. Therefore, the ball was not always detected correctly and that the fine-tuning attempt was a fail, further improvements on finding a better dataset to fine-tune the model would help increase the accuracy of detecting the ball.
