# Football-Object-Detection

## Description
This project follows the tutorial by [Code In a Jiffy](https://www.youtube.com/@codeinajiffy) in their video [Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python](https://youtu.be/neBZ6huolkg?si=Efj-YhWzZoyq3m3p). The goal is to build a football analysis system with YOLO and understand more on how to use OpenCV and how to utilize the data obtained.

##Future Improvements
Due to my current hardware limitation, I used a less accurate model compared to the current model used in the tutorial. This can be further improved by fine-tuning my model to be able to track players and the ball more accurately.
In addition, the estimation of the distance from camera to the field is calculated specifically for the video. If other videos are used, it could get inaccurate results. This could be further improved by replacing it with a Depth Estimation model to help calculate the depth estimation rather than using cv2.getPerspectiveTransform.
