# YOLO-Object-Detector
Implementation is based on TensorFlow and OpenCV

In OpenCV implementation, you do not need to convert the weights of DarkNet yolo model because OpenCV can work with weights directly. But in TensorFlow implementation  DarkNet weights must be converted to ckpt or pb files - formats that model can be stored and loaded. There are two files whose names show which one is considered for converting to which format.

For more information and downloading YOLO Weights see [Darknet project website](https://pjreddie.com/darknet/yolo/)

There is a main.py in each folder that is the start point of runing the programs. Also, the initial configurations must be done in settings.py. After appropriate change of values in settings.py, you can run the code in main.py to see outputs.

