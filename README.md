# YOLO-Object-Detector
The implementation is based on TensorFlow and OpenCV

In OpenCV implementation, you do not need to convert the weights of DarkNet yolo model because OpenCV can work with weights directly. But in TensorFlow implementation  DarkNet weights must be converted to ckpt or pb files - the file formats that TensorFlow uses to  store and load models. There are two files whose names show which one is considered for converting to which file format.

For more information and downloading YOLO Weights see [Darknet project website](https://pjreddie.com/darknet/yolo/).

There is a main.py in each folder that is the start point of runing the programs. Also, the initial configurations must be done in settings.py. After appropriate change of values in settings.py, you can run the code in main.py to see outputs. In addition, the objects are cropped and stored in directories whose name is extracted_regions or extracted_objects. Some samples of output are shown in following pictures.

<p align="center">
<img src="https://user-images.githubusercontent.com/15813546/54441223-89b87100-4751-11e9-8a7e-a4d75566d025.png">
</p>
