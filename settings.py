import numpy as np
input_video = 'video1.avi'
image_address = 'dog.jpg'
classes_file_address = 'yolov3.txt'
wieght_arg = 'yolov3.weights'
config_arg = 'yolov3.cfg'

# read class names from text
classes = None
with open(classes_file_address, 'r') as f:
    classes = [line.strip() for line in f.readlines()]  


# generate different colors for bounding boxes of diffrent classes
COLORS = np.random.uniform(0,255, size=(len(classes), 3))