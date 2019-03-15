input_img = 'dog.jpg'
output_img = ''

input_video = 'video.avi'
output_video = ''

class_names = 'coco.names'
weights_file = 'yolov3.weights'
data_format = 'NHWC'  #Data format: NCHW (gpu only) / NHWC'
ckpt_file = './saved_model/model.ckpt'
frozen_model = 'frozen_darknet_yolov3_model.pb'
size = 416
tiny = False
conf_threshold = 0.5
iou_threshold = 0.4
gpu_memory_fraction = 1.0

'''-------- converting weights to use in TensorFlow ----'''
output_graph = 'frozen_darknet_yolov3-tiny_model.pb'