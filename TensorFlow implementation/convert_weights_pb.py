# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import yolo_v3
import yolo_v3_tiny
from PIL import Image, ImageDraw

from utils import load_weights, load_coco_names, detections_boxes, freeze_graph
from settings import *



def main():
    if tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
    else:
        model = yolo_v3.yolo_v3

    classes = load_coco_names(class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, size, size, 3], "inputs")

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, output_graph)

if __name__ == '__main__':
    tf.app.run()
