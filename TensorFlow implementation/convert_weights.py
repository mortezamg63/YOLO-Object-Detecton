# -*- coding: utf-8 -*-

import tensorflow as tf

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, load_weights
from settings import *


def main():
    if tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
    else:
        model = yolo_v3.yolo_v3

    classes = load_coco_names(class_names)

    # placeholder for detector inputs
    # any size > 320 will work here
    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes),data_format=data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), weights_file)

    saver = tf.train.Saver(tf.global_variables(scope='detector'))

    with tf.Session() as sess:
        sess.run(load_ops)

        save_path = saver.save(sess, save_path=ckpt_file)
        print('Model saved in path: {}'.format(save_path))


if __name__ == '__main__':
    main()
