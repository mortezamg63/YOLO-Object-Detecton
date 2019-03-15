# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2
import matplotlib.pyplot as plt
import yolo_v3
import yolo_v3_tiny
from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image
from settings import *


def main():

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)

    config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False,)
    #----------- Initialization --------------
    # Settings data+ following initializations
    classes = load_coco_names(class_names)
    cap = cv2.VideoCapture('video.avi')
    ret, _ = cap.read()
    plt.ion()
    frame_index = 0

    # defining model
    if frozen_model: #The protobuf file contains the graph definition as well as the weights of the model. 

        t0 = time.time()
        # loading model and related weights
        frozenGraph = load_graph(frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        
        with tf.device("/GPU:0"):
            with tf.Session(graph=frozenGraph, config=config) as sess:
                # Is there any frame to read?
                while ret:
                    frame_index += 1
                    ret, frame = cap.read()
                    # applying transformation and apropriate changes to frame to feed the loaded model
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    img_resized = letter_box_image(img, size, size, 128)
                    img_resized = img_resized.astype(np.float32)                    
                    t0 = time.time()
                    # feeding tensor to loaded model
                    detected_boxes = sess.run(boxes, feed_dict={inputs: [img_resized]})
                    #obtaining the bounding boxes of detected objects
                    filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=conf_threshold, iou_threshold=iou_threshold)
                    print("Predictions found in {:.2f}s".format(time.time() - t0))

                    #croping and extracting bounding boxes of detected objects in frame
                    rois = draw_boxes(filtered_boxes, img, classes, (size, size), True)
                    if len(rois)>0:
                        for i in range(len(rois)):
                            # saving the cropped images in Hard Disk = './extracted_regions/' Directory
                            rois[i].save('./extracted_regions/frame'+str(frame_index)+'_ExtObj_'+str(i)+'.jpg')
                    plt.imshow(np.array(img))
                    plt.pause(0.02)
                    plt.show()

    else:
        # using ckpt file for loading the model weights
        #----------- Initialization --------------
        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
        cap = cv2.VideoCapture('video.avi')
        ret, _ = cap.read()
        plt.ion()
        t0 = time.time()
        frame_index = 0

        # loading model and related weights
        if tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), size, data_format)        
        t0 = time.time()
        saver.restore(sess, ckpt_file)
        print('Model restored in {:.2f}s'.format(time.time()-t0))
        
        with tf.Session(config=config) as sess:                                    
            # is there any frame to read?
            while ret:
                frame_index += 1
                ret, frame = cap.read()
                # applying transformation and apropriate changes to frame to feed the loaded model
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_resized = letter_box_image(img, size, size, 128)
                img_resized = img_resized.astype(np.float32)                    
                t0 = time.time()
                # feeding tensor to loaded model
                detected_boxes = sess.run(boxes, feed_dict={inputs: [img_resized]})
                #obtaining the bounding boxes of detected objects
                filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=conf_threshold, iou_threshold=iou_threshold)
                print("Predictions found in {:.2f}s".format(time.time() - t0))
                #croping and extracting bounding boxes of detected objects
                rois = draw_boxes(filtered_boxes, img, classes, (size, size), True)

                if len(rois)>0:
                        for i in range(len(rois)):
                            # saving the cropped images in Hard Disk = './extracted_regions/' Directory
                            rois[i].save('./extracted_regions/frame'+str(frame_index)+'_ExtObj_'+str(i)+'.jpg')

                
                plt.imshow(np.array(img))
                plt.pause(0.02)
                plt.show()
        


if __name__ == '__main__':
    main()
