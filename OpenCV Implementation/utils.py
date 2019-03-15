import cv2
import numpy as np
from settings import *

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    x, y, x_plus_w, y_plus_h = int(x), int(y), int(x_plus_w), int(y_plus_h)
    label = str(classes[class_id])

    color = COLORS[class_id]
    # cv2.imwrite('extracted_objects/')

    cv2.rectangle(img, (x,y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detection(img, net):
    
    width = img.shape[1]
    height = img.shape[0]

    scale = 0.00392          

    # create input blob
    blob = cv2.dnn.blobFromImage(img, scale,  (416, 416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    #initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5:
                center_x = int(detection[0]*width)                
                center_y = int(detection[1]*height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y -h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x,y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, indices

    # # go through the detections remaining
    # # after nms and draw bounding box
    # rois = []
    # class_of_
    # for i in indices:
    #     i = i[0]
    #     for j, v in enumerate(boxes[i]):
    #         if v<0:
    #             boxes[i][j] = 0

    #     box = boxes[i]
    #     x = round(box[0])
    #     y = round(box[1])
    #     w = round(box[2])
    #     h = round(box[3])
    #     rois.append([x,y,w+x,h+y])
    #     draw_bounding_box(img, class_ids[i], confidences[i], x,y, x+w, y+h)#round(x), round(y), round(x+w), round(y+h))
    
    # return img, rois