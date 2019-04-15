import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import copy 
from settings import *
from utils import detection, draw_bounding_box



def main():
    #----------------initialization----------------
    #reading data from settings.py    

    # loading pre-trained model and config file
    t0 = time.time()
    net = cv2.dnn.readNet(wieght_arg, config_arg)
    print("Loaded graph in {:.2f}s".format(time.time()-t0))
    capture = cv2.VideoCapture(input_video)    
    ret, _ = capture.read()
    plt.ion()
    index = 0

    # Is there any frame to read?
    while ret:
        index += 1
        ret, frame = capture.read() 
        img = copy.deepcopy(frame)           
        # applying transformation and apropriate changes to frame to feed the loaded model
        #---- this section is empty in OpenCV ----
        
        t0 = time.time()
        # feeding tensor to loaded model and obtaining the bounding boxes of detected objects
        roi_boxes, roi_confidences, roi_class, roi_indices = detection(frame, net)
        # drawing the boxes around the detected objects and saving the objects simultaneously        
        for i in roi_indices:
            i = i[0]
            for j, v in enumerate(roi_boxes[i]):
                if v<0:
                    roi_boxes[i][j] = 0

            box = roi_boxes[i]
            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])
            
            # #croping and extracting bounding boxes of detected objects in frames, then save them in './extracted_regions/' Directory
            cv2.imwrite('./extracted_objects/frame_'+str(index)+'_obj_'+str(i)+'.jpg', img[y:y+h, x:x+w])

            draw_bounding_box(frame, roi_class[i], roi_confidences[i], x,y, x+w, y+h)
            
        print("Predictions found in {:.2f}s".format(time.time() - t0))
        
        # show results of processing each frame to user
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.pause(0.02)
        plt.show()



if __name__=='__main__':
    main()
