
import numpy as np
import cv2
from hawk_eyes.face import ArcFace, Landmark, RetinaFace
from hawk_eyes.tracking import BYTETracker
from glob import glob
import os
img_list = os.listdir('data')
retina_face = RetinaFace(model_name='retina_m')
bt = BYTETracker()
for i in range(len(img_list)):
    img_path = os.path.join('data', img_list[i])
    img = cv2.imread(img_path)
    fboxes, kpss = retina_face.detect(img)
    box = fboxes[0].astype(int)
    face = img[box[1]:box[3],box[0]:box[2]]
    
    # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 1)
    cv2.imshow(img_list[i], img)
    cv2.imshow('face' + img_list[i], face)
    cv2.imwrite(img_path, face)
    cv2.waitKey(0)
