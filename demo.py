import cv2
import numpy as np
import time
from hawk_eyes.face import RetinaFace, ArcFace, Landmark
from hawk_eyes.tracking import BYTETracker
from collections import defaultdict
import threading
import argparse
import math
import os
# from flask import Flask, request  # import class FastAPI() từ thư viện fastapi
# from flask import Flask, render_template, Response
import base64
import cv2

retina = RetinaFace(model_name='retina_s')
arc_face = ArcFace(model_name='arcface_s')
bt = BYTETracker()
landmark = Landmark()

parser = argparse.ArgumentParser(description='Chooose option')
parser.add_argument('-d', '--dataset', type=str, default="data")
parser.add_argument('-a', '--cosin', type=bool, default=True)
# parser.add_argument('-l', '--local', type=bool, default=True)
args = parser.parse_args()

database_emb = {
    'userID': [],
    'embs': []
}
data = args.dataset
img_data_list = os.listdir(data)
for i in range(len(img_data_list)):
    img_path = os.path.join(data, img_data_list[i])
    img = cv2.imread(img_path)
    fbox, kpss = retina.detect(img)
    tbox, tids = bt.predict(img, fbox)
    print(kpss[0])
    # face = img[box[1]:box[3],box[0]:box[2]]
    emb = arc_face.get(img, kpss[0])
    database_emb['embs'].append(emb)
    database_emb['userID'].append(img_data_list[i][:-4])
print('Extract feature on databse done!')
# img = cv2.imread('path/to/image.png')
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    bboxes, kpss = retina.detect(frame)
    # bboxes, kpss = retina.detect(img)
    for box,kps in zip(bboxes, kpss):
        box = box[:4].astype(int)
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,255,0),thickness=2)
        emb = arc_face.get(img, kps)
        print(emb)
    
        land = landmark.get(frame, box)
        angle = landmark.get_face_angle(frame, land, False)[1]
        
        if abs(angle) < 15:
            
        #     localtime = time.asctime(time.localtime(time.time()))
        #     # times_check.append(localtime)
            # emb = arc_face.get(frame, kpss)
            emb = arc_face.get(img, kps)
            # print(emb)
            dis = np.linalg.norm(database_emb['embs'] - emb, axis=1)
            if (min(dis) < 24):
                    # if (max(dis_cosin) > 0.01):
                idx = np.argmin(dis)
                # idx = np.argmax(dis_cosin)
                # if (time.time() - t) > 
                print("Name: " + database_emb['userID'][idx].split('_')[0] + " --- MSV: " + database_emb['userID'][idx].split('_')[1] + " --- Time: " + time_check)
                # name.append(database_emb['userID'][idx].split('_')[0])
                # msv.append(database_emb['userID'][idx].split('_')[1])
                # time_checkin.append(str(time_check))
                cv2.putText(frame, str(database_emb['userID'][idx]), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Stranger")
                cv2.putText(
                    frame, "Stranger", (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # embs.append(emb)
            # ids.append(tid)
            # times_check.append(localtime)
        # for box in bboxes:
    
    
    cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (0,255,0),thickness=2)
    
    cv2.imshow('image', frame)
    cv2.waitKey(1)