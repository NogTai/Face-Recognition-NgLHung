import cv2
import numpy as np
import time
from hawk_eyes.face import RetinaFace, ArcFace, Landmark
from hawk_eyes.tracking import BYTETracker
from collections import defaultdict 
import threading
import argparse

from main import def_value

parser = argparse.ArgumentParser(description='Input some parameter')
args = parser.parse_args()
retina_face = RetinaFace(model_name='retina_m')
arc_face = ArcFace(model_name='arcface_m')
bt = BYTETracker()
landmark = Landmark()
gamma = 37.8

def val_value():
    return '_'

recog_data = {
    'time': [],
    'userID': [],
    'emb': [],
    'trackID': [],
    'count_time':[]
}
# Khoi tao dictionary
track_name = defaultdict(def_value)
track_emb = {}
current_tracking = {}
name_idx = 0
cap = cv2.VideoCapture(0)

def recog():
    #Khoi tao bien global
    global track_emb, track_name, current_tracking, recog_data
    ret, _ = cap.read()
    #start time
    t = time.time()
    while ret:
        ret,frame = cap.read()
        frame = cv2.flip(frame, 1)
        #detect face with retinaface
        fboxes, kpss = retina_face.detect(frame)
        #tracking face with arcface
        tboxes, tids = bt.predict(frame, fboxes)
        tkpss = [None]*len(fboxes)
        for i in range(len(tboxes)):
            min_d = 9e5
            tb = tboxes[i]
            for j in range(len(fboxes)):
                fb = fboxes[i]
                d = abs(tb[0] - fb[0]) + abs(tb[1] - fb[1]) + abs(tb[2]- fb[2]) + abs(tb[3] - fb[3])
                if d < min_d:
                    min_d = d
                    tkpss[i] = kpss[j]
        embs = []
        ids = []
        for tid, tbox, tkps in zip(tids, tboxes, tkpss):
            box = tbox[:4].astype(int)
            if tid in recog_data['trackID']:
                idx = recog_data['trackID'].index(tid)
                recog_data['time'][idx] = time.time()
            else:
                land = landmark.get(frame, tbox)
                angle = landmark.get_face_angle(frame, land, False)[1]
                
                if abs(angle) < 15:
                    emb = arc_face.get(frame, tkps)
                    embs.append(emb)
                    ids.append(tid)
        current_tracking = {'frame: ':frame.copy(), 'track_id':ids, 'embs':embs}
       
        for box, tid in zip(tboxes, tids):
            box = box[:4].astype(int)
            st = str(tid)
            if tid in recog_data['trackID']:
                st = recog_data['userID'][recog_data['trackID'].index(tid)]
            cv2.putText(frame, st, (box[0], box[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        for i in range(len(recog_data['userID'])):                
            cv2.putText(frame, '{}: {:0.3f}'.format(recog_data['userID'][i], recog_data['count_time'][i]), (10, 20+i*25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 2  )
        cv2.imshow('qwe', frame)
        cv2.waitKey(1)

def check_emb_in_data():
    global track_emb, recog_data, track_name, current_tracking, name_idx
    while True:
        if 'track_id' not in current_tracking.keys():
            time.sleep(1)
            continue
        time.sleep(0.1)
        for idt, emb in zip(current_tracking['track_id'], current_tracking['embs']):
            if len(recog_data['emb']) == 0:
                print('check2')
                recog_data['time'].append(time.time)
                recog_data['userID'].append('user_{}'.format(name_idx))
                name_idx+=1
                recog_data['emb'].append(emb) 
                recog_data['trackID'].append(idt)
            else:
                distance = np.linalg.norm(recog_data['emb'] - emb, axis=1)
                
                if min(distance) < 21:
                    idx = np.argmin(distance)
                    name = recog_data['userID'][idx]
                    recog_data['time'][idx] = time.time()
                    recog_data['emb'][idx] = emb
                    recog_data['trackID'][idx] = idt
                else:
                    recog_data['userID'].append('user_{}'.format(name_idx))
                    recog_data['time'].append(time.time())
                    name_idx += 1
                    recog_data['emb'].append(emb)
                    recog_data['trackID'].append(idt)
                    recog_data['count_time'].append(0)

def remove_10s():
    global recog_data, track_emb, track_name, current_tracking
    while True:
        t_rm = time.time()
        for i in range(len(recog_data['userID'])):
            if (t_rm - recog_data['time'][i] > 10):
                del recog_data['userID'][i]
                del recog_data['trackID'][i]
                del recog_data['emb'][i]
                del recog_data['time'][i]
                del recog_data['count_time'][i]
                break
        time.sleep(0.5)
threading.Thread(target=check_emb_in_data, args=()).start()
recog()