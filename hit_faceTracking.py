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
from flask import Flask, request  # import class FastAPI() từ thư viện fastapi
from flask import Flask, render_template, Response
import base64
import cv2
import requests

# app = Flask(__name__)


def def_value():
    return "_"

parser = argparse.ArgumentParser(description='Chooose option')
parser.add_argument('-d', '--dataset', type=str, default="data")
parser.add_argument('-a', '--cosin', type=bool, default=True)
# parser.add_argument('-l', '--local', type=bool, default=True)
args = parser.parse_args()


retina_face = RetinaFace(model_name='retina_s')
arc_face = ArcFace(model_name='arcface_s')
bt = BYTETracker()
landmark = Landmark()

recog_data={}
database_emb = {
    'userID': [],
    'embs': []
}

msvs={'ids': []}
data = args.dataset
img_data_list = os.listdir(data)
for i in range(len(img_data_list)):
    img_path = os.path.join(data, img_data_list[i])
    img = cv2.imread(img_path)
    fbox, kpss = retina_face.detect(img)
    tbox, tids = bt.predict(img, fbox)
    print(kpss[0])
    # face = img[box[1]:box[3],box[0]:box[2]]
    emb = arc_face.get(img, kpss[0])
    database_emb['embs'].append(emb)
    database_emb['userID'].append(img_data_list[i][:-4])
print('Extract feature on databse done!')

track_name = defaultdict(def_value)
track_emb = {}
current_tracking = {}
name_idx = 0

cap = cv2.VideoCapture(0)
name_id = []

def recog():
    global track_emb, track_name, current_tracking, recog_data, msvs
    ret, _ = cap.read()
    # t = time.time()
    prev_frame_time = 0
    new_frame_time = 0
    while ret:
        new_frame_time = time.time()
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        fboxes, kpss = retina_face.detect(frame)
        tboxes, tids = bt.predict(frame, fboxes)
        tkpss = [None]*len(fboxes)
        for i in range(len(tboxes)):
            min_d = 9e5
            tb = tboxes[i]
            for j in range(len(fboxes)):
                fb = fboxes[j]
                d = abs(tb[0]-fb[0])+abs(tb[1]-fb[1]) + \
                    abs(tb[2]-fb[2])+abs(tb[3]-fb[3])
                if d < min_d:
                    min_d = d
                    tkpss[i] = kpss[j]
        embs = []
        ids = []
        times_check = []
        for tid, tbox, tkps in zip(tids, tboxes, tkpss):
            # print(localtime)
            box = tbox[:4].astype(int)
            land = landmark.get(frame, tbox)
            angle = landmark.get_face_angle(frame, land, False)[1]
            if abs(angle) < 15:
                localtime = time.asctime(time.localtime(time.time()))
                emb = arc_face.get(frame, tkps)
                embs.append(emb)
                ids.append(tid)
                times_check.append(localtime)
            draw_fancy_box(frame, (box[0], box[1]), (box[2], box[3]), (127, 255, 255), 2, 10, 20)

        current_tracking = {'track_id': ids, 'embs': embs, 'times': times_check}
        
        
        time_checkin = []
        for idt, emb, tbox, time_check in zip(current_tracking['track_id'], current_tracking['embs'], tboxes, current_tracking['times']):
            box = tbox[:4].astype(int)
            dis = np.linalg.norm(database_emb['embs'] - emb, axis=1)
            # dis_cosin = np.dot(database_emb['embs'], emb) / (np.linalg.norm(database_emb['embs']) * np.linalg.norm(emb))
            
            
            t = time.time()
            if (min(dis) < 25):
            # if (max(dis_cosin) > 0.01):
                idx = np.argmin(dis)
                # idx = np.argmax(dis_cosin)
                # if (time.time() - t) > 
                print("Name: " + database_emb['userID'][idx].split('_')[0] + " --- MSV: " + database_emb['userID'][idx].split('_')[1] + " --- Time: " + time_check)
                # name.append(database_emb['userID'][idx].split('_')[0])
                # msv.append(database_emb['userID'][idx].split('_')[1])
                name_id.append(database_emb['userID'][idx])
                time_checkin.append(str(time_check))

                cv2.putText(frame, str(database_emb['userID'][idx]), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Stranger")
                cv2.putText(frame, "Stranger", (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
        recog_data={'name_id': list(set(name_id))}
        print("==============================")
        # time.sleep(0.5)
        
        # cv2.putText(frame, "aaa",(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 1)
        # if args.local != True:
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # draw_fancy_box(frame, (100, 100), (350, 350), (0, 255, 255), 2, 10, 20)
        # for idx in range(len(recog_data['msv'])):
        #     if idx >1 and (recog_data['msv'][idx-1] == recog_data['msv'][idx]):
        #         del recog_data['msv'][idx-1]
        #         del recog_data['name'][idx-1]
        # recog_data = set()
        # print(recog_data)
        # recog_data['msv'] = set(recog_data['msv'])
        # recog_data['name'] = set(recog_data['name'])

        print(recog_data)
        # for i in range(len(recog_data['name_id'])):
        #     msvs['ids'].append(recog_data['name_id'][i].split('_')[1])
        # print(msvs)
        cv2.putText(frame, 'fps: {}'.format(int(fps)),  (frame.shape[1]-150, frame.shape[0]-100), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2 )
        cv2.imshow('qwe', frame)

        # msvs = recog_data['msv']
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        
        # else:
        # retval, buffer = cv2.imencode('.jpg', frame)
        # frame1 = buffer.tobytes()
        # yield (b'--frame\r\n'
        #     b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')


def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


if __name__ == '__main__':
    
    recog()
    url = 'http://192.168.1.7:8080/sang-al'
    reponse = requests.post(url=url, json=recog_data) 
    dict_json = reponse.json()
    print(dict_json)
    # @app.route('/', methods=['GET'])
    # def video_feed():
    #     return Response(recog(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # print("App run!")
    # app.run(debug=False, host='127.0.0.1', threaded=False)
    
    # print(current_tracking['track_id'])
    # print(current_tracking['times'])