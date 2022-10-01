import numpy as np
import cv2
from hawk_eyes.face import ArcFace, Landmark, RetinaFace
from hawk_eyes.tracking import BYTETracker
from glob import glob
import os
img_list = os.listdir('data')
retina_face = RetinaFace(model_name='retina_m')
arc_face = ArcFace(model_name='arcface_m')
bt = BYTETracker()
landmark = Landmark()

database_emb = {
    'userID': [],
    'embs': []
}

img_data_list = os.listdir('data')
for i in range(len(img_data_list)):
    img_path = os.path.join('data', img_data_list[i])
    img = cv2.imread(img_path)
    fbox, kpss = retina_face.detect(img)
    tbox, tids = bt.predict(img, fbox)
    print(kpss[0])
    # face = img[box[1]:box[3],box[0]:box[2]]    
    emb = arc_face.get(img, kpss[0])
    database_emb['embs'].append(emb)
    database_emb['userID'].append(img_data_list[i][:-4])
    
print(database_emb)