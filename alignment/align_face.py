from __future__ import print_function
# from typing import final

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from matlab_cp2tform import get_similarity_transform_for_cv2

import face_alignment
from skimage import io
from skimage import img_as_ubyte

def alignment(src_img,src_pts):
    # For 96x112
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014], [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041]]
    crop_size = (96, 112)

    # For 160x160
    # ref_pts = [ [61.4356, 54.6963],[118.5318, 54.6963], [93.5252, 90.7366],[68.5493, 122.3655],[110.7299, 122.3641]]
    # crop_size = (160, 160)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

dataDir = 'your data directory'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 0, 255)
fontScale = 0.3
le_eye_pos = [36, 37, 38, 39, 40, 41]
r_eye_pos = [42, 43, 44, 45, 47, 46]
classid = -1
f = open('output.txt', 'w')
curPer = ''
with open('input.txt', 'r') as ifd:
    for i, line in enumerate(ifd):
        line = line.strip()
        # pos = line.split('/')[-2]
        per = line.split('/')[-2]
        if per != curPer:
            classid += 1
            curPer = per
        imgName = line.split('/')[-1]
        outPath = os.path.join(dataDir, per)
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        input = io.imread(line)
        preds = fa.get_landmarks(input) 
        if preds == None:
            continue
        lmks = preds[0]
        img = img_as_ubyte(input) #cv2.imread(line)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        le_eye_x, le_eye_y = 0.0, 0.0
        r_eye_x, r_eye_y = 0.0, 0.0
        for l_p, r_p in zip(le_eye_pos, r_eye_pos):
            le_eye_x += lmks[l_p][0]
            le_eye_y += lmks[l_p][1]
            r_eye_x += lmks[r_p][0]
            r_eye_y += lmks[r_p][1]
        le_eye_x = int(le_eye_x / len(le_eye_pos))
        le_eye_y = int(le_eye_y/ len(le_eye_pos))
        r_eye_x  = int(r_eye_x / len(r_eye_pos))
        r_eye_y =  int(r_eye_y / len(r_eye_pos))
        nose = (int(lmks[30][0]), int(lmks[30][1]))
        left_mo = (int(lmks[60][0]), int(lmks[60][1]))
        ri_mo = (int(lmks[64][0]), int(lmks[64][1]))
        final_lmks = [(le_eye_x, le_eye_y), (r_eye_x, r_eye_y), nose, left_mo, ri_mo]
        landmark = []
        for lmk in final_lmks:
            landmark.append(lmk[0])
            landmark.append(lmk[1])
        cropped_align = alignment(img,landmark)
        img_path = os.path.join(outPath, imgName)
        print('out: {}'.format(img_path))
        f.write('{},{}\n'.format(img_path, classid))
        cv2.imwrite(img_path, cropped_align)