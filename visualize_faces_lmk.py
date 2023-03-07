import torch
import torch.nn.functional as F
import torch.multiprocessing
import argparse
from PIL import Image, ImageDraw
import json
import numpy as np
from utils.heatmap import combine_img_heatmap

parser = argparse.ArgumentParser(
    description="Visualize DeepFace-EMD"
)

parser.add_argument("-lmk_json_query", type=str, default=None, help="LMK JSON query file",)
parser.add_argument("-lmk_json_right", type=str, default=None, help="LMK JSON right image file",)
parser.add_argument("-fm", type=str, default="arcface",help="face model",)
parser.add_argument("-l", type=int, default=4,help="level of grid size",)
parser.add_argument("-in1", type=str, default='', help="First input image",)
parser.add_argument("-in2", type=str, default='', help="Second input image",)
args = parser.parse_args()

def tuplize(lmk):
    face_landmarks = {}
    for facial_feature in lmk.keys():
         face_landmarks[facial_feature] = []
         for elem in lmk[facial_feature]:
            face_landmarks[facial_feature].append(tuple(elem))
    
    return face_landmarks

def draw_landmarks(face_landmarks, image):
    face_landmarks = tuplize(face_landmarks)
    pil_image = image.copy()
    d = ImageDraw.Draw(pil_image)
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5, fill="red")
    return pil_image

def create_w_from_lmk(lmk_json_path, level=4, size=128, scale=1.0):
    # with open(lmk_file, 'r') as ifd:
    #     for i, json_path in enumerate(ifd):
    f = open(lmk_json_path.strip(), 'r')
    lmk = json.load(f)

    w, w_temp = np.zeros((level, level)), np.zeros((level, level))
    # w[:, 1:3] = 1. / (level * 2.0)
    scale = scale * float(level/size)
    direction = ((-1,0), (0,-1), (0,1), (1,0), (1,1), (-1,-1), (-1, 1), (1,-1))
    for key in lmk.keys():
        for point in lmk[key]:
            row = int(point[1] * scale) 
            col = int(point[0] * scale)
            if row >= level or row < 0 or col >= level or col < 0:
                continue
            w[row, col] += 1

    w = w / w.sum() 
    w = w.reshape((1,-1))
    return w, lmk

def main():
    size = (128, 128)
    level = args.l
    shape = [(0, 0), (size[0], size[1])]
    img_path_1 = args.in1
    img_path_2 = args.in2
    query_img = Image.open(img_path_1).convert('RGB').resize((size[0],size[1]))
    right_img = Image.open(img_path_2).convert('RGB').resize((size[0],size[1]))
    u, lmks = create_w_from_lmk(args.lmk_json_query)
    query_img_lmk = draw_landmarks(lmks, query_img)
    v, lmks = create_w_from_lmk(args.lmk_json_right)
    right_img_lmk = draw_landmarks(lmks, right_img)
    u = torch.from_numpy(u).float()
    v = torch.from_numpy(v).float()
    image = Image.new('RGB', (4 * size[0], size[1]))
    u, v = u.view(level,level), v.view(level,level)
    u, v = u.view(1,1,level,level), v.view(1,1,level,level)
    u = F.interpolate(u,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
    v = F.interpolate(v,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
    q_img_heatmap = combine_img_heatmap(query_img, u.cpu().detach().numpy())
    right_img_heatmap = combine_img_heatmap(right_img, v.cpu().detach().numpy())
    image.paste(q_img_heatmap, ((0, 0)))
    image.paste(right_img_heatmap, ((3*size[0], 0)))
    image.paste(query_img_lmk, ((size[0], 0)))
    image.paste(right_img_lmk, ((2*size[0], 0)))
    imgname = 'results/heatmap/heatmap_face_lmk.jpg'
    
    print('img: {}'.format(imgname))
    image.save(imgname)

if __name__ == '__main__':
    main()

# python visualize_faces_lmk.py -in1 ./test_inputs/Daniel_Radcliffe/Daniel_Radcliffe_0003.jpg -in2 ./test_inputs/Daniel_Radcliffe/Daniel_Radcliffe_0004.jpg -lmk_json_query ./test_inputs/Daniel_Radcliffe_lmk/Daniel_Radcliffe_0003.json -lmk_json_right ./test_inputs/Daniel_Radcliffe_lmk/Daniel_Radcliffe_0004.json
