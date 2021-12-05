import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_patch_location(index, sc_max, face_model, level=4):
    patch_size = 16
    if face_model == 'arcface':
        patch_size = int(sc_max / level)
        row=int(index/level)
        col=int(index%level)
    elif face_model == 'cosface':
        row=int(index/7)
        col=int(index%6)
    else:
        print('No face model found!!')
        exit(0)
    row_location = (row * patch_size, (row + 1) * patch_size) 
    col_location = (col * patch_size, (col + 1) * patch_size)
    return row_location, col_location, row, col

def combine_img_heatmap(img, heatmap):
    cmap = plt.get_cmap('jet') # colormap for the heatmap
    heatmap = heatmap - np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap =  cmap(heatmap)
    if np.max(heatmap) < 255.:
        heatmap *= 255
    
    fg = Image.fromarray(heatmap.astype('uint8')).convert('RGBA')
    img = img.convert('RGBA')
    outIm = Image.blend(img,fg,alpha=0.5)
    return outIm