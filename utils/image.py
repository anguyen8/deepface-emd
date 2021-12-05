import torch
import torchvision
import cv2
import numpy  as np
from PIL import Image
from utils.heatmap import get_patch_location
from torchvision import transforms

def preprocessing(img_path, shape, fm='arcface'):
    if fm == 'arcface':
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, shape)
        img = img.reshape((128,128,1))
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32, copy=False)
        img -= 127.5
        img /= 127.5
    elif fm == 'cosface':
        transform = transforms.Compose([
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        return img
    else:
        print('No face model found!!')
        exit(0)
    return torch.from_numpy(img).float()

def draw_grid_img(flow, image_transform, size, fm='arcface', level=4):
    patch_list = []
    weight = flow.sum(-1)
    nums = flow.shape[0]
    # weight=(weight-weight.min())/(weight.max()-weight.min())
    weight=weight/ weight.max()
    for index_grid in range(nums):
        index_patch=torch.argmax(flow[index_grid]).item()
        row_location, col_location, _ , _ = get_patch_location(index_patch, size[0], fm, level=level)
        patch = image_transform[:, row_location[0]:row_location[1], col_location[0]:col_location[1]].cuda()
        patch = patch * weight[index_grid]
        patch_list.append(patch)

    patch_list = torch.stack(patch_list, dim=0)
    if fm == 'arcface':
        grids = torchvision.utils.make_grid(patch_list,nrow=level,padding=0)
    else:
        grids = torchvision.utils.make_grid(patch_list,nrow=7,padding=0)
    grids = grids.permute(1,2,0).cpu().detach().numpy() * 255.0
    grid_img = Image.fromarray(grids.astype('uint8'))
    return grid_img