import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_datadir(img_path):
    parts = img_path.split('/')
    del parts[-1]
    del parts[-1]
    path = '/'+ parts[0]
    del parts[0]
    for p in parts:
        path = os.path.join(path, p)
    return path

class FaceDataset(Dataset):
    def __init__(self, img_file, data_dir='', lmk_file=None, fm='arcface', resize=False, level=4, size=128):
        self.image_paths = []
        self.image_labels = []
        self.index = {}
        self.resize = resize
        self.lmk_file = lmk_file
        self.lmks = []
        self.fm = fm
        if fm == 'cosface':
            self. transform = transforms.Compose([
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
            ])
        elif fm == 'facenet':
            self. transform = transforms.Compose([
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Resize((160,160))
            ])

        with open(img_file, 'r') as ifd:
            for i, line in enumerate(ifd):
                parts = line.strip().split(',')
                img_path = os.path.join(data_dir, parts[0])
                self.index[img_path] = i
                label = int(parts[1])
                self.image_paths.append(img_path)
                self.image_labels.append(label)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        classid = self.image_labels[idx]
        if self.fm == 'arcface':
            img = cv2.imread(img_path, 0)
            if self.resize:
                img = cv2.resize(img, (128,128))
            img = img.reshape((128,128,1))
            img = img.transpose((2, 0, 1))
            img = img.astype(np.float32, copy=False)
            img -= 127.5
            img /= 127.5
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            
        label = np.zeros((1,1), np.float32)
        label[0,0] = classid

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        
        return (img, torch.from_numpy(label[:,0]).long())


def get_face_dataloader(batch_size, data_dir='', folder='', fm='arcface', num_workers=4, level=4, size=128):
    # if folder == 'data_small':
    #     filedir = 'data_files/small' 
    #     lfw_128_File = os.path.join(filedir, 'lfw_128.txt')
    #     lfw_128_mask_File = os.path.join(filedir, 'lfw_128_mask.txt')
    #     lfw_128_glass_File = os.path.join(filedir, 'lfw_128_glass.txt')
    #     lfw_128_crop_File = os.path.join(filedir, 'lfw_128_crop.txt')

    #     face_dataset = {
    #         'lfw128':FaceDataset(lfw_128_File, data_dir=data_dir,  fm=fm, level=level, size=size, resize=True),
    #         'lfw128_masked':FaceDataset(lfw_128_mask_File, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
    #         'lfw128_glass':FaceDataset(lfw_128_glass_File, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
    #         'lfw128_crop':FaceDataset(lfw_128_crop_File, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
    #     }

    #     data_list = ['lfw128', 'lfw128_masked', 'lfw128_glass', 'lfw128_crop']
    # else:
    filedir = 'img_data/' 
        # lfw_128_File = os.path.join(filedir, 'lfw_128.txt')
        # lfw_128_1680_File = os.path.join(filedir, 'lfw_128x128_1680.txt')
        # lfw_128_mask_File = os.path.join(filedir, 'lfw_128_masked_label.txt')
        # lfw_128_glass_File = os.path.join(filedir, 'lfw_128_glass.txt')
        # lfw_128_crop_File = os.path.join(filedir, 'lfw_128_crop70.txt')
    
        # lfwFile = os.path.join(filedir, 'lfw_96x112.txt')
        # lfw_96_mask = os.path.join(filedir, 'lfw_112x96_masked.txt')
        # lfw_96_glass = os.path.join(filedir, 'lfw_112x96_glass.txt')
        # lfw_96_crop = os.path.join(filedir, 'lfw_112x96_crop70.txt')
    ffhd_file = os.path.join(filedir, 'ffhd_1000.txt')


    face_dataset = {
            # 'lfw128':FaceDataset(lfw_128_File, data_dir=data_dir,  fm=fm, level=level, size=size, resize=True),
            # 'lfw128_1680':FaceDataset(lfw_128_1680_File, data_dir=data_dir,  fm=fm, level=level, size=size, resize=True),
            # 'lfw128_masked':FaceDataset(lfw_128_mask_File, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
            # 'lfw128_glass':FaceDataset(lfw_128_glass_File, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
            # 'lfw128_crop':FaceDataset(lfw_128_crop_File, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
            
            # 'lfw':FaceDataset(lfwFile, fm=fm),
            # 'lfw96_mask':FaceDataset(lfw_96_mask, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
            # 'lfw96_glass':FaceDataset(lfw_96_glass, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
            # 'lfw96_crop':FaceDataset(lfw_96_crop, data_dir=data_dir, fm=fm, level=level, size=size, resize=True),
        'ffhd':FaceDataset(ffhd_file, data_dir=data_dir,  fm=fm, level=level, size=size, resize=True),
    }

        # data_list = ['lfw',  'lfw128', 'lfw128_1680', 'lfw128_masked', 'lfw128_glass', 'lfw128_crop', 'lfw96_mask', 'lfw96_glass', 'lfw96_crop']
    data_list = ['ffhd']

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False, num_workers=num_workers) 
        for x in data_list 
    }

    return face_dataset, dataloaders