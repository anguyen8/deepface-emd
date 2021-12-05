## DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover’s Distance Improves Out-Of-Distribution Face Identification

`Official Implementation` for the paper "DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover’s Distance Improves Out-Of-Distribution Face Identification" , by Hai Phan and Anh Nguyen.

![](figs/framework.png) 
![](figs/results.png) 

## Requirements
```
Python >= 3.5
Pytorch > 1.0
Opencv >= 3.4.4
pip install tqmd
```

## Download datasets and models

- Download LFW, out-of-distribution of LFW, and models: [Google Drive](https://drive.google.com/drive/folders/1hoyO7IWaIx2Km-pe4-Sn2D_uTFNLC7Ph)

- Create the following folders:

```
mkdir data
mkdir pretrained
```

- Extract LFW datasets (e.g. `lfw_crop_96x112.tar.gz`) to `data/`
- Copy models (e.g. `resnet18_110.pth`) to `pretrained/` 

## How to run small exmaples
- Run testing LFW images

```
bash run_test.sh
```

- Run demo: The demo gives results of top-5 images of stage 1 and stage 2.
```
bash run_demo.sh
```
## How to run full exmaples

- Change `args.data_folder` to `data` in `.sh` files.

## Run visualization with two images

```
python visualize_faces.py -method [methods] -fm [face models] -model_path [model dir] -in1 [1st image] -in2 [2nd image] -weight [1/0: showing weight heatmaps] 
```

## Modify with your own dataset.
- Create a folder including all persons (folders: name of person) and put it to '/data' 
- Create a `txt` file with format: `[image_path],[label]` of that folder (See [lfw file](data_files/full/lfw_128.txt) for details)
- Modify [face loader](data_loader/facedata_loader.py): Add your `txt` file in function: `get_face_dataloader`. 


