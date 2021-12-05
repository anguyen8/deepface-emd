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

## How to run
- Run testing LFW images

```
bash run_test.sh
```

- Run demo 
```
bash run_test.sh
```

- Run visualization with two images

```
python visualize_faces.py -method [methods] -fm [face models] -model_path [model dir] -in1 [1st image] -in2 [2nd image] -weight [1/0: showing weight heatmaps] 
```


