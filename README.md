## DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover’s Distance Improves Out-Of-Distribution Face Identification
_**tldr:** Improving OOD face identification (e.g. on masked faces) by harnessing pre-trained face models for patch-wise similarity-based re-ranking. Accuracy improved without any further training and without synthetic or augmented data._


**Official Implementation** for the paper [DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover’s Distance Improves Out-Of-Distribution Face Identification](https://arxiv.org/abs/2112.04016) (2022) by Hai Phan and Anh Nguyen.

:star2: **Online web demo**: https://aub.ie/face ([video](https://www.youtube.com/watch?v=rESuo-su1bE))

If you use this software, please consider citing:

    @inproceedings{hai2022deepface,
      title={DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover’s Distance Improves Out-Of-Distribution Face Identification},
      author={Hai Phan, Anh Nguyen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2022}
    }

![](figs/framework.png) 
![](figs/results.png) 


## 1. Requirements
```
Python >= 3.5
Pytorch > 1.0
Opencv >= 3.4.4
pip install tqdm
```

## 2. Download datasets and pretrained models

1. Download LFW, _out-of-distribution_ (OOD) LFW test sets, and pretrained models: [Google Drive](https://drive.google.com/drive/folders/1hoyO7IWaIx2Km-pe4-Sn2D_uTFNLC7Ph?usp=sharing)

2. Create the following folders:

```
mkdir data
mkdir pretrained
```

3. Extract LFW datasets (e.g. `lfw_crop_96x112.tar.gz`) to `data/`
4. Copy models (e.g. `resnet18_110.pth`) to `pretrained/` 

## 3. How to run 
 
### 3.1 Run examples
- Run testing LFW images
  +  `-mask`, `-sunglass`, `-crop`: flags for using corresponding OOD query images (i.e., faces with masks or sunglasses or randomly-cropped images).
  ```
  bash run_test.sh
  ```

- Run demo: The demo gives results of top-5 images of stage 1 and stage 2 (including flow visualization of EMD).
  + `-mask`: image retrieval using a masked-face query image given a gallery of normal LFW images.
  + `-sunglass` and `-crop`: similar to the setup of `-mask`.
  + The results will be saved in the [results/demo](https://github.com/anguyen8/deepface-emd/tree/main/results/demo) directory.
  ```
  bash run_demo.sh
  ```
- Run retrieval using the full LFW gallery
  + Set the argument `args.data_folder` to `data` in `.sh` files.

### 3.2 Reproduce results
- Make sure `lfw-align-128` and `lfw-align-128-crop70` dataset in `data/` directory (e.g. `data/lfw-align-128-crop70`), ArcFace [2] model `resnet18_110.pth` in `pretrained/` directory (e.g. `pretrained/resnet18_110.pth`). Run the following commands to reproduce the Table 1 results in our paper.
  + Arguments: 
      + Methods can be `apc`, `uniform`, or `sc`
      + `-l`: 4 or 8 for `4x4` and `8x8` respectively.
      + `-a`: alpha parameter mentioned in the paper.

  + Normal LFW with 1680 classes:
  ```
  python test_face.py -method apc -fm arcface -d lfw_1680 -a -1 -data_folder data -l 4
  ```
  + LFW-crop:
  ```
  python test_face.py -method apc -fm arcface -d lfw -a 0.7 -data_folder data -l 4 -crop 
  ```
  * Note: The full LFW dataset have 5,749 people for a total of 13,233 images; however, only 1,680 people have two or more images (See [LFW](http://vis-www.cs.umass.edu/lfw/) for details). However, in our normal LFW dataset, the identical images will not be considered in face identification. So, the difference between `lfw` and `lfw_1680` is that the `lfw` setup uses the full LFW (including people with a single image) but the `lfw_1680` uses only 1,680 people who have two or more images.     

- For other OOD datasets, run the following command:
  + LFW-mask:
  ```
  python test_face.py -method apc -fm arcface -d lfw -a 0.7 -data_folder data -l 4 -mask 
  ```
  + LFW-sunglass:
  ```
  python test_face.py -method apc -fm arcface -d lfw -a 0.7 -data_folder data -l 4 -sunglass 
  ```

### 3.3 Run visualization with two images

```
python visualize_faces.py -method [methods] -fm [face models] -model_path [model dir] -in1 [1st image] -in2 [2nd image] -weight [1/0: showing weight heatmaps] 
```
The results are in [results/flow](https://github.com/anguyen8/deepface-emd/tree/main/results/flows) and [results/heatmap](https://github.com/anguyen8/deepface-emd/tree/main/results/heatmap) (if `-weight` flag is on).

![](results/flows/daniel_radcliffe_flow_face.jpg) 
![](results/heatmap/daniel_radcliffe_heatmap_face.jpg) 

### 3.4 Use your own images
1. Facial alignment. See [align_face.py](alignment/align_face.py) for details.
- Install [face_alignment](https://github.com/1adrianb/face-alignment) to extract landmarks. 
```
pip install scikit-image
pip install face-alignment
```
- For making face alignment with size of `160x160` for Arcface (`128x128`) and FaceNet (`160x160`), the reference points are as follow (see function `alignment` in [align_face.py](alignment/align_face.py)).
```python
ref_pts = [ [61.4356, 54.6963],[118.5318, 54.6963], [93.5252, 90.7366],[68.5493, 122.3655],[110.7299, 122.3641]]
crop_size = (160, 160)
```

2. Create a folder including all persons (folders: name of person) and put it to '/data' 
3. Create a `txt` file with format: `[image_path],[label]` of that folder (See [lfw file](data_files/full/lfw_128.txt) for details)
4. Modify [face loader](data_loader/facedata_loader.py): Add your `txt` file in function: `get_face_dataloader`. 

### 3.5. Training on masked images
See this [folder](https://github.com/anguyen8/deepface-emd/tree/main/arcface-pytorch) for the data and instructions for train on masked images following the experiments in [Sec. 4.3 Re-ranking rivals finetuning on masked images](https://arxiv.org/abs/2112.04016) in the paper.

## 4. License
MIT

## 5. References
1. W. Zhao, Y. Rao, Z. Wang, J. Lu, Zhou. Towards interpretable deep metric learning with structural matching, ICCV 2021 [DIML](https://github.com/wl-zhao/DIML)
2. J.  Deng,   J. Guo,   X. Niannan,   and   StefanosZafeiriou.   Arcface:  Additive angular margin loss for deepface recognition, CVPR 2019 [Arcface Pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
3. H.  Wang,  Y. Wang,  Z. Zhou,  X. Ji,  DihongGong,  J. Zhou,  Z. Li,  W. Liu.   Cosface: Large margin cosine loss for deep face recognition, CVPR 2018 [CosFace Pytorch](https://github.com/MuggleWang/CosFace_pytorch)
4. F. Schroff,  D. Kalenichenko, J. Philbin. Facenet: A unified embedding for face recognition and clustering. CVPR 2015 [FaceNet Pytorch](https://github.com/timesler/facenet-pytorch)
5. L. Weiyang, W. Yandong, Y. Zhiding, L. Ming, R. Bhiksha, S. Le. SphereFace: Deep Hypersphere Embedding for Face Recognition, CVPR 2017 [sphereface](https://github.com/wy1iu/sphereface), [sphereface pytorch](https://github.com/clcarwin/sphereface_pytorch)
6. Chi Zhang, Yujun Cai, Guosheng Lin, Chunhua Shen. Deepemd: Differentiable earth mover’s distance for few-shotlearning, CVPR 2020 [paper](https://arxiv.org/pdf/2003.06777.pdf)
