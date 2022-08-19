## Training Face Mask

**Prepare data**:
Download cropped face and masked face datasets. [Google Drive](https://drive.google.com/drive/folders/1hoyO7IWaIx2Km-pe4-Sn2D_uTFNLC7Ph?usp=sharing)

**How to run**
1. Put your cropped face and masked face data into the same folder. See file `casia_160x160.txt` for reference of these folders.
2. Change `data_dir` variable with your directory where contains two above folders in `train_mask_face.py` script.
3. Run the following command:
```
python train_mask_face.py
```

**References
1. [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
    