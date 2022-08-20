## Finetuning ArcFace on masked images (Sec. 4.3 in the paper) 

**Prepare data**:
Download cropped face and masked face datasets. [Google Drive](https://drive.google.com/drive/folders/1Tra0rpZMYiqEZT9gbxL2ECepDOd34R5R)

**How to run**
1. Put your cropped face and masked face data into the same folder. See file `casia_160x160.txt` for reference of these folders.
2. Change `data_dir` variable with your directory where contains two above folders in `train_mask_face.py` script.
3. Run the following command:
```
python train_mask_face.py
```

**References**
1. [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
    
