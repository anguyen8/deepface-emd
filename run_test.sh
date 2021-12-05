# normal lfw
# python test_face.py -method apc -fm arcface -d lfw -a 0.3 -data_folder data_small

# ood lfw: mask, sunglass, or crop
# python test_face.py -method apc -fm arcface -d lfw -a 0.7 -crop 1 -data_folder data -l 8
python test_face.py -method apc -fm facenet -d lfw -a 0.7 -crop 1 -data_folder data_small -l 8
