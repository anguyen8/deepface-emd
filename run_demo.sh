# python face_demo.py -query /home/hai/datasets/calfw/calfw_aligned_160/Winona_Ryder/Winona_Ryder_0003.jpg -sunglass 1 -fm arcface  
# python face_demo.py -query /home/hai/workspace/lfw-align-128/Winona_Ryder/Winona_Ryder_0018.jpg -sunglass 0 -fm arcface  -gallery lfw -method apc
python face_demo.py -query /home/hai/workspace/lfw-align-128/Heather_Mills/Heather_Mills_0002.jpg -sunglass 1 -fm arcface  -gallery lfw -method apc -data_folder data_small -query_person Heather_Mills
# python face_demo.py -query /home/hai/workspace/lfw-align-128/Winona_Ryder/Winona_Ryder_0020.jpg -crop 1 -fm arcface  -gallery lfw -method apc -data_folder data_small
# pudb3 face_demo.py -query /home/hai/workspace/lfw-align-128/Winona_Ryder/Winona_Ryder_0022.jpg -sunglass 1 -fm arcface  -gallery lfw