# Author: haiphan
# Created: 05 Jan,2022, 11:38 PM CT timezone
# Email: pthai1204@gmail.com
import os
import dlib
import cv2
from mask_utils.aux_functions import *

class FaceTool:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
        # if not os.path.exists(path_to_dlib_model):
        #     download_dlib_model()
        self.predictor = dlib.shape_predictor(path_to_dlib_model)
        self.mask_type = 'random'

    def mask_face(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return mask_image(img, self.detector, self.predictor, mask_type='surgical')

