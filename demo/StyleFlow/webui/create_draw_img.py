import streamlit as st

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide"
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="DeepFace-EMD web demo",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)
import sys
sys.path.insert(0, "../")

from options.test_options import TestOptions

import numpy as np

from utils import Build_model
import torch
import torch.nn
from module.flow import cnf
import os
import tensorflow as tf
import pickle
import copy
import cv2

""" Welcome to DeepFace-EMD WebUI demo
"""

DATA_ROOT = "../data"
HASH_FUNCS = {tf.Session: id,
              torch.nn.Module: id,
              Build_model: lambda _ : None,
              torch.Tensor: lambda x: x.cpu().numpy()}

# @st.cache(hash_funcs=HASH_FUNCS)
def init_model():
    # Open a new TensorFlow session.
    #config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    
    opt = TestOptions().parse()
    with session.as_default():
        model = Build_model(opt)
        w_avg = model.Gs.get_var('dlatent_avg')

    prior = cnf(512, '512-512-512-512-512', 17, 1)
    prior.load_state_dict(torch.load('../flow_weight/modellarge10k.pt'))
    prior.eval()

    return session, model, w_avg, prior.cpu()

@torch.no_grad()
def generate_image(session, model, w):
    with session.as_default():
        img = model.generate_im_from_w_space(w)[0].copy()
    return img

def main():
    session, model, w_avg, flow_model = init_model()
    raw_w = pickle.load(open(os.path.join(DATA_ROOT, "sg2latents.pickle"), "rb"))
    all_w = np.array(raw_w['Latent'])
    f = open('ffhd_1000.txt', 'w')
    for i, w in enumerate(all_w):
        img_raw = generate_image(session, model, w)
        im_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        path = os.path.join('img_data/ffhd/', '{}'.format(i))
        if not os.path.exists(path):
            os.makedirs(path)
        img_name = os.path.join(path, '{}.jpg'.format(i)) #'img_data/ffhd/{}.jpg'.format(i)
        f.write('{},{}\n'.format(img_name, i))
        print('{}:{}'.format(i, img_name))
        cv2.imwrite(img_name, im_rgb)
    f.close()
    print('Done')

if __name__ == '__main__':
    main()

