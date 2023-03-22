import streamlit as st

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide"
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="DeepFace-EMD web demo",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)
st.markdown('<style>.css-rncmk8 > *{margin-top:0; margin-right:0;} .css-18e3th9 {padding-top:0;} </style>', unsafe_allow_html=True)

import sys
sys.path.insert(0, "../")

from options.test_options import TestOptions

import numpy as np
import random
from utils import Build_model
import torch
import torch.nn
from module.flow import cnf
import os
import cv2
import tensorflow as tf
import pickle
import copy
from torchvision import transforms
from PIL import Image, ImageDraw
from face_models.resnet import *
from emd_utils.emd import emd_similarity
from emd_utils.image import preprocessing, draw_grid_img, preprocessingImage
from emd_utils.extract_features import extract_embedding
from data_loader.facedata_loader import get_face_dataloader
from facetool import  FaceTool
torch.multiprocessing.set_sharing_strategy('file_system')

# import tornado.httpserver
# import tornado.ioloop
# import tornado.web


""" 
## DeepFace-EMD demo
A 2-stage face identification system from [DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover’s Distance Improves Out-Of-Distribution Face Identification](https://arxiv.org/abs/2112.04016). Hai Phan & Anh Nguyen (2021). 
For each query, we search for matching images from a gallery of 1,000 [StyleGAN-v2-generated images](https://github.com/RameenAbdal/StyleFlow) (decoded from 1,000 FFHQ images).
**Stage 1** column: top-5 shortlisted candidates after Stage 1 (sorting using cosine distance at the image-embedding level of [CASIA](https://arxiv.org/abs/1411.7923)-trained ArcFace). 
**Stage 2** column: top-5 matching faces after Stage 2 (re-ranking using EMD at the patch-embedding level) using APC and a 8x8 grid.
"""

DATA_ROOT = "../data"
HASH_FUNCS = {tf.Session: id,
              torch.nn.Module: id,
              Build_model: lambda _ : None,
              torch.Tensor: lambda x: x.cpu().numpy()}

# Select images
all_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362,
                             369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301,
                             599, 600, 601, 602, 610, 700, 701,702,705,706,710, 800, 801, 802, 803, 804, 810, 820, 900, 910], dtype='int')

# all_idx = np.array([i for i in range(900)], dtype='int')
EPS = 1e-3  # arbitrary positive value
data_dir = os.path.join(os.getcwd(), 'img_data')
size = (128, 128)
original_size = (1024, 1024)
dataset = 'ffhd'
face_model_path = 'pretrained/resnet18_110.pth'
topK = 100
tops = 5
level = 8
cur_idx = 0
alpha = 0.7
fea_key = 'fea'
if level == 4:
    embed_key = 'embedding_44'
    avg_pool_key = 'adpt_pooling_44'
else:
    embed_key = 'embedding_88'
    avg_pool_key = 'adpt_pooling_88'
transformRaw = transforms.Compose([
                    transforms.Resize([original_size[0], original_size[1]]),
                    transforms.ToTensor()])

class State:  # Simple dirty hack for maintaining state
    prev_attr = None
    prev_idx = None
    first = True
    masked = False
    cur_mask_reset = 1e3
    cur_attr_reset = 1e5
    cur_light_reset = 1e7
    # ... and other state variables

if not hasattr(st, 'data'):  # Run only once. Save data globally

    st.state = State()
    with st.spinner("Setting up... This might take a few minutes"):
        raw_w = pickle.load(open(os.path.join(DATA_ROOT, "sg2latents.pickle"), "rb"))
        # raw_TSNE = np.load(os.path.join(DATA_ROOT, 'TSNE.npy'))  # We are picking images here by index instead
        raw_attr = np.load(os.path.join(DATA_ROOT, 'attributes.npy'))
        raw_lights = np.load(os.path.join(DATA_ROOT, 'light.npy'))

        all_w = np.array(raw_w['Latent'])[all_idx]
        all_attr = raw_attr[all_idx]
        all_lights = raw_lights[all_idx]

        masked = 0

        light0 = torch.from_numpy(raw_lights[8]).float()
        light1 = torch.from_numpy(raw_lights[33]).float()
        light2 = torch.from_numpy(raw_lights[641]).float()
        light3 = torch.from_numpy(raw_lights[547]).float()
        light4 = torch.from_numpy(raw_lights[28]).float()
        light5 = torch.from_numpy(raw_lights[34]).float()

        pre_lighting = [light0, light1, light2, light3, light4, light5]

        st.data = dict(masked=masked, raw_w=raw_w, all_w=all_w, all_attr=all_attr, all_lights=all_lights,
                             pre_lighting=pre_lighting)


def make_slider(name, min_value=0.0, max_value=1.0, step=0.1, **kwargs):
    return st.sidebar.slider(name, min_value, max_value, step=step, **kwargs)

@st.cache(allow_output_mutation=True, hash_funcs={dict: id}, show_spinner=False)
def get_idx2init(raw_w):
    print('get_idx2init')
    print(type(raw_w))
    idx2init = {i: np.array(raw_w['Latent'])[i] for i in all_idx}
    return idx2init

@st.cache(allow_output_mutation=True, hash_funcs={dict: id}, show_spinner=False)
def init_face_outputs():
    raw_w = pickle.load(open(os.path.join(DATA_ROOT, "sg2latents.pickle"), "rb"))
    all_w = np.array(raw_w['Latent'])
    face_model = resnet_face18(False, use_reduce_pool=True)
    state_dict = torch.load(face_model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    face_model.load_state_dict(new_state_dict)
    face_model.eval()
    face_model.cuda()

    face_datasets, data_loaders = get_face_dataloader(16,data_dir=data_dir, num_workers=64, level=level, size=size[0])
    face_dataset_gallery = face_datasets[dataset]
    feature_bank_gallery, feature_bank_center_gallery, avgpool_bank_center_gallery, _, _ = extract_embedding(data_loaders, dataset, face_model, level=level)
    mask_tool = FaceTool()
    return raw_w, all_w, face_model, face_dataset_gallery, feature_bank_gallery, feature_bank_center_gallery, avgpool_bank_center_gallery, mask_tool
    

@st.cache(hash_funcs=HASH_FUNCS)
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

@st.cache(allow_output_mutation=True, show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def flow_w_to_z(flow_model, w, attributes, lighting):
    w_cuda = torch.Tensor(w)
    att_cuda = torch.from_numpy(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    z = flow_model(w_cuda, features, zero_padding)[0].clone().detach()

    return z

@st.cache(allow_output_mutation=True, show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def flow_z_to_w(flow_model, z, attributes, lighting):
    att_cuda = torch.Tensor(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    w = flow_model(z, features, zero_padding, True)[0].clone().detach().numpy()

    return w

@st.cache(show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def generate_image(session, model, w):
    with session.as_default():
        img = model.generate_im_from_w_space(w)[0].copy()
    return img

def preserve_w_id(w_new, w_orig, attr_index):
    # Ssssh! secret sauce to strip vectors
    w_orig = torch.Tensor(w_orig)
    if attr_index == 0:
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 1:
        w_new[0][:2] = w_orig[0][:2]
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 2:

        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 3:
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 4:
        w_new[0][6:] = w_orig[0][6:]

    elif attr_index == 5:
        w_new[0][:5] = w_orig[0][:5]
        w_new[0][10:] = w_orig[0][10:]

    elif attr_index == 6:
        w_new[0][0:4] = w_orig[0][0:4]
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 7:
        w_new[0][:4] = w_orig[0][:4]
        w_new[0][6:] = w_orig[0][6:]
    return w_new


def is_new_idx_set(idx):
    if st.state.first:
        st.state.first = False
        st.state.prev_idx = idx
        return True

    if idx != st.state.prev_idx:
        st.state.prev_idx = idx
        return True
    return False

def reset_state(idx, reset_button=False):
    prev_attr_reset = st.state.cur_attr_reset
    prev_light_reset = st.state.cur_light_reset
    prev_mask_reset = st.state.cur_mask_reset
    st.state = State()
    if reset_button:
        st.state.cur_attr_reset = 1e7 if prev_attr_reset == 1e5 else 1e5
        st.state.cur_light_reset = 1e8 if prev_light_reset == 1e6 else 1e6
    st.state.cur_mask_reset = 1e10 if prev_mask_reset == 1e9 else 1e9
    st.state.first = False
    st.state.prev_idx = idx
    st.state.masked = False

def np_copy(*args):  # shortcut to clone multiple arrays
    return [np.copy(arg) for arg in args]

def get_changed_light(lights, light_names):
    for i, name in enumerate(light_names):
        change = abs(lights[name] - st.state.prev_lights[i])
        if change > EPS:
            return i
    return None

def re_ranking(img, face_model, face_dataset_gallery, feature_bank_gallery, feature_bank_center_gallery, avgpool_bank_center_gallery, size, method='apc'):
    # img = preprocessing(img_path, size).unsqueeze(0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocessingImage(img, size)
    out = face_model(img.cuda())
    anchor = out[embed_key]
    N, C, _, _ = anchor.size()
    anchor_center = out[fea_key]
    avgpool_bank_center_query = out[avg_pool_key].squeeze(-1).squeeze(-1)
    anchor = anchor.view(N, C, -1)
    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
    avgpool_bank_center_query = torch.nn.functional.normalize(avgpool_bank_center_query, p=2, dim=1)
    anchor_center = torch.nn.functional.normalize(anchor_center, p=2, dim=1)
    approx_sim, _, _, _ = emd_similarity(None, anchor_center[0], None, feature_bank_center_gallery, 0)
    approx_tops = torch.argsort(approx_sim, descending=True)
    top_inds = approx_tops[:topK]
    sim_avg, flows, _, _ = emd_similarity(anchor[0], avgpool_bank_center_query[0], feature_bank_gallery[top_inds], avgpool_bank_center_gallery[top_inds], 1, method=method)
    rank_in_tops = torch.argsort(alpha * sim_avg + (1.0 - alpha) * approx_sim[top_inds], descending=True)
    rank_in_tops_real = top_inds[rank_in_tops][:topK]
    final_tops = rank_in_tops_real.data.cpu()[:tops] #torch.cat([rank_in_tops_real, approx_tops[topK:]], dim=0).data.cpu()[:tops]
    topFlows = flows[rank_in_tops][:tops] #flows[final_tops][:tops]
    topPersonsStage1 = [face_dataset_gallery.image_paths[idx] for idx in top_inds.cpu().numpy()[:tops]]
    topPersons = [face_dataset_gallery.image_paths[idx] for idx in final_tops.numpy()[:tops]]
    return topPersonsStage1, topPersons, topFlows


def main():
    attribute_names = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7, 0.93, 1.]

    light_names = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

    att_min = {'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 'Baldness': -1, 'Beard': 0.0, 'Age': 0,
               'Expression': 0}
    att_max = {'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1}

    with st.spinner("Setting up... This might take a few minutes... Please wait!"):
        raw_w, all_w, face_model, face_dataset_gallery, feature_bank_gallery, feature_bank_center_gallery, avgpool_bank_center_gallery, mask_tool = init_face_outputs()
        all_w, all_attr, all_lights = np_copy(st.data["all_w"], st.data["all_attr"], st.data["all_lights"])
        pre_lighting = list(st.data["pre_lighting"])
        idx2w_init = get_idx2init(st.data["raw_w"])
        session, model, w_avg, flow_model = init_model()
        cur_mask_state = 'Non-Mask'


    idx_selected = st.selectbox("Choose an image:", list(range(len(idx2w_init))),
                                format_func= lambda opt : all_idx[opt])
    # print('------------ idx_selected : {} -----------'.format(idx_selected))
    w_selected = all_w[idx_selected]
    attr_selected = all_attr[idx_selected].ravel()
    lights_selected = all_lights[idx_selected]
    z_selected = flow_w_to_z(flow_model, w_selected, attr_selected, lights_selected)
    st.sidebar.markdown("# Attributes")
    resetButton = st.sidebar.button('Reset')

    if is_new_idx_set(idx_selected) or resetButton:
        reset_button = True if resetButton else False
        reset_state(idx_selected, reset_button=reset_button)
        st.state.masked = False
        st.state.prev_attr = attr_selected.copy()
        st.state.prev_lights = lights_selected.ravel().copy()
        st.state.z_current = copy.deepcopy(z_selected)
        st.state.w_current = torch.Tensor(w_selected)
        st.state.w_prev = torch.Tensor(w_selected)
        st.state.light_current = torch.Tensor(lights_selected).float()

    
    # mask_attr = st.sidebar.select_slider("", ["Non-Mask", "Mask"], value=cur_mask_state)
    num = st.state.cur_mask_reset + 1
    mask_attr = st.sidebar.slider("Mask", min_value=0.0, max_value=1.0, step=1.0, value=0.0, key=num)
    attributes = {}
    num = st.state.cur_attr_reset
    for i, att in enumerate(attribute_names):
        if idx_selected == 0:
            key = hash((idx_selected + 2000)*num + i)
        else:
            key = hash(idx_selected*num + i)

        attributes[att] = make_slider(att, float(att_min[att]), float(att_max[att]),
                                            value=float(attr_selected.ravel()[i]),  # value on first render
                                            key=key  # re-render if index selected is changed!
                                            )

    st.sidebar.markdown("# Lighting")
    lights = {}
    num = st.state.cur_light_reset
    for i, lt in enumerate(light_names):
        if idx_selected == 0:
            key = hash((idx_selected + 2000)*num + i)
        else:
            key = hash(idx_selected*num + i)
        lights[lt] = make_slider(lt,
                                    value=float(lights_selected.ravel()[i]), # value on first render
                                    key=key  # re-render if index selected is changed!
                                    )

    img_source = img_source = generate_image(session, model, w_selected) #cv2.imread(img_selected_path)

    att_new = list(attributes.values())

    for i, att in enumerate(attribute_names):  # Not the greatest code, but works!
        attr_change = attributes[att] - st.state.prev_attr[i]
        if abs(attr_change) > EPS:
            print(f"Changed attr {att} : {attr_change}")
            attr_final = attr_degree_list[i] * attr_change + st.state.prev_attr[i]
            att_new[i] = attr_final
            print("\n")

            if hasattr(st.state, 'prev_changed') and st.state.prev_changed != att:
                st.state.z_current  = flow_w_to_z(flow_model, st.state.w_current, st.state.prev_attr_factored, lights_selected)
            st.state.prev_attr[i] = attributes[att]
            st.state.prev_changed = att
            st.state.prev_attr_factored = att_new
            st.state.w_current = flow_z_to_w(flow_model, st.state.z_current, att_new, lights_selected)
            break  # Streamlit re-runs on each interaction. Probably works but need to test for any bugs here

    pre_lighting_distance = [pre_lighting[i] - st.state.light_current for i in range(len(light_names))]
    lights_magnitude = np.zeros(len(light_names))
    changed_light_index = get_changed_light(lights, light_names)

    if changed_light_index is not None:
        lights_magnitude[changed_light_index] = lights[light_names[changed_light_index]]

        lighting_final = torch.Tensor(st.state.light_current)
        for i in range(len(light_names)):
            lighting_final += lights_magnitude[i] * pre_lighting_distance[i]

        w_current = flow_z_to_w(flow_model, st.state.z_current, att_new, lighting_final)

        w_current[0][0:7] = st.state.w_current[0][0:7] # some stripping
        w_current[0][12:18] = st.state.w_current[0][12:18]

        st.state.w_current = w_current
        lights_new = lighting_final

        st.state.prev_lights[changed_light_index] = lights[light_names[changed_light_index]]
    else:
        lights_new = lights_selected

    st.state.w_current = preserve_w_id(st.state.w_current, st.state.w_prev, i)
    img_target = generate_image(session, model, st.state.w_current)
    # if mask_attr == 'Mask':
    if mask_attr == 1.0:
        masked_images, _, _, _ = mask_tool.mask_face(img_target)
        masked_img = masked_images[0]
        img_target = masked_img.copy()

    img_stage_1_paths, img_stage_2_paths, topFlows = re_ranking(img_target, face_model, face_dataset_gallery, feature_bank_gallery, feature_bank_center_gallery, avgpool_bank_center_gallery, size, method='apc')
    img_selected_path = face_dataset_gallery.image_paths[all_idx[idx_selected]]
    person_selected = img_selected_path.split('/')[-2]
    thickness = 10
    h = 512
    w = None
    scale_size = (512, 512)
    # print('------------ selected per: {} -----------'.format(person_selected))
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Original image')
    with col2:
        st.write('Query')
    with col3:
        st.write('Stage 1')
    with col4:
        st.write('Flow')
    with col5:
        st.write('Stage 2')

    for i in range(tops):
        cols = st.columns(5)
        if i == 0:
            cols[0].image(cv2.resize(img_source, scale_size), use_column_width=True, width=w)
            # if cur_mask_state == 'Mask':
            #     cols[1].image(cv2.resize(masked_img, scale_size),use_column_width=True, width=w)
            # else:
            cols[1].image(cv2.resize(img_target, scale_size),use_column_width=True, width=w)
        img_stage_1 = cv2.imread(img_stage_1_paths[i])
        img_stage_1 = cv2.cvtColor(img_stage_1, cv2.COLOR_BGR2RGB)
        img_stage_2 = cv2.imread(img_stage_2_paths[i])
        img_stage_2 = cv2.cvtColor(img_stage_2, cv2.COLOR_BGR2RGB)
        person_stage_1 = img_stage_1_paths[i].split('/')[-2]
        person_stage_2 = img_stage_2_paths[i].split('/')[-2]
        # print('------------ Stage 1 per: {} -----------'.format(person_stage_1))
        # print('------------ Stage 2 per: {} -----------'.format(person_stage_2))
        if person_stage_1 == person_selected:
            img_stage_1 = cv2.rectangle(img_stage_1, (0,0), original_size, (0, 255, 0), thickness)
        else:
            img_stage_1 = cv2.rectangle(img_stage_1, (0,0), original_size, (255, 0, 0), thickness)

        if person_stage_2 == person_selected:
            img_stage_2 = cv2.rectangle(img_stage_2, (0,0), original_size, (0, 255, 0), thickness)
        else:
            img_stage_2 = cv2.rectangle(img_stage_2, (0,0), original_size, (255, 0, 0), thickness)

        cols[2].image(cv2.resize(img_stage_1, scale_size), use_column_width=True, width=w)
        cols[4].image(cv2.resize(img_stage_2, scale_size), use_column_width=True, width=w)

        right_img = Image.open(img_stage_2_paths[i]).convert('RGB').resize((original_size[0],original_size[1]))
        right_img_transform = transformRaw(right_img)
        grid_img = draw_grid_img(topFlows[i], right_img_transform, original_size, level=level)
        cols[3].image(cv2.resize(grid_img, scale_size), use_column_width=True, width=w)

    st.state.z_current = flow_w_to_z(flow_model, st.state.w_current, att_new, lights_new)
    st.state.w_prev = torch.Tensor(st.state.w_current).clone().detach()

# class MainHandler(tornado.web.RequestHandler):
#     def get(self):
#         if __name__ == '__main__':
#             main()
            

# application = tornado.web.Application([
#     (r'/', MainHandler),
# ])

if __name__ == '__main__':
    main()
    # http_server = tornado.httpserver.HTTPServer(application, ssl_options={
    #     "certfile": "/home/hai/deepface-emd_cse_eng_auburn_edu_cert.cer",
    #     "keyfile": "/etc/ssl/private/gpu4.key",
    # })
    # # http_server.listen(443)
    # http_server.listen(8888)
    # tornado.ioloop.IOLoop.instance().start()
