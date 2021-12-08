import os
import argparse
import torch
import torch.multiprocessing
from tqdm import trange
from PIL import Image

torch.multiprocessing.set_sharing_strategy('file_system')

from face_models.resnet import *
from face_models.net_cos import *
from face_models.facenet import InceptionResnetV1
from utils.emd import emd_similarity
from utils.metrics import get_metrics_rank
from utils.extract_features import extract_embedding
from data_loader.facedata_loader import get_face_dataloader

parser = argparse.ArgumentParser(
    description="Test DeepFace-EMD"
)


parser.add_argument("-method", type=str, default="apc",help="Methods: uniform, apc, and sc",)
parser.add_argument("-fm", type=str, default="sphereface",help="face model",)
parser.add_argument("-l", type=int, default=4,help="level of grid size",)
parser.add_argument('-mask', action='store_true', help="If True, masked on",)
parser.add_argument('-crop', action='store_true', help="If True, crop on",)
parser.add_argument('-sunglass', action='store_true', help="If True, sunglass on",)
parser.add_argument("-a", type=float, default=0.0, help="scale for emd: alpha",)
parser.add_argument("-d", type=str, default="lfw", help="dataset",)
parser.add_argument("-data_folder", type=str, default="data_small", help="dataset dir: data_small or data",)

args = parser.parse_args()

def main():
    print("args = {}".format(args))
    data_dir = os.path.join(os.getcwd(), args.data_folder)
        
    print('dataset dir: {}'.format(data_dir))
    if args.fm == 'arcface':
        size = (128, 128)
        datasets = { 'lfw':['lfw128','lfw128_masked','lfw128_glass', 'lfw128_crop'],
                    'lfw_1680':['lfw128_1680']}
    elif args.fm == 'cosface':
        size = (112, 96)
        datasets = {'lfw':['lfw', 'lfw96_mask', 'lfw96_glass', 'lfw96_crop']
                    }
    elif args.fm == 'facenet':
        size = (160, 160)
        datasets = { 'lfw':['lfw128', 'lfw128_masked', 'lfw128_glass', 'lfw128_crop']}
    else:
        print('No face model found')
        exit(0)

    
    if args.mask:
        query_data = datasets[args.d][1]
    elif args.crop:
        query_data = datasets[args.d][3] 
    elif args.sunglass:
        query_data = datasets[args.d][2] 
    else:    
        query_data = datasets[args.d][0]

    gallery_data = datasets[args.d][0] 
    print('query data: {} - gallery: {}'.format(query_data, gallery_data))
    _, data_loaders = get_face_dataloader(16, data_dir=data_dir, folder=args.data_folder, fm=args.fm, num_workers=64, level=args.l, size=size[0])
    if args.fm == 'arcface':
        model_path =  'pretrained/resnet18_110.pth'
        print('model : {}'.format(model_path))
        model = resnet_face18(False, use_reduce_pool=False)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    elif args.fm == 'cosface':
        model_path = 'pretrained/ACC99.28.pth'
        model = sphere()
        model.load_state_dict(torch.load(model_path))
    elif args.fm == 'facenet':
        model_path = 'pretrained/20180402-114759-vggface2.pt'
        model = InceptionResnetV1()
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()
    feature_bank_query, feature_bank_center_query, avgpool_bank_center_query, labels_query, _ = extract_embedding(data_loaders, query_data, model, fm=args.fm, level=args.l)
    feature_bank_gallery, feature_bank_center_gallery, avgpool_bank_center_gallery, labels_gallery, _ = extract_embedding(data_loaders, gallery_data, model, fm=args.fm, level=args.l)
    stages = [0, 100]
    overall_r1 = {k: 0.0 for k in stages}
    overall_rp = {k: 0.0 for k in stages}
    overall_mapr = {k: 0.0 for k in stages}

    N, C, _ = feature_bank_query.size()
    alpha = args.a
    for idx in trange(len(feature_bank_query)):
        anchor_center = feature_bank_center_query[idx]
        approx_sim, _, _, _ = emd_similarity(None, anchor_center, None, feature_bank_center_gallery, 0)
        if query_data == gallery_data:
            approx_sim[idx] = -100
        approx_tops = torch.argsort(approx_sim, descending=True)
        if max(stages) > 0:
            top_inds = approx_tops[:max(stages)]
            anchor = feature_bank_query[idx]
            feature_query = avgpool_bank_center_query[idx]
            feature_gallery = avgpool_bank_center_gallery[top_inds]
            sim_avg, _, _, _ = emd_similarity(anchor, feature_query, feature_bank_gallery[top_inds], feature_gallery, 1, method=args.method)
            if alpha < 0:
                rank_in_tops = torch.argsort(sim_avg + approx_sim[top_inds], descending=True)
            else:
                rank_in_tops = torch.argsort(alpha* sim_avg + (1.0 - alpha) * approx_sim[top_inds], descending=True)
        
        for stage in stages:
            if stage == 0:
                final_tops = approx_tops
            else:
                rank_in_tops_real = top_inds[rank_in_tops][:stage]
                final_tops = torch.cat([rank_in_tops_real, approx_tops[stage:]], dim=0)
            
            r1, rp, mapr = get_metrics_rank(final_tops.data.cpu(), labels_query[idx], labels_gallery)
            overall_r1[stage] += r1
            overall_rp[stage] += rp
            overall_mapr[stage] += mapr
    
    for i, stage in enumerate(stages):
        overall_r1[stage] /= float(N / 100)
        overall_rp[stage] /= float(N / 100)
        overall_mapr[stage] /= float(N / 100)
        print('[stage %d] acc=%f, RP=%f, MAP@R=%f' % (i+1, overall_r1[stage], overall_rp[stage], overall_mapr[stage]))

if __name__ == '__main__':
    main()


