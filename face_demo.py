import os
import torch
import torch.multiprocessing
import argparse
from torchvision import transforms
from PIL import Image, ImageDraw
from face_models.resnet import *
from face_models.net_cos import *
from utils.emd import emd_similarity
from utils.image import preprocessing, draw_grid_img
from utils.extract_features import extract_embedding
from data_loader.facedata_loader import get_face_dataloader
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description="DeepFace-EMD : Demos"
)

parser.add_argument("-method", type=str, default="apc",help="Method",)
parser.add_argument("-fm", type=str, default="arcface",help="face model",)
parser.add_argument("-gallery", type=str, default="lfw",help="face gallery",)
parser.add_argument("-model_path", type=str, default="pretrained/resnet18_110.pth", help="model path",)
parser.add_argument("-l", type=int, default=4,help="level of grid size",)
parser.add_argument("-query", type=str, default='', help="First input image",)
parser.add_argument('-mask', action='store_true', help="If True, masked on",)
parser.add_argument('-crop', action='store_true', help="If True, crop on",)
parser.add_argument('-sunglass', action='store_true', help="If True, sunglass on",)
parser.add_argument("-data_folder", type=str, default="data_small", help="dataset dir: data_small or data",)
parser.add_argument("-query_person", type=str, default="Winona_Ryder", help="query name person",)
args = parser.parse_args()

def find_query_img(face_dataset, query_person):
    # query_name = query.split('/')[-1].split('.')[0]
    img_query = None
    pos = -1
    for idx, img_path in enumerate(face_dataset.image_paths):
        name = img_path.split('/')[-2] #img_path.split('/')[-1].split('.')[0]
        if query_person in name:
            img_query = img_path 
            pos = idx
            break
    return img_query, pos


def main():
    model_path = args.model_path
    data_dir = os.path.join(os.getcwd(), args.data_folder)
    if args.fm == 'arcface':
        size = (128, 128)
        datasets = { 'lfw':['lfw128','lfw128_masked','lfw128_glass', 'lfw128_crop']}
    elif args.fm == 'cosface':
        size = (96, 112)
        datasets = {'lfw':['lfw', 'lfw96_mask', 'lfw96_glass', 'lfw96_crop']}
    else:
        print('No face model found')
        exit(0)
    isNormal = args.mask != 1 and args.sunglass != 1 and args.crop != 1
    query_person = args.query_person
    query_name = args.query.split('/')[-1].split('.')[0]
    if args.mask:
        query_data = datasets[args.gallery][1] # 'lfw128_masked' #'cfp128_frontal_masked' 'agedb128_masked'
        imgname =  'results/demo/{}_{}_mask.jpg'.format(query_name, args.fm)
    elif args.crop:
        query_data = datasets[args.gallery][3]
        imgname =  'results/demo/{}_{}_crop.jpg'.format(query_name, args.fm)
    elif args.sunglass:
        query_data = datasets[args.gallery][2]
        imgname =  'results/demo/{}_{}_sunglass.jpg'.format(query_name, args.fm)
    else:    
        query_data = datasets[args.gallery][0] #'cfp128_profile' #'agedb128' #
        imgname =  'results/demo/{}_{}_normal.jpg'.format(query_name, args.fm)

    face_datasets, data_loaders = get_face_dataloader(16,data_dir=data_dir, folder=args.data_folder,  fm=args.fm, num_workers=64, level=args.l, size=size[0])
    if isNormal == False:
        img_path_query, idx = find_query_img(face_datasets[query_data], query_person)  #args.query
    else:
        img_path_query = args.query
        
    if img_path_query == None:
        print('No found query image')
        exit(0)

    gallery = datasets[args.gallery][0]
    if args.fm == 'arcface':
        embed_key = 'embedding_44'
        avg_pool_key = 'adpt_pooling_44'
        model = resnet_face18(False, use_reduce_pool=True)
        state_dict = torch.load(model_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    elif args.fm == 'cosface':
        embed_key = 'embedding'
        avg_pool_key = 'adpt_pooling'
        model_path = 'pretrained/ACC99.28.pth'
        model = sphere()
        model.load_state_dict(torch.load(model_path))
    else:
        print('No found face model!!!')
        exit(0)

    model.eval()
    model = nn.DataParallel(model)
    model.cuda()

    fea_key = 'fea'
    topK = 100
    tops = 5
    alpha = 0.7
    if isNormal:
        alpha = 0.3
    level = args.l
    method = args.method
    shape = [(0, 0), (size[0], size[1])]

    query_person = img_path_query.split('/')[-2]
    img = preprocessing(img_path_query, size, fm=args.fm).unsqueeze(0)

    out = model(img.cuda())

    transformRaw = transforms.Compose([
                    transforms.Resize([size[0], size[1]]),
                    transforms.ToTensor()])
    print('query img: {}'.format(img_path_query))
    
    face_dataset_gallery = face_datasets[gallery] 
    query_img = Image.open(img_path_query).convert('RGB').resize((size[0],size[1]))
    feature_bank_gallery, feature_bank_center_gallery, avgpool_bank_center_gallery, _, _ = extract_embedding(data_loaders, gallery, model, fm=args.fm, level=args.l)
    anchor = out[embed_key]
    N, C, _, _ = anchor.size()
    anchor_center = out[fea_key]
    avgpool_bank_center_query = out[avg_pool_key].squeeze(-1).squeeze(-1)
    anchor = anchor.view(N, C, -1)
    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
    avgpool_bank_center_query = torch.nn.functional.normalize(avgpool_bank_center_query, p=2, dim=1)
    anchor_center = torch.nn.functional.normalize(anchor_center, p=2, dim=1)
    approx_sim, _, _, _ = emd_similarity(None, anchor_center[0], None, feature_bank_center_gallery, 0)
    if isNormal:
        approx_sim[idx] = -100
        
    approx_tops = torch.argsort(approx_sim, descending=True)
    top_inds = approx_tops[:topK]
    sim_avg, flows, _, _ = emd_similarity(anchor[0], avgpool_bank_center_query[0], feature_bank_gallery[top_inds], avgpool_bank_center_gallery[top_inds], 1, method=method)
    rank_in_tops = torch.argsort(alpha * sim_avg + (1.0 - alpha) * approx_sim[top_inds], descending=True)
    # rank_in_tops = torch.argsort(sim_avg + approx_sim[top_inds], descending=True)
    rank_in_tops_real = top_inds[rank_in_tops][:topK]
    final_tops = rank_in_tops_real.data.cpu()[:tops] #torch.cat([rank_in_tops_real, approx_tops[topK:]], dim=0).data.cpu()[:tops]
    topFlows = flows[rank_in_tops][:tops] #flows[final_tops][:tops]
    topPersonsStage1 = [face_dataset_gallery.image_paths[idx] for idx in top_inds.cpu().numpy()[:tops]]
    topPersons = [face_dataset_gallery.image_paths[idx] for idx in final_tops.numpy()[:tops]]
    img_rerank = Image.new('RGB', (3 * size[0], int(tops * 1 + 1)*size[1]))
    img_rerank.paste(query_img, (0*size[0], 0*size[1]))
    img_rerank.paste(query_img, (1*size[0], 0*size[1]))
    for j, flow in enumerate(topFlows):
        right_img = Image.open(topPersons[j]).convert('RGB').resize((size[0],size[1]))
        stage1Img = Image.open(topPersonsStage1[j]).convert('RGB').resize((size[0],size[1]))
        draw = ImageDraw.Draw(right_img)
        draw_stage1 = ImageDraw.Draw(stage1Img)
        right_img_transform = transformRaw(right_img)
        stage1Person = topPersonsStage1[j].split('/')[-2]
        right_person = topPersons[j].split('/')[-2]
        if query_person == stage1Person:
            draw_stage1.rectangle(shape, width = 5, outline ="green")
        else:
            draw_stage1.rectangle(shape, width = 5, outline ="red")

        if query_person == right_person:
            draw.rectangle(shape, width = 5, outline ="green")
        else:
            draw.rectangle(shape, width = 5, outline ="red")

        grid_img = draw_grid_img(flow, right_img_transform, size, fm=args.fm, level=level)
        img_rerank.paste(stage1Img, (0*size[0], (j+1)*size[1]))
        img_rerank.paste(grid_img, (1*size[0], (j+1)*size[1]))
        img_rerank.paste(right_img, (2*size[0], (j+1)*size[1]))    
    
    print('img: {}'.format(imgname))
    img_rerank.save(imgname)

if __name__ == '__main__':
    main()