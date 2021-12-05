import torch
import torch.nn.functional as F
import torch.multiprocessing
import argparse
from torchvision import transforms
import torchvision
from PIL import Image
from face_models.resnet import *
from utils.emd import emd_similarity
from utils.heatmap import combine_img_heatmap, get_patch_location
from utils.image import preprocessing

parser = argparse.ArgumentParser(
    description="Visualize DeepFace-EMD"
)

parser.add_argument("-method", type=str, default="apc",help="Method",)
parser.add_argument("-fm", type=str, default="arcface",help="face model",)
parser.add_argument("-model_path", type=str, default="pretrained/resnet18_110.pth", help="model path",)
parser.add_argument("-l", type=int, default=4,help="level of grid size",)
parser.add_argument("-in1", type=str, default='', help="First input image",)
parser.add_argument("-in2", type=str, default='', help="Second input image",)
parser.add_argument("-weight", action='store_true', help="Showing weight only, otherwise grid images",)
args = parser.parse_args()

def main():
    model_path = args.model_path
    size = (128, 128)
    model = resnet_face18(False, use_reduce_pool=True)
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()

    embed_key = 'embedding_44'
    avg_pool_key = 'adpt_pooling_44'
    level = args.l
    method = args.method
    shape = [(0, 0), (size[0], size[1])]
    img_path_1 = args.in1
    img_path_2 = args.in2
    img1 = preprocessing(img_path_1).unsqueeze(0)
    img2 = preprocessing(img_path_2).unsqueeze(0)
    out1 = model(img1.cuda())
    out2 = model(img2.cuda())

    transformRaw = transforms.Compose([
                    transforms.Resize([size[0], size[1]]),
                    transforms.ToTensor()])
    query_img = Image.open(img_path_1).convert('RGB').resize((size[0],size[1]))
    right_img = Image.open(img_path_2).convert('RGB').resize((size[0],size[1]))
    anchor = out1[embed_key]
    avgpool_bank_center_query = out1[avg_pool_key].squeeze(-1).squeeze(-1)
    feature_bank_gallery = out2[embed_key]
    avgpool_bank_center_gallery = out2[avg_pool_key].squeeze(-1).squeeze(-1)
    N, C, _, _ = anchor.size()
    anchor = anchor.view(N, C, -1)
    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
    feature_bank_gallery = feature_bank_gallery.view(N, C, -1)
    feature_bank_gallery = torch.nn.functional.normalize(feature_bank_gallery, p=2, dim=1)

    _, flows, u, v = emd_similarity(anchor[0], avgpool_bank_center_query[0], feature_bank_gallery, avgpool_bank_center_gallery, 1, method=method)
    if args.weight:
        image = Image.new('RGB', (2 * size[0], size[1]))
        u, v = u.view(level,level), v.view(level,level)
        u, v = u.view(1,1,level,level), v.view(1,1,level,level)
        u = F.interpolate(u,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
        v = F.interpolate(v,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
        q_img_heatmap = combine_img_heatmap(query_img, u.cpu().detach().numpy())
        right_img_heatmap = combine_img_heatmap(right_img, v.cpu().detach().numpy())
        image.paste(q_img_heatmap, ((0, 0)))
        image.paste(right_img_heatmap, ((1*size[0], 0)))
        imgname = 'results/heatmap/heatmap_face.jpg'
    else:
        image = Image.new('RGB', (3 * size[0], size[1]))
        right_img_transform = transformRaw(right_img)
        flow = flows[0]
        patch_list = []
        weight = flow.sum(-1)
        nums = flow.shape[0]
        weight=(weight-weight.min())/(weight.max()-weight.min())
        for index_grid in range(nums):
            index_patch=torch.argmax(flow[index_grid]).item()
            row_location, col_location, _ , _ = get_patch_location(index_patch, size[0], args.fm, level=level)
            patch = right_img_transform[:, row_location[0]:row_location[1], col_location[0]:col_location[1]].cuda()
            patch = patch * weight[index_grid]
            patch_list.append(patch)

        patch_list = torch.stack(patch_list, dim=0)
        grids = torchvision.utils.make_grid(patch_list,nrow=level,padding=0)
        grids = grids.permute(1,2,0).cpu().detach().numpy() * 255.0
        grid_img = Image.fromarray(grids.astype('uint8'))
        image.paste(query_img, ((0, 0)))
        image.paste(grid_img, (1*size[0], 0))
        image.paste(right_img, (2*size[0], 0))
        imgname = 'results/flows/flow_face.jpg'
    
    
    print('img: {}'.format(imgname))
    image.save(imgname)

if __name__ == '__main__':
    main()