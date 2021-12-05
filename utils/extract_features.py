import torch 
from tqdm import tqdm

def extract_embedding(data_loaders, dataset, model, fm='arcface', level=4):
    model.eval()
    dataloader = data_loaders[dataset]
    labels = []
    with torch.no_grad():
        feature_bank = []
        feature_bank_center = []
        avgpool_bank_center = []
        weights = []
        final_iter = tqdm(dataloader, desc='Embedding Data...')
        for idx, inp in enumerate(final_iter):
            input_img, target = inp[0], inp[1]
            out = model(input_img.cuda())
            fea = out['fea']
            if fm == 'arcface':
                if level == 4:
                    aux_f = out['embedding_44']
                    avg_pool = out['adpt_pooling_44']
                elif level == 8:
                    aux_f = out['embedding_88'] 
                    avg_pool = out['adpt_pooling_88']
                else:
                    aux_f = out['embedding_16']
                    avg_pool = out['adpt_pooling_16']
            elif fm == 'sphereface' or fm == 'cosface' or fm == 'facenet':
                aux_f = out['embedding']
                avg_pool = out['adpt_pooling']
            no_avg_feat = aux_f

            avgpool_bank_center.append(avg_pool.data)
            feature_bank.append(no_avg_feat.data)
            feature_bank_center.append(fea.data)
            labels.append(target)
        if len(weights) > 0:
            weights = torch.cat(weights, dim=0)

        labels = torch.cat(labels, dim=0)
        labels = labels.squeeze(-1)
        feature_bank = torch.cat(feature_bank, dim=0)
        N, C, _, _ = feature_bank.size()
        feature_bank = feature_bank.view(N, C, -1)
        feature_bank_center = torch.cat(feature_bank_center, dim=0)
        avgpool_bank_center = torch.cat(avgpool_bank_center, dim=0).squeeze(-1).squeeze(-1)

    feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)
    feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1)
    avgpool_bank_center = torch.nn.functional.normalize(avgpool_bank_center, p=2, dim=1)

    return feature_bank, feature_bank_center, avgpool_bank_center, labels, weights