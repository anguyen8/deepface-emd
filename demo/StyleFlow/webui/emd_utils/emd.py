import torch
import torch.nn.functional as F

def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-1
    for _ in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

def emd_similarity(anchor, anchor_center, fb, fb_center, stage, method=''):
    flows = None
    u = v = None
    if stage == 0:  # stage 1: Cosine similarity
        sim = torch.einsum('c,nc->n', anchor_center, fb_center)
    else:  # stage 2: re-ranking with EMD
        N, _, R = fb.size()
        # print('------------ fb : {} -----------'.format(fb.size()))
        # print('------------ anchor : {} -----------'.format(anchor.size()))
        sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R)
        dis = 1.0 - sim
        K = torch.exp(-dis / 0.05)

        if method == 'uniform':
            u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
            v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
        elif method == 'sc':
            u = torch.sum(dis, 2)
            u = u / (u.sum(dim=1, keepdim=True) + 1e-7)
            v = torch.sum(dis, 1)
            v = v / (v.sum(dim=1, keepdim=True) + 1e-7)
        elif method == 'apc':
            att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
            u = att / (att.sum(dim=1, keepdim=True) + 1e-7)

            att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
            v = att / (att.sum(dim=1, keepdim=True) + 1e-7)
        elif method == 'uew':
            att1 = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
            att2 = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
            s = att1.sum(dim=1, keepdim=True) + att2.sum(dim=1, keepdim=True) + 1e-7
            u = att1 / s
            v = att2 / s
        else:
            print('No found method.')
            exit(0)

        
        T = Sinkhorn(K, u, v)
        sim = torch.sum(T * sim, dim=(1, 2))
        # sim = torch.nan_to_num(sim) 
        flows = T
        
    return sim, flows, u, v
