from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch import nn
from torch.nn import functional as F
import torch

import numpy as np
import json
from sklearn import metrics, preprocessing

class MultiDataset2(Dataset):
    def __init__(self, vfeats, tfeats, labels, normalize=1):
        self.vfeats = vfeats
        self.tfeats = tfeats
        self.labels = np.array(labels).astype(np.int)
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vfeat = self.vfeats[idx]
        tfeat = self.tfeats[idx]
        label = self.labels[idx]

        if self.normalize:
            vfeat = preprocessing.normalize(vfeat.reshape(1,-1), axis=1).flatten() 
            tfeat = preprocessing.normalize(tfeat.reshape(1,-1), axis=1).flatten() 

        return torch.FloatTensor(vfeat), torch.FloatTensor(tfeat), torch.tensor(label)


class MultiMLP_2Mod(nn.Module):
    def __init__(self, vdim, tdim):
        super(MultiMLP_2Mod, self).__init__()
        
        self.vfc1 = nn.Linear(vdim, 128)
        self.tfc1 = nn.Linear(tdim, 128)
        self.vbn1 = nn.BatchNorm1d(128)
        self.tbn1 = nn.BatchNorm1d(128)
        self.cf = nn.Linear(256, 3)

        self.vdp1 = nn.Dropout(0.5)
        self.tdp1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.vdp1(self.relu(self.vbn1(self.vfc1(x1))))
        x2 = self.tdp1(self.relu(self.tbn1(self.tfc1(x2))))
        x = torch.cat((x1,x2), axis=1)

        return self.cf(x)



def get_visual_feats(mvsa, vtype, ftype, htag):
    if vtype == 'places':
        feats_img = json.load(open('features/places_%s.json'%(mvsa), 'r'))[ftype]
        vdim = 2048 if ftype == 'feats' else 365
    elif vtype == 'emotion':
        feats_img = json.load(open('features/emotion_%s.json'%(mvsa), 'r'))[ftype]
        vdim = 2048 if ftype == 'feats' else 8
    elif vtype == 'imagenet':
        feats_img  = json.load(open('features/imagenet_%s.json'%(mvsa), 'r'))[ftype]
        vdim = 2048 if ftype == 'feats' else 1000
    elif vtype == 'clip':
        feats_img  = json.load(open('features/clip_%s_ht%d.json'%(mvsa,htag), 'r'))['img_feats']
        vdim = 512
    else:
        feats_img = json.load(open('features/faces_%s.json'%(mvsa),'r'))[ftype]
        vdim = 512 if ftype == 'feats' else 7

    return np.array(feats_img), vdim


def cal_loss(pred, gold, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.mean()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
        
    return loss
