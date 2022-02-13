from torchvision import models
import torch

import pickle
import numpy as np
import json

import clip

from helpers import *

import argparse

parser = argparse.ArgumentParser(description='Extract Image and CLIP Features')
parser.add_argument('--vtype', type=str, default='imagenet',
                    help='imagenet | places | emotion | clip')           
parser.add_argument('--mvsa', type=str, default='single',
                    help='single | multiple')
parser.add_argument('--ht', type=bool, default=True,
                    help='True | False')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

txt_processor = get_text_processor(htag=args.ht)
txt_transform = process_tweet

def get_resnet_feats():
    feats, logits = [], []

    def feature_hook(module, input, output):
        return feats.extend(output.view(-1,output.shape[1]).data.cpu().numpy().tolist())

    if args.vtype == 'imagenet':
        print('imgnet')
        model = models.__dict__['resnet50'](pretrained=True)
    elif args.vtype == 'places':
        print('places')
        model_file = 'pre_trained/resnet101_places_best.pth.tar'
        model = models.__dict__['resnet101'](pretrained=False, num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    elif args.vtype == 'emotion':
        print('emotion')
        model_file = 'pre_trained/best_emo_resnet50.pt'
        model = models.__dict__['resnet50'](pretrained=False, num_classes=8)
        model.load_state_dict(torch.load(model_file))

    model.eval().to(device)

    model._modules.get('avgpool').register_forward_hook(feature_hook)

    dataset = MMDataset(dloc, img_transforms, txt_transform, txt_processor)
    dt_loader = DataLoader(dataset, batch_size=128, sampler=SequentialSampler(dataset))

    for i, batch in enumerate(dt_loader):
        print(i)

        img_inputs = batch[0].to(device)

        with torch.no_grad():
            outputs = model(img_inputs)
        
        logits.extend(outputs.view(-1,outputs.shape[1]).data.cpu().numpy().tolist())


    return feats, logits



def get_clip_feats():
    img_feats, txt_feats = [], []

    model, img_preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    dataset = MMDataset(dloc, img_transform=img_preprocess, txt_transform=txt_transform, txt_processor=txt_processor)
    dt_loader = DataLoader(dataset, batch_size=128, sampler=SequentialSampler(dataset))

    for i, batch in enumerate(dt_loader):
        print(i)
        img_inps, txt_inps = batch[0].to(device), batch[1]

        txt_inps = clip.tokenize(txt_inps).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_inps)
            text_features = model.encode_text(txt_inps)
        
            img_feats.extend(image_features.cpu().numpy().tolist())
            txt_feats.extend(text_features.cpu().numpy().tolist())

    return img_feats, txt_feats



dloc = 'data/mvsa_%s/'%(args.mvsa)

if args.vtype != 'clip':
    feats, logits = get_resnet_feats()
    print(np.array(feats).shape, np.array(logits).shape)
    json.dump({'feats': feats, 'logits': logits}, open('features/%s_%s.json'%(args.vtype,args.mvsa), 'w'))
else:
    img_feats, text_feats = get_clip_feats()
    print(np.array(img_feats).shape, np.array(text_feats).shape)
    json.dump({'img_feats': img_feats, 'text_feats': text_feats}, open('features/%s_%s_ht%d.json'%(args.vtype,args.mvsa,args.ht), 'w'))