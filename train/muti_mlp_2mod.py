import torch.optim as optim

import random, copy
import pandas as pd
import time

from helpers import *

import argparse

parser = argparse.ArgumentParser(description='Train Multimodal MLP Models for Sentiment')
parser.add_argument('--vtype', type=str, default='clip',
                    help='imagenet | places | emotion | clip')
parser.add_argument('--ttype', type=str, default='clip',
                    help='bertbase | robertabase | clip')             
parser.add_argument('--mvsa', type=str, default='single',
                    help='single | multiple')
parser.add_argument('--ht', type=bool, default=True,
                    help='True | False')
parser.add_argument('--bs', type=int, default=32,
                    help='32, 64, 128')
parser.add_argument('--epochs', type=int, default=100,
                    help='50, 75, 100')
parser.add_argument('--lr', type=str, default='2e-5',
                    help='1e-4, 5e-5, 2e-5')
parser.add_argument('--ftype', type=str, default='feats',
                    help='feats | logits')
parser.add_argument('--layer', type=str, default='sumavg',
                    help='sumavg, 2last, last')
parser.add_argument('--norm', type=int, default=1,
                    help='0 | 1')
parser.add_argument('--split', type=int, default=1,
                    help='1-10')
parser.add_argument('--smooth', type=bool, default=False,
                    help='False | True')

args = parser.parse_args() 

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, lr_scheduler, num_epochs):

    since = time.time()

    best_model = model
    best_acc = 0.0
    best_val_loss = 100
    best_epoch = 0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        since2 = time.time()

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        tot = 0.0
        cnt = 0
        # Iterate over data.
        for inputs1, inputs2, labels in tr_loader:

            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs1, inputs2)
            _, preds = torch.max(outputs, 1)

            # loss = criterion(outputs, labels)
            loss = cal_loss(outputs, labels, smoothing=smooth)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
            tot += len(labels)

            if cnt % 50 == 0:
                print('[%d, %5d] loss: %.5f, Acc: %.2f' %
                      (epoch, cnt + 1, loss.item(), (100.0 * running_corrects) / tot))

            cnt = cnt + 1

        train_loss = running_loss / len(tr_loader)
        train_acc = running_corrects * 1.0 / (len(tr_loader.dataset))

        print('Training Loss: {:.6f} Acc: {:.2f}'.format(train_loss, 100.0 * train_acc))

        test_loss, test_acc, test_f1, _, _ = evaluate(model, vl_loader)

        print('Epoch: {:d}, Val Loss: {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}'.format(epoch, test_loss,test_acc, test_f1))

        if lr_scheduler:
        	lr_scheduler.step(test_loss)

        # deep copy the model
        if test_loss <= best_val_loss:
            best_acc = test_acc
            best_val_loss = test_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch

    time_elapsed2 = time.time() - since2
    print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed2 // 60, time_elapsed2 % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_model, best_epoch


def evaluate(model, loader):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs1, inputs2, labels in loader:

            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            outputs = model(inputs1, inputs2)

            preds = torch.argmax(outputs.data, 1)
            
            # test_loss += criterion(outputs, labels).item()
            test_loss += cal_loss(outputs, labels, smoothing=smooth).item()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        acc = metrics.accuracy_score(all_labels, all_preds)
        f1 = metrics.f1_score(all_labels, all_preds, average='weighted')

    return test_loss/len(loader), acc, f1, all_preds, all_labels



mvsa = args.mvsa
batch_size = args.bs
normalize = args.norm
init_lr = float(args.lr)
epochs = args.epochs
ftype = args.ftype
vtype = args.vtype
ttype = args.ttype
layer = args.layer
split = args.split
smooth = args.smooth
htag = args.ht

dloc = 'data/mvsa_%s/'%(args.mvsa)

## Load splits
tr_ids = pd.read_csv(dloc+'splits/train_%d.txt'%(split), header=None).to_numpy().flatten()
vl_ids = pd.read_csv(dloc+'splits/val_%d.txt'%(split), header=None).to_numpy().flatten()
te_ids = pd.read_csv(dloc+'splits/test_%d.txt'%(split), header=None).to_numpy().flatten()

pair_df = pd.read_csv(dloc+'valid_pairlist.txt', header=None)
all_labels = pair_df[1].to_numpy().flatten()

lab_train = all_labels[tr_ids]
lab_val = all_labels[vl_ids]
lab_test = all_labels[te_ids]

## Read text features
feats_text = json.load(open('features/%s_%s_ht%d.json'%(ttype, mvsa, htag),'r'))

if ttype == 'clip':
    feats_text = feats_text['text_feats']
    tdim = 512
else:
    feats_text = feats_text[layer]
    tdim = 3072 if 'catavg' in layer else 768

feats_text = np.array(feats_text)

ft_tr_txt = feats_text[tr_ids]
ft_vl_txt = feats_text[vl_ids]
ft_te_txt = feats_text[te_ids]

## Read visual features
feats_img, vdim = get_visual_feats(mvsa, vtype, ftype, htag)

ft_tr_img = feats_img[tr_ids]
ft_vl_img = feats_img[vl_ids]
ft_te_img = feats_img[te_ids]

tr_data = MultiDataset2(ft_tr_img, ft_tr_txt, lab_train, normalize)
vl_data = MultiDataset2(ft_vl_img, ft_vl_txt, lab_val, normalize)
te_data = MultiDataset2(ft_te_img, ft_te_txt, lab_test, normalize)

tr_loader = DataLoader(dataset=tr_data, batch_size=batch_size, num_workers=2, 
                        shuffle=True)
vl_loader = DataLoader(dataset=vl_data, batch_size=16, num_workers=2)
te_loader = DataLoader(dataset=te_data, batch_size=16, num_workers=2)


criterion = nn.CrossEntropyLoss().to(device)

model_ft = MultiMLP_2Mod(vdim, tdim)

model_ft.to(device)
print(model_ft)

optimizer_ft = optim.Adam(model_ft.parameters(), init_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=5,verbose=True, factor=0.1)

model_ft, best_epoch = train(model_ft, optimizer_ft, scheduler, num_epochs=epochs)

torch.save(model_ft.state_dict(), 'saved_models/multmlp2_%s_%s_%s_%s_ht%d_%s_%d.pt'%(vtype, ftype, ttype, layer, htag, mvsa, split))

te_loss, te_acc, te_f1, all_preds, all_labels = evaluate(model_ft, te_loader)
print('Best Epoch: %d, Test Acc: %.4f, %.4f, %.4f'%(best_epoch, np.round(te_loss,4), np.round(te_acc,4), np.round(te_f1,4)))