import json
import string

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer

from helpers import *

import argparse

parser = argparse.ArgumentParser(description='Extract BERT Features')
parser.add_argument('--btype', type=str, default='robertabase',
                    help='bertbase | robertabase')           
parser.add_argument('--mvsa', type=str, default='single',
                    help='single | multiple')
parser.add_argument('--ht', type=bool, default=True,
                    help='True | False')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


txt_processor = get_text_processor(htag=args.ht)
txt_transform = process_tweet

    
dloc = 'data/mvsa_%s/'%(args.mvsa)
bert_type = {'bertbase': (BertModel,    BertTokenizer, 'bert-base-uncased'), 
            'robertabase': (RobertaModel,    RobertaTokenizer, 'roberta-base')}[args.btype]

tokenizer = bert_type[1].from_pretrained(bert_type[2])
model = bert_type[0].from_pretrained(bert_type[2], output_hidden_states=True)
model.to(device).eval()


embed_dict = {'catavg':[], 'sumavg': [], '2last': [], 'last': []}

ph_data = MMDataset(dloc, txt_transform=txt_transform, txt_processor=txt_processor)
ph_loader = DataLoader(ph_data, batch_size=1, sampler=SequentialSampler(ph_data))

## Loader batch size is 1 because getting features according to tweet's length.
## Not padding all tweets to the same length as done usually during training a sequential model
for i, batch in enumerate(ph_loader):
    print(i)
    txt_inps = batch[1]

    sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last \
        = get_bert_embeddings(txt_inps, model, tokenizer, device)
    
    # embed_dict['catavg'].append(sent_word_catavg.tolist())
    embed_dict['sumavg'].append(sent_word_sumavg.tolist())
    embed_dict['2last'].append(sent_emb_2_last.tolist())
    embed_dict['last'].append(sent_emb_last.tolist())

json.dump(embed_dict, open('features/%s_%s_ht%d.json'%(args.btype, args.mvsa, args.ht), 'w'))
