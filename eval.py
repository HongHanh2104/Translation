import torch
from torch.utils.data import DataLoader
import nltk
from tqdm import tqdm
import numpy as np

from src.dataset.en_vi_dataset import EN_VIDataset
from src.models.model import Transformer
from src.utils.utils import input_target_collate_fn

import csv
import argparse


def postprocess_wordpiece(x, tokenizer):
    return ' '.join([
        tokenizer.id_to_token(xx)
        for xx in x
    ]).replace(' ##', '').split()


def postprocess_batch(batch, eos_id):
    ps = []
    for t in batch:
        eos = np.where(t == eos_id)[0]
        if len(eos) > 0:
            ps.append(t[:eos[0]])
        else:
            ps.append(t)
    return ps


parser = argparse.ArgumentParser(description='Evaluate a model.')
parser.add_argument('--cuda',
                    help='Toggle using GPU',
                    action='store_true')
parser.add_argument('-w', '--weights',
                    help='Path to pretrained weights',
                    type=str)
parser.add_argument('-s', '--source',
                    help='Source language validation file',
                    type=str)
parser.add_argument('-t', '--target',
                    help='Target language validation file',
                    type=str)
parser.add_argument('-bs', '--batch_size',
                    help='Size of batches',
                    type=int, default=32)
args = parser.parse_args()

dev = 'cuda' if args.cuda else 'cpu'
weight_fn = args.weights
src_path = args.source
trg_path = args.target
batch_size = args.batch_size

config = torch.load(weight_fn, map_location=dev)

token_type = config['config']['dataset']['train']
token_type = token_type.get(
    'config', {'token_type': 'bpe'}).get('token_type', 'bpe')


data_cfg = {
    'src_path': src_path,
    'trg_path': trg_path,
}
if token_type == 'bpe':
    data_cfg.update({
        'token_type': 'bpe',
        'src_vocab': ["vocab/english_bpe/en-bpe-minfreq5-vocab.json",
                      "vocab/english_bpe/en-bpe-minfreq5-merges.txt"],
        'trg_vocab': ["vocab/vietnamese_bpe/vi-bpe-minfreq5-vocab.json",
                      "vocab/vietnamese_bpe/vi-bpe-minfreq5-merges.txt"],
        # src_vocab: ["vocab/shared/shared-vocab.json", "vocab/shared/shared-merges.txt"]
        # trg_vocab: ["vocab/shared/shared-vocab.json", "vocab/shared/shared-merges.txt"]
    })
elif token_type == 'wordpiece':
    data_cfg.update({
        'token_type': 'wordpiece',
        'src_vocab': 'vocab/english_word/en-wordpiece-minfreq5-vocab.txt',
        'trg_vocab': 'vocab/vietnamese_word/vi-wordpiece-minfreq5-vocab.txt',
    })
ds = EN_VIDataset(**data_cfg)
dl = DataLoader(ds, batch_size=batch_size,
                collate_fn=input_target_collate_fn)


TRG_EOS_TOKEN = '</s>'
TRG_EOS_ID = ds.vi_tokenizer.token_to_id(TRG_EOS_TOKEN)
SRC_EOS_TOKEN = '</s>'
SRC_EOS_ID = ds.en_tokenizer.token_to_id(SRC_EOS_TOKEN)

model = Transformer(
    n_src_vocab=ds.en_tokenizer.get_vocab_size(),
    n_trg_vocab=ds.vi_tokenizer.get_vocab_size(),
    src_pad_idx=ds.en_tokenizer.token_to_id('<pad>'),
    trg_pad_idx=ds.vi_tokenizer.token_to_id('<pad>'),
    **config['config']['model']
).to(dev)

model.load_state_dict(config['model_state_dict'])

model.eval()

with torch.no_grad():
    trgs, preds = [], []
    bar = tqdm(dl)
    score = 0
    for i, (src, trg) in enumerate(bar):
        bar.set_description(f'BLEU: {score:.06f}')

        src = src.to(dev)
        trg = trg.to(dev)

        trg_ = trg.cpu().numpy()[:, 1:]

        ps = postprocess_batch(trg_, SRC_EOS_ID)
        if token_type == 'bpe':
            ps = [[x.split()] for x in ds.vi_tokenizer.decode_batch(ps)]
        elif token_type == 'wordpiece':
            ps = [
                [postprocess_wordpiece(x, ds.vi_tokenizer)]
                for x in ps
            ]
        # print(ps)
        trgs += ps

        if True:
            pred = model.predict(src,
                                 max_length=256,
                                 eos_id=TRG_EOS_ID)
        else:
            pred = model(src, trg[:, :-1]).argmax(-1)
        pred_ = pred.cpu().numpy()
        # for t in pred_:
        #     print(t)
        # print(pred_)
        ps = postprocess_batch(pred_, TRG_EOS_ID)
        if token_type == 'wordpiece':
            ps = [
                postprocess_wordpiece(x, ds.vi_tokenizer)
                for x in ps
            ]
        elif token_type == 'bpe':
            ps = [x.split() for x in ds.vi_tokenizer.decode_batch(ps)]
        preds += ps

        score = nltk.translate.bleu_score.corpus_bleu(trgs, preds)

        csv.writer(open('output.txt', 'w'), delimiter=' ').writerows(preds)

    print('Final', score)
