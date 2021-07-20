import torch
from torch.utils.data import DataLoader
import nltk
from tqdm import tqdm

from dataset.en_vi_dataset import EN_VIDataset
from models.model import Transformer
from utils.utils import input_target_collate_fn

import sys

src_path = 'data/en-vi/raw-data/test/tst2013.en'
trg_path = 'data/en-vi/raw-data/test/tst2013.vi'

ds = EN_VIDataset(src_path=src_path, trg_path=trg_path)
dl = DataLoader(ds, batch_size=32, shuffle=True,
                collate_fn=input_target_collate_fn)


TRG_EOS_TOKEN = '</s>'
TRG_EOS_ID = ds.vi_tokenizer.token_to_id(TRG_EOS_TOKEN)

dev = 'cuda'

config = torch.load(sys.argv[1], map_location=dev)

model_cfg = config['config']['model']
model = Transformer(
    n_src_vocab=ds.en_tokenizer.get_vocab_size(),
    n_trg_vocab=ds.vi_tokenizer.get_vocab_size(),
    src_pad_idx=ds.en_tokenizer.token_to_id('<pad>'),
    trg_pad_idx=ds.vi_tokenizer.token_to_id('<pad>'),
    max_len=model_cfg['max_len'],
    d_model=model_cfg['d_model'],
    d_ffn=model_cfg['d_ffn'],
    n_layer=model_cfg['n_layer'],
    n_head=model_cfg['n_head'],
    dropout=model_cfg['dropout']
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

        # src_ = src.cpu().numpy()
        # src_ = ds.en_tokenizer.decode_batch(src_)[0]
        # print(src_)

        trg_ = trg.cpu().numpy()[:, 1:]
        trg_ = ds.vi_tokenizer.decode_batch(trg_)
        ps = []
        for p in trg_:
            eos = p.find('</s>')
            if eos != -1:
                p = p[:eos]
            p = p.split()
            ps.append([p])
        trgs += ps

        if True:
            pred = model.predict(src,
                                 max_length=256,
                                 eos_id=TRG_EOS_ID)
        else:
            pred = model(src, trg[:, :-1]).argmax(-1)
        pred_ = pred.cpu().numpy()
        pred_ = ds.vi_tokenizer.decode_batch(pred_)
        ps = []
        for p in pred_:
            eos = p.find('</s>')
            if eos != -1:
                p = p[:eos]
            p = p.split()
            ps.append(p)
        preds += ps

        # if i == 5:
        #     for j in range(i):
        #         print('Target', ' '.join(trgs[j][0]))
        #         print('Pred', ' '.join(preds[j]))
        #         input()

        score = nltk.translate.bleu_score.corpus_bleu(trgs, preds)
        # print(trgs, preds)
        # input()

    print('Final', score)
