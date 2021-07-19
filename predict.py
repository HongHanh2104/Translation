import torch
from torch.utils.data import DataLoader

from dataset.en_vi_dataset import EN_VIDataset
from models.model import Transformer
from utils.utils import input_target_collate_fn

src_path = 'data/en-vi/raw-data/train/train.en'
trg_path = 'data/en-vi/raw-data/train/train.vi'

ds = EN_VIDataset(src_path=src_path, trg_path=trg_path)
dl = DataLoader(ds, batch_size=1, shuffle=False,
                collate_fn=input_target_collate_fn)

dev = 'cpu'

config = torch.load(
    'checkpoints/Transformers-en-to-vi-2021_07_18-22_23_54/best_bleu-bleu=0.144.pth',
    map_location=dev)

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

# src_data = open(src_path).read().splitlines()
# trg_data = open(trg_path).read().splitlines()

with torch.no_grad():
    for src, trg in dl:
        src = src.to(dev)
        trg = trg.to(dev)
        print(ds.en_tokenizer.decode_batch(src.cpu().numpy()))
        print(ds.vi_tokenizer.decode_batch(trg.cpu().numpy()))
        out = model(src, trg[:, :-1])
        print(ds.vi_tokenizer.decode_batch(out.argmax(-1).cpu().numpy()))
        input()
