import torch
from tqdm import tqdm
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from models.model import Transformer

import sys

MAX_LEN = 256

vi_tokenizer = ByteLevelBPETokenizer(
    "vocab/vietnamese/vietnamese-vocab.json",
    "vocab/vietnamese/vietnamese-merges.txt",
)

vi_tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", vi_tokenizer.token_to_id("</s>")),
    ("<s>", vi_tokenizer.token_to_id("<s>")),
)
vi_tokenizer.enable_truncation(max_length=MAX_LEN)

en_tokenizer = ByteLevelBPETokenizer(
    "vocab/english/english-vocab.json",
    "vocab/english/english-merges.txt",
)

en_tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", en_tokenizer.token_to_id("</s>")),
    ("<s>", en_tokenizer.token_to_id("<s>")),
)
en_tokenizer.enable_truncation(max_length=MAX_LEN)


TRG_EOS_TOKEN = '</s>'
TRG_EOS_ID = vi_tokenizer.token_to_id(TRG_EOS_TOKEN)

dev = 'cuda'

config = torch.load(sys.argv[1], map_location=dev)

model_cfg = config['config']['model']
model = Transformer(
    n_src_vocab=en_tokenizer.get_vocab_size(),
    n_trg_vocab=vi_tokenizer.get_vocab_size(),
    src_pad_idx=en_tokenizer.token_to_id('<pad>'),
    trg_pad_idx=vi_tokenizer.token_to_id('<pad>'),
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
    while True:
        src = input('Q: ')
        src = torch.tensor(en_tokenizer.encode(src).ids)
        src = src.long().unsqueeze(0).to(dev)
        pred = model.predict(src, max_length=MAX_LEN, eos_id=TRG_EOS_ID)
        pred_ = pred.cpu().numpy()
        pred_ = vi_tokenizer.decode_batch(pred_)[0]
        print('A:', pred_)