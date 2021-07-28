import torch
from tqdm import tqdm
from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

from src.models.model import Transformer

import sys

MAX_LEN = 256

dev = 'cuda'

config = torch.load(sys.argv[1], map_location=dev)
data_cfg = config['config']['dataset']['train']['config']
token_type = data_cfg['token_type']

if token_type == 'bpe':
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
elif token_type == 'wordpiece':
    vi_tokenizer = BertWordPieceTokenizer(
        data_cfg['trg_vocab'],
        lowercase=False,
        handle_chinese_chars=True,
        strip_accents=False,
        cls_token='<s>',
        pad_token='<pad>',
        sep_token='</s>',
        unk_token='<unk>',
        mask_token='<mask>',
    )
    vi_tokenizer.enable_truncation(max_length=MAX_LEN)

    en_tokenizer = BertWordPieceTokenizer(
        data_cfg['src_vocab'],
        lowercase=False,
        handle_chinese_chars=True,
        strip_accents=False,
        cls_token='<s>',
        pad_token='<pad>',
        sep_token='</s>',
        unk_token='<unk>',
        mask_token='<mask>',
    )
    en_tokenizer.enable_truncation(max_length=MAX_LEN)


TRG_EOS_TOKEN = '</s>'
TRG_EOS_ID = vi_tokenizer.token_to_id(TRG_EOS_TOKEN)

model = Transformer(
    n_src_vocab=en_tokenizer.get_vocab_size(),
    n_trg_vocab=vi_tokenizer.get_vocab_size(),
    src_pad_idx=en_tokenizer.token_to_id('<pad>'),
    trg_pad_idx=vi_tokenizer.token_to_id('<pad>'),
    **config['config']['model']
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
