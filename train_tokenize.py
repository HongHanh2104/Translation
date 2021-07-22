from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer

tok_type = 'wordpiece'
lowercase = False
files = ['data/en-vi/raw-data/train/train.vi',
         'data/en-vi/raw-data/train/train.en']
min_freq = 5
output_dir = 'vocab/shared_word'
output_name = 'shared-wordpiece-minfreq5'

if tok_type == 'bpe':
    tokenizer = ByteLevelBPETokenizer(lowercase=lowercase)
elif tok_type == 'wordpiece':
    tokenizer = BertWordPieceTokenizer(
        lowercase=lowercase,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        cls_token='<s>',
        pad_token='<pad>',
        sep_token='</s>',
        unk_token='<unk>',
        mask_token='<mask>',
    )

tokenizer.train(
    files=files,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
    min_frequency=min_freq,
)

tokenizer.save_model(output_dir, output_name)
