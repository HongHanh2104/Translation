from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
import argparse

parser = argparse.ArgumentParser(description='Train a tokenizer')
parser.add_argument('-t', '--token_type',
                    help='Tokenization type',
                    type=str,
                    choices=['wordpiece', 'bpe'])
parser.add_argument('-l', '--lowercase',
                    help='Toggle using lowercase',
                    action='store_true')
parser.add_argument('-f', '--files',
                    help='List of files to train tokenizers',
                    type=str,
                    nargs='+')
parser.add_argument('-m', '--min_freq',
                    help='Minimum number of occurrences for a token to be included',
                    type=int,
                    default=0)
parser.add_argument('-o', '--out_dir',
                    help='Output directory',
                    type=str,
                    default='.')
parser.add_argument('-of', '--out_fn',
                    help='Output filename',
                    type=str,
                    default='tokenization')
args = parser.parse_args()

tok_type = args.token_type
lowercase = args.lowercase
files = args.files
min_freq = args.min_freq
output_dir = args.out_dir
output_name = args.out_fn

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
