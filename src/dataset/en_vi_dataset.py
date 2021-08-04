from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

import torch
from torch.utils import data
import os

# from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer


# class _EN_VIDataset(data.Dataset):
#     def __init__(self, data_dir, phase, vocab_size=None, limit=None, max_len=256):
#         self.data = []
#         self.unk_index = 1
#         self.vocab_size = vocab_size
#         self.limit = limit
#         self.max_len = max_len

#         def check_index(
#             index): return index if index < vocab_size else self.unk_index

#         with open(os.path.join(data_dir, f'indexed-{phase}.txt')) as file:
#             for line in file:
#                 srcs, trgs = line.strip().split('\t')
#                 if vocab_size is not None:
#                     indexed_srcs = [check_index(int(index))
#                                     for index in srcs.strip().split(' ')]
#                     indexed_trgs = [check_index(int(index))
#                                     for index in trgs.strip().split(' ')]

#                 else:
#                     indexed_srcs = [int(index)
#                                     for index in srcs.strip().split(' ')]
#                     indexed_trgs = [int(index)
#                                     for index in trgs.strip().split(' ')]

#                 self.data.append((indexed_srcs, indexed_trgs))
#                 if limit is not None and len(self.data) > limit:
#                     break

#     def __getitem__(self, item):
#         if self.limit is not None and item >= self.limit:
#             raise IndexError()

#         indexed_src, indexed_trg = self.data[item]
#         if len(indexed_src) > self.max_len:
#             indexed_src = indexed_src[:self.max_len]
#         if len(indexed_trg) > self.max_len:
#             indexed_trg = indexed_trg[:self.max_len]

#         return indexed_src, indexed_trg

#     def __len__(self):
#         if self.limit is None:
#             return len(self.data)
#         else:
#             return self.limit


# class __EN_VIDataset(data.Dataset):
#     def __init__(self,
#                  src_path,
#                  trg_path,
#                  max_len=256):
#         self.max_len = max_len

#         self.src_data = open(src_path).read().splitlines()
#         self.trg_data = open(trg_path).read().splitlines()

#         self.rdrsegmenter = VnCoreNLP("../vncorenlp/VnCoreNLP-1.1.1.jar",
#                                       annotators="wseg", max_heap_size='-Xmx500m')
#         self.vi_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
#         self.en_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

#     def __getitem__(self, idx):

#         src, trg = self.src_data[idx], self.trg_data[idx]

#         indexed_src = self._tokenize_src_data(src)
#         indexed_trg = self._tokenize_trg_data(trg)

#         return torch.tensor(indexed_src), torch.tensor(indexed_trg)

#     def __len__(self):
#         return len(self.src_data)

#     def _tokenize_trg_data(self, trg_sen):
#         preprocesed_sen = ''
#         sentences = self.rdrsegmenter.tokenize(trg_sen)
#         for sentence in sentences:
#             preprocesed_sen += " ".join(sentence) + ' '

#         token = self.vi_tokenizer(preprocesed_sen, padding='max_length',
#                                   truncation=True, max_length=self.max_len)["input_ids"]
#         return token

#     def _tokenize_src_data(self, src_sen):
#         token = self.en_tokenizer(
#             src_sen, padding='max_length', truncation=True, max_length=self.max_len)["input_ids"]
#         return token


class EN_VIDataset(data.Dataset):
    def __init__(self,
                 src_path,
                 trg_path,
                 token_type='bpe',
                 lowercase=False,
                 src_vocab=["vocab/english_bpe/en-bpe-minfreq5-vocab.json",
                            "vocab/english_bpe/en-bpe-minfreq5-merges.txt"],
                 trg_vocab=["vocab/vietnamese_bpe/vi-bpe-minfreq5-vocab.json",
                            "vocab/vietnamese_bpe/vi-bpe-minfreq5-merges.txt"],
                 max_len=256):
        self.max_len = max_len

        self.src_data = open(src_path).read().splitlines()
        self.trg_data = open(trg_path).read().splitlines()

        if token_type == 'bpe':
            self.vi_tokenizer = ByteLevelBPETokenizer(
                *trg_vocab,
                lowercase=lowercase
            )
            self.vi_tokenizer._tokenizer.post_processor = BertProcessing(
                ("</s>", self.vi_tokenizer.token_to_id("</s>")),
                ("<s>", self.vi_tokenizer.token_to_id("<s>")),
            )
            self.vi_tokenizer.enable_truncation(max_length=max_len)

            self.en_tokenizer = ByteLevelBPETokenizer(
                *src_vocab,
                lowercase=lowercase
            )
            self.en_tokenizer._tokenizer.post_processor = BertProcessing(
                ("</s>", self.en_tokenizer.token_to_id("</s>")),
                ("<s>", self.en_tokenizer.token_to_id("<s>")),
            )
            self.en_tokenizer.enable_truncation(max_length=max_len)
        elif token_type == 'wordpiece':
            self.vi_tokenizer = BertWordPieceTokenizer(
                trg_vocab,
                lowercase=lowercase,
                # clean_text=True,
                handle_chinese_chars=True,
                strip_accents=False,
                cls_token='<s>',
                pad_token='<pad>',
                sep_token='</s>',
                unk_token='<unk>',
                mask_token='<mask>',
            )
            self.vi_tokenizer.enable_truncation(max_length=max_len)

            self.en_tokenizer = BertWordPieceTokenizer(
                src_vocab,
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
            self.en_tokenizer.enable_truncation(max_length=max_len)

    def __getitem__(self, idx):
        src, trg = self.src_data[idx], self.trg_data[idx]

        indexed_src = self._tokenize_src_data(src)
        indexed_trg = self._tokenize_trg_data(trg)

        return indexed_src, indexed_trg

    def __len__(self):
        return len(self.src_data)

    def _tokenize_trg_data(self, trg_sen):
        token = self.vi_tokenizer.encode(trg_sen).ids
        return token

    def _tokenize_src_data(self, src_sen):
        token = self.en_tokenizer.encode(src_sen).ids
        return token
