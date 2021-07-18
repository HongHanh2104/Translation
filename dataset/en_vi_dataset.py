from preprocess import preprocess_vi_data
import torch
from torch.utils import data
import os

from vncorenlp import VnCoreNLP
from transformers import AutoModel, AutoTokenizer

class _EN_VIDataset(data.Dataset):
    def __init__(self, data_dir, phase, vocab_size=None, limit=None, max_len=256):
        self.data = []
        self.unk_index = 1
        self.vocab_size = vocab_size
        self.limit = limit
        self.max_len = max_len

        check_index = lambda index: index if index < vocab_size else self.unk_index

        with open(os.path.join(data_dir, f'indexed-{phase}.txt')) as file:
            for line in file:
                srcs, trgs = line.strip().split('\t')
                if vocab_size is not None:
                    indexed_srcs = [check_index(int(index)) for index in srcs.strip().split(' ')]
                    indexed_trgs = [check_index(int(index)) for index in trgs.strip().split(' ')]
                
                else:
                    indexed_srcs = [int(index) for index in srcs.strip().split(' ')]
                    indexed_trgs = [int(index) for index in trgs.strip().split(' ')]

                self.data.append((indexed_srcs, indexed_trgs))
                if limit is not None and len(self.data) > limit:
                    break
        
    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()
        
        indexed_src, indexed_trg = self.data[item]
        if len(indexed_src) > self.max_len:
            indexed_src = indexed_src[:self.max_len]
        if len(indexed_trg) > self.max_len:
            indexed_trg = indexed_trg[:self.max_len]
        
        return indexed_src, indexed_trg
    
    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else: return self.limit

class EN_VIDataset(data.Dataset):
    def __init__(self, 
                 data_dir, 
                 phase, 
                 vocab_size=None, 
                 max_len=256):
        self.src_data = []
        self.trg_data = []
        self.unk_index = 1
        self.vocab_size = vocab_size
        self.max_len = max_len

        check_index = lambda index: index if index < vocab_size else self.unk_index

        self.root_path = os.path.join(data_dir, phase)

        with open(os.path.join(self.root_path, 'train.en')) as file:
            for line in file:
                self.src_data.append(line)
        
        with open(os.path.join(self.root_path, 'train.vi')) as file:
            for line in file:
                self.trg_data.append(line)
        
        self.rdrsegmenter = VnCoreNLP("../vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
        self.vi_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.en_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=False)
        
        
    def __getitem__(self, idx):
        
        src, trg = self.src_data[idx], self.trg_data[idx]
        if len(src) > self.max_len:
            indexed_src = src[:self.max_len]
        if len(trg) > self.max_len:
            indexed_trg = trg[:self.max_len]
        
        indexed_src = self._tokenize_src_data(src)
        indexed_trg = self._tokenize_trg_data(trg)

        return torch.tensor(indexed_src), torch.tensor(indexed_trg)
    
    def __len__(self):
        return len(self.src_data)
    
    def _tokenize_trg_data(self, trg_sen):
        preprocesed_sen = ''
        sentences = self.rdrsegmenter.tokenize(trg_sen)
        for sentence in sentences:
            preprocesed_sen += " ".join(sentence) + ' '

        token = self.vi_tokenizer.encode(preprocesed_sen)  #(line, padding=False, max_length=1000)["input_ids"]
        return token
    
    def _tokenize_src_data(self, src_sen):
        token = self.en_tokenizer.encode(src_sen)  #(line, padding=False, max_length=1000)["input_ids"]
        return token
