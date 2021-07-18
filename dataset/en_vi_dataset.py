import torch
from torch.utils import data
import os

class EN_VIDataset(data.Dataset):
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