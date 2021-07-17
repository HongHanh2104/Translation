import torch
from torch.utils import data
import os

class EN_VIDataset(data.Dataset):
    def __init__(self, data_dir, phase, vocab_size=None, limit=None):
        self.data = []
        self.unk_index = 1
        self.vocab_size = vocab_size
        self.limit = limit

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
        if self.limit is not None and item >= self.limie:
            raise IndexError()
        
        indexed_srcs, indexed_trgs = self.data[item]
        return indexed_srcs, indexed_trgs
    
    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else: return self.limit