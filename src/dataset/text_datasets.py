import torch
from torch.utils import data
import os

START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'

class TranslationDataset(data.Dataset):
    def __init__(self, data_dir, phase, limit=None):
        super().__init__()

        assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'."

        self.limit = limit
        self.data = []

        with open(os.path.join(data_dir, f'raw-{phase}.txt')) as file:
            for line in file:
                src, trg = line.strip().split('\t')
                self.data.append((src, trg))
        
    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()
        
        return self.data[item]
    
    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else:
            return self.limit
    
    @staticmethod
    def prepare(train_src, train_trg, val_src, val_trg, save_data_dir):

        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)

        for phase in ('train', 'val'):
            if phase == 'train':
                src_filepath = train_src
                trg_filepath = train_trg
            else:
                src_filepath = val_src
                trg_filepath = val_trg

            with open(src_filepath) as src_file:
                src_data = src_file.readlines()

            with open(trg_filepath) as trg_filepath:
                trg_data = trg_filepath.readlines()

            with open(os.path.join(save_data_dir, f'raw-{phase}.txt'), 'w') as file:
                for src_line, trg_line in zip(src_data, trg_data):
                    src_line = src_line.strip()
                    trg_line = trg_line.strip()
                    line = f'{src_line}\t{trg_line}\n'
                    file.write(line)

class TokenizedTranslationDataset(data.Dataset):
    def __init__(self, data_dir, phase, limit=None):
        super().__init__()
        self.dataset = TranslationDataset(data_dir, phase, limit)
    
    def __getitem__(self, item):
        src, trg = self.dataset[item]
        tokenized_src = src.split()
        tokenized_trg = trg.split()
        return tokenized_src, tokenized_trg

    def __len__(self):
        return len(self.dataset)

class InputTargetTranslationDataset(data.Dataset):
    def __init__(self, data_dir, phase, limit=None):
        self.tokenized_dataset = TokenizedTranslationDataset(data_dir, phase, limit)
    
    def __getitem__(self, item):
        tokenized_src, tokenized_trg = self.tokenized_dataset[item]
        full_trg = [START_TOKEN] + tokenized_trg + [END_TOKEN]
        inputs = full_trg[:-1]
        targets = full_trg[1:]
        return tokenized_src, inputs, targets 
    
    def __lem__(self):
        return len(self.tokenized_dataset)

class IndexedInputTargetTranslationDataset(data.Dataset):
    def __init__(self, data_dir, phase, vocab_size=None, limit=None):
        self.data = []
        self.unk_index = 1
        self.vocab_size = vocab_size
        self.limit = limit

        unknownify = lambda index: index if index < vocab_size else self.unk_index

        with open(os.path.join(data_dir, f'indexed-{phase}.txt')) as file:
            for line in file:
                srcs, inps, trgs = line.strip().split('\t')
                if vocab_size is not None:
                    indexed_srcs = [unknownify(int(index)) for index in srcs.strip().split(' ')]
                    indexed_inps = [unknownify(int(index)) for index in inps.strip().split(' ')]
                    indexed_trgs = [unknownify(int(index)) for index in trgs.strip().split(' ')]
                
                else:
                    indexed_srcs = [int(index) for index in srcs.strip().split(' ')]
                    indexed_inps = [int(index) for index in inps.strip().split(' ')]
                    indexed_trgs = [int(index) for index in trgs.strip().split(' ')]

                self.data.append((indexed_srcs, indexed_inps, indexed_trgs))
                if limit is not None and len(self.data) > limit:
                    break
        
    def __getitem__(self, item):
        if self.limit is not None and item >= self.limie:
            raise IndexError()
        
        indexed_srcs, indexed_inps, indexed_trgs = self.data[item]
        return indexed_srcs, indexed_inps, indexed_trgs
    
    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else: return self.limit
    
    @staticmethod
    def prepare(data_dir, src_dict, trg_dict):
        join_inds = lambda indexes: ' '.join(str(index) for index in indexes)
        for phase in ('train', 'val'):
            inp_trg_dataset = InputTargetTranslationDataset(data_dir, phase)
            
            with open(os.path.join(data_dir, f'indexed-{phase}.txt'), 'w') as file:
                for srcs, inps, trgs in inp_trg_dataset:
                    indexed_srcs = join_inds(src_dict.index_sentence(srcs))
                    indexed_inps = join_inds(trg_dict.index_sentence(inps))
                    indexed_trgs = join_inds(trg_dict.index_sentence(trgs))
                    file.write(f'{indexed_srcs}\t{indexed_inps}\t{indexed_trgs}\n')
    
    @staticmethod
    def preprocess(source_dictionary):

        def preprocess_function(source):
            source_tokens = source.strip().split()
            indexed_source = source_dictionary.index_sentence(source_tokens)
            return indexed_source

        return preprocess_function

if __name__ == '__main__':
    data = InputTargetTranslationDataset(
                data_dir='./data/test',
                phase='train')
    data[1]            
