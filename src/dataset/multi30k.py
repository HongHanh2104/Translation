import torch
from torch.utils.data import Dataset
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k

from utils.tokenize import Tokenizer

class Multi30kLoader():
    source: Field = None
    target: Field = None

    def __init__(self, ext):
        super().__init__()

        self.ext = ext
        self.init_token = '<sos>'
        self.eos_token = '<eos>'
        
        # Tokenize
        self.tokenizer = Tokenizer()
        self.source, self.target = self._make_field()

        
    def _make_field(self):
        if self.ext == ('.de', '.en'):
            source = Field(tokenize=self.tokenizer.tokenize_de, 
                           init_token=self.init_token, 
                           eos_token=self.eos_token,
                           lower=True, 
                           batch_first=True
                        )

            target = Field(tokenize=self.tokenizer.tokenize_en, 
                           init_token=self.init_token, 
                           eos_token=self.eos_token,
                           lower=True, 
                           batch_first=True
                        )
        elif self.ext == ('.en', '.de'):
            source = Field(tokenize=self.tokenizer.tokenize_en, 
                           init_token=self.init_token, 
                           eos_token=self.eos_token,
                           lower=True, 
                           batch_first=True
                        )

            target = Field(tokenize=self.tokenizer.tokenize_de, 
                           init_token=self.init_token, 
                           eos_token=self.eos_token,
                           lower=True, 
                           batch_first=True
                        )
        return source, target
    
    def create_dataset(self):
        self.train_data, self.val_data, self.test_data = Multi30k.splits(
                                                                exts=self.ext, 
                                                                fields=(self.source, self.target)
                                                                )
        return self.train_data, self.val_data, self.test_data
    
    def build_vocab(self, data, min_freq):
        self.source.build_vocab(data, min_freq=min_freq)
        self.target.build_vocab(data, min_freq=min_freq)

    def make_iter(self, batch_size, device):
        train_iter, valid_iter, test_iter = BucketIterator.splits(
                                                    (self.train_data, self.val_data, self.test_data),
                                                     batch_size=batch_size,
                                                     device=device,
                                                     shuffle=True,
                                            )
        return train_iter, valid_iter, test_iter

    def get_pad_idx(self):
        src_pad_idx = self.source.vocab.stoi['<pad>']
        trg_pad_idx = self.target.vocab.stoi['<pad>']
        return src_pad_idx, trg_pad_idx
    
    def get_voc_size(self):
        enc_voc_size = len(self.source.vocab)
        dec_voc_size = len(self.target.vocab)
        return enc_voc_size, dec_voc_size

