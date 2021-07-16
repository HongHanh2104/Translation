import torch 
import torch.nn as nn 
from torch.autograd import Variable

from models.layers import EncoderLayer, DecoderLayer
from models.embedding import PositionalEncoding
from models.mask import get_pad_mask, get_subsequent_mask

import numpy as np

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, 
                 max_len, 
                 d_model,
                 d_ffn, 
                 n_layer, 
                 n_head,  
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(
                            vocab_size=n_src_vocab, 
                            d_model=d_model, 
                            max_len=max_len,
                            dropout=dropout
                        )
        self.layer_stack = nn.ModuleList([
                    EncoderLayer(d_model=d_model, 
                                d_ffn=d_ffn, 
                                n_head=n_head, 
                                dropout=dropout)
                    for _ in range(n_layer)
                ])
        
    def forward(self, x, src_mask):
        # src_seq: [b, seq_len, embed_size]
        x = self.pos_encoding(x)
        for layer in self.layer_stack:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, n_dec_vocab, 
                 max_len, 
                 d_model,
                 d_ffn, 
                 n_layer, 
                 n_head,  
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(
                            vocab_size=n_dec_vocab, 
                            d_model=d_model, 
                            max_len=max_len,
                            dropout=dropout
                            )
        self.layer_stack = nn.ModuleList([
                    DecoderLayer(d_model=d_model, 
                                d_ffn=d_ffn, 
                                n_head=n_head, 
                                dropout=dropout)
                    for _ in range(n_layer)
                    ])
        self.linear = nn.Linear(d_model, n_dec_vocab)

    def forward(self, trg, enc_outs, memory_mask, mask_2):
        # trg: [b, seq_len - 1, d_model]
        # enc_outs: [b, seq_len, d_model]
        trg = self.pos_encoding(trg)
        
        for layer in self.layer_stack:
            trg = layer(trg, enc_outs, memory_mask, mask_2)
        
        out = self.linear(trg) # [b, seq_len, trg_vocab_size]
        return out

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, 
                 n_trg_vocab, 
                 src_pad_idx, 
                 trg_pad_idx,
                 max_len=5000, 
                 d_model=512,
                 d_ffn=2048, 
                 n_layer=6, 
                 n_head=8, 
                 dropout=0.1):
        super().__init__()
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        
        self.encoder = Encoder(n_src_vocab=n_src_vocab, 
                               max_len=max_len,
                               d_model=d_model,
                               d_ffn=d_ffn, 
                               n_layer=n_layer, 
                               n_head=n_head, 
                               dropout=dropout
                            )
        
        self.decoder = Decoder(n_dec_vocab=n_trg_vocab, 
                               max_len=max_len,
                               d_model=d_model,
                               d_ffn=d_ffn, 
                               n_layer=n_layer, 
                               n_head=n_head, 
                               dropout=dropout
                            )
        
    def forward(self, src_seq, trg_seq):
        # src_seq: [batch, src_len]
        # trg_seq: [batch, trg_len]
        mask_1 = get_pad_mask(src_seq, self.src_pad_idx) # [b, src_seq, src_seq]
        mask_2 = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        
        enc_outs = self.encoder(src_seq, mask_1) # [batch, seq_len, d_model]
        dec_outs = self.decoder(trg_seq, enc_outs, mask_1, mask_2) # [b, seq_len, trg_vocab_size]
        return dec_outs


