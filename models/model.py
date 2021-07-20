import torch
import torch.nn as nn
from torch.autograd import Variable

from models.layers import EncoderLayer, DecoderLayer
from models.embedding import TransformerEmbedding
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

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         d_ffn=d_ffn,
                         n_head=n_head,
                         dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, x, src_mask):
        # src_seq: [b, seq_len, embed_size]
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
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         d_ffn=d_ffn,
                         n_head=n_head,
                         dropout=dropout)
            for _ in range(n_layer)
        ])
        
    def forward(self, trg, enc_outs, dec_enc_attn_mask=None, slf_attn_mask=None):
        # trg: [b, seq_len - 1, d_model]
        # enc_outs: [b, seq_len, d_model]

        for layer in self.layer_stack:
            trg = layer(trg, enc_outs, dec_enc_attn_mask, slf_attn_mask)

        return trg


class Transformer(nn.Module):
    def __init__(self,
                 n_src_vocab,
                 n_trg_vocab,
                 src_pad_idx,
                 trg_pad_idx,
                 max_len=256,
                 d_model=512,
                 d_ffn=2048,
                 n_layer=6,
                 n_head=8,
                 dropout=0.1):
        super().__init__()
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.max_len = max_len
        self.d_model = d_model

        self.enc_pos_encoding = TransformerEmbedding(
                                vocab_size=n_src_vocab,
                                d_model=d_model,
                                max_len=max_len,
                                dropout=dropout
                                )

        self.dec_pos_encoding = TransformerEmbedding(
                                vocab_size=n_trg_vocab,
                                d_model=d_model,
                                max_len=max_len,
                                dropout=dropout
                                )

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
        
        self.proj_linear = nn.Linear(d_model, n_trg_vocab)

    def forward(self, src_seq, trg_seq):
        # src_seq: [batch, src_len]
        # trg_seq: [batch, trg_len]

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(
            trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        src_pos_encoded = self.enc_pos_encoding(src_seq)
        trg_pos_encoded = self.dec_pos_encoding(trg_seq)
        #print(trg_pos_encoded.shape)

        enc_outs = self.encoder(src_pos_encoded, src_mask)  # [batch, seq_len, d_model]
        # [b, seq_len, trg_vocab_size]
        dec_outs = self.decoder(trg_pos_encoded, enc_outs, src_mask, trg_mask)
        
        out = self.proj_linear(dec_outs)  # [b, seq_len, trg_vocab_size]
        
        return out

    def predict(self, src, max_length=256, eos_id=0):
        q = torch.zeros(src.size(0), 1).long().to(src.device)  # [B, 1]
        done = torch.zeros_like(q).bool()  # [B, 1]
        for _ in range(max_length):
            # q: [B, T]
            predict = self.forward(src, q)  # [B, T, V]
            predict = predict[:, -1].argmax(-1, keepdim=True)  # [B, 1]
            q = torch.cat([q, predict], dim=-1)  # [B, T+1]

            done = done | (predict == eos_id)
            if done.all():
                break
        return q[:, 1:]
