import torch
import torch.nn as nn

from models.layers import EncoderLayer, DecoderLayer
from models.embedding import TransformerEmbedding, ComplexEmbedding
from models.mask import get_pad_mask, get_subsequent_mask

import numpy as np


class Encoder(nn.Module):
    def __init__(self,
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
    def __init__(self,
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
                 dropout=0.1,
                 emb_type='tf',
                 xavier_init=False,
                 share_emb_prj=False,
                 share_emb_emb=False):
        super().__init__()
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.max_len = max_len
        self.d_model = d_model

        if emb_type == 'tf':
            self.enc_emb = TransformerEmbedding(
                vocab_size=n_src_vocab,
                d_model=d_model,
                max_len=max_len,
                dropout=dropout
            )

            self.dec_emb = TransformerEmbedding(
                vocab_size=n_trg_vocab,
                d_model=d_model,
                max_len=max_len,
                dropout=dropout
            )
        elif emb_type == 'complex':
            self.enc_emb = ComplexEmbedding(
                vocab_size=n_src_vocab,
                d_model=d_model
            )

            self.dec_emb = ComplexEmbedding(
                vocab_size=n_trg_vocab,
                d_model=d_model
            )

        self.encoder = Encoder(d_model=d_model,
                               d_ffn=d_ffn,
                               n_layer=n_layer,
                               n_head=n_head,
                               dropout=dropout
                               )

        self.decoder = Decoder(d_model=d_model,
                               d_ffn=d_ffn,
                               n_layer=n_layer,
                               n_head=n_head,
                               dropout=dropout
                               )

        self.proj_linear = nn.Linear(d_model, n_trg_vocab)

        if xavier_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if share_emb_prj:
            self.proj_linear.weight = self.dec_emb.word_emb.weight

        if share_emb_emb:
            self.dec_emb.word_emb.weight = self.enc_emb.word_emb.weight

    def forward(self, src_seq, trg_seq):
        # src_seq: [batch, src_len]
        # trg_seq: [batch, trg_len]

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(
            trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        src_emb = self.enc_emb(src_seq)  # [b, src_len, d_model]
        trg_emb = self.dec_emb(trg_seq)  # [b, trg_len, d_model]

        # [batch, seq_len, d_model]
        enc_outs = self.encoder(src_emb, src_mask)
        # [b, seq_len, trg_vocab_size]
        dec_outs = self.decoder(trg_emb, enc_outs, src_mask, trg_mask)

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
