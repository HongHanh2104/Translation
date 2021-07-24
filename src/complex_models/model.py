import torch
import torch.nn as nn

from complex_models.layers import EncoderLayer, DecoderLayer
from complex_models.embedding import ComplexEmbedding
from complex_models.mask import get_pad_mask, get_subsequent_mask

import numpy as np


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 d_ffn,
                 n_layer,
                 n_head,
                 dropout=0.1, 
                 continue_complex=False):
        super().__init__()
        self.d_model = d_model

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         d_ffn=d_ffn,
                         n_head=n_head,
                         dropout=dropout,
                         continue_complex=continue_complex)
            for _ in range(n_layer)
        ])

    def forward(self, x_real, x_phase, src_mask):
        # src_seq: [b, seq_len, embed_size]
        for layer in self.layer_stack:
            x_real, x_phase = layer(x_real, x_phase, 
                                    src_mask)
        
        return x_real, x_phase


class Decoder(nn.Module):
    def __init__(self,
                 d_model,
                 d_ffn,
                 n_layer,
                 n_head,
                 dropout=0.1,
                 continue_complex=False):
        super().__init__()
        self.d_model = d_model

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         d_ffn=d_ffn,
                         n_head=n_head,
                         dropout=dropout, 
                         continue_complex=continue_complex)
            for _ in range(n_layer)
        ])

    def forward(self, x_real, x_phase, enc_out_real, 
                enc_put_phase, dec_enc_attn_mask=None, 
                slf_attn_mask=None):
        # trg: [b, seq_len - 1, d_model]
        # enc_outs: [b, seq_len, d_model]
        for layer in self.layer_stack:
            x_real, x_phase = layer(x_real, x_phase, enc_out_real, 
                        enc_put_phase, dec_enc_attn_mask, 
                        slf_attn_mask)

        return x_real, x_phase


class ComplexTransformer(nn.Module):
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
                 continue_complex=False,
                 xavier_init=True,
                 share_emb_prj=False):
        super().__init__()
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.max_len = max_len
        self.d_model = d_model

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
                               dropout=dropout,
                               continue_complex=continue_complex
                               )

        self.decoder = Decoder(d_model=d_model,
                               d_ffn=d_ffn,
                               n_layer=n_layer,
                               n_head=n_head,
                               dropout=dropout,
                               continue_complex=continue_complex
                               )

        self.proj_linear = nn.Linear(d_model, n_trg_vocab)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if share_emb_prj:
            self.proj_linear.weight = self.dec_emb.word_emb.weight

    def forward(self, src_seq, trg_seq):
        # src_seq: [batch, src_len]
        # trg_seq: [batch, trg_len]

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(
            trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        src_emb_real, src_emb_phase = self.enc_emb(src_seq)
        trg_emb_real, trg_emb_phase = self.dec_emb(trg_seq)
        # print(trg_pos_encoded.shape)

        # [batch, seq_len, d_model]
        enc_out_real, enc_out_phase = self.encoder(
                                        src_emb_real, src_emb_phase, 
                                        src_mask)
        # [b, seq_len, trg_vocab_size]
        dec_out_real, dec_out_phase = self.decoder(
                                        trg_emb_real, trg_emb_phase, 
                                        enc_out_real, enc_out_phase, 
                                        src_mask, trg_mask)
        
        dec_out = dec_out_real * dec_out_real + dec_out_phase * dec_out_phase

        out = self.proj_linear(dec_out)  # [b, seq_len, trg_vocab_size]

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
