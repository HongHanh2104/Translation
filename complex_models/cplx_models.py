import torch
import torch.nn as nn

import numpy as np

def get_attn_key_pad_mask(seq_k, seq_q):

    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_src_vocab, d_hid, padding_idx=None):

    def cal_angle(hid_idx):
        return 1 / np.power(10000, 2 * (hid_idx//2) / d_hid)

    def get_posi_angle_vec():
        return [cal_angle(hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec() for _ in range(n_src_vocab)])


    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class Encoder(nn.Module):
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec):

        super().__init__()

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_src_vocab, d_word_vec, padding_idx=0),
            freeze=True)

    def forward(self, src_seq, src_pos, return_attns=False):

        
        enc_output_real_real = self.src_word_emb(src_seq)
        enc_output_phase = self.position_enc(src_seq)
        print(src_pos.shape)
        pos = torch.unsqueeze(torch.LongTensor(src_pos),2)
        
        # 10000^{2k/d_model} * pos ???
        enc_output_phase=torch.mul(pos.float(), enc_output_phase)

        cos = torch.cos(enc_output_phase)
        sin = torch.sin(enc_output_phase)

        enc_output_real=enc_output_real_real*cos
        enc_output_phase=enc_output_real_real*sin

        print(enc_output_real.shape)

        return enc_output_real,enc_output_phase

if __name__ == '__main__':
    seq = torch.tensor(np.random.randint(
        0, 100, size=(1, 6)), requires_grad=False) #([225, 24, 95, 34, 26, 71]).unsqueeze(0)
    pos = torch.tensor([1, 2, 3, 4, 5, 6]).unsqueeze(0)
    encoder = Encoder(100, 256, 512)
    result = encoder(seq, pos)
    #print(result[0])