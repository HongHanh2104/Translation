import torch 
import torch.nn as nn

import math
import numpy as np

def get_sinusoid_encoding_table(n_vocab, d_model, padding_idx=None):
    
    def cal_angle(hid_idx):
        return 1 / np.power(10000, 2 * (hid_idx//2) / d_model)

    def get_posi_angle_vec():
        return [cal_angle(hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec() for _ in range(n_vocab)])


    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class ComplexEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 max_len=256,
                 padding_idx=1,
                 dropout=0.1):
        super(ComplexEmbedding, self).__init__()
        
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(
                    vocab_size,
                    d_model,
                    padding_idx=padding_idx
                    )

        # Positional encoding
        self.pos = torch.arange(0, max_len).unsqueeze(0).float()

        mul_term = 1.0 / torch.pow(10000.0, 
                        torch.arange(0, d_model // 2, 2) / d_model).float()
        
        encoding_table = torch.zeros(max_len, d_model)
        encoding_table[:, 0::2] = mul_term  # dim 2i
        encoding_table[:, 1::2] = mul_term  # dim 2i + 1
        #encoding_table = encoding_table.unsqueeze(0)  # [b, max_len, d_model]
        
        self.pos_enc = nn.Embedding.from_pretrained(
                        get_sinusoid_encoding_table(
                                vocab_size, 
                                d_model, 
                                padding_idx
                                ),
                        freeze=True
                        )
        
    def forward(self, x):
        # x: [b, seq_len]
        _, length = x.shape
        
        emb = self.embedding(x) # [b, seq_len, d_model]
    
        # self.pos: [b, seq_len, 1]
        # self.pos_enc: [b, seq_len, d_model]
        phase = torch.mul(self.pos[:, :length].unsqueeze(2),
                          self.pos_enc(x)) # [b, seq_len, d_model]
        
        out_real = emb * torch.cos(phase)
        out_phase = emb * torch.sin(phase)
        print(out_phase)

        # print(out_phase)

        # return out_real, out_phase  # [b, seq_len, d_model]
        return 0

if __name__ == '__main__':
    # seq = torch.tensor(np.random.randint(
    #     0, 100, size=(1, 6)), requires_grad=False) #([225, 24, 95, 34, 26, 71]).unsqueeze(0)
    # pos = torch.tensor([1, 2, 3, 4, 5, 6]).unsqueeze(0)
    
    # cplx_emb = ComplexEmbedding(100, 512)
    # result = cplx_emb(seq)

    out = get_sinusoid_encoding_table(100, 512)
    print(out.shape)