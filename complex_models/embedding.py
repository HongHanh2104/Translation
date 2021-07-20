import torch 
import torch.nn as nn

import math
import numpy as np

class ComplexEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model):
        super(ComplexEmbedding, self).__init__()
        
        d_emb = d_model

        self.word_emb = nn.Embedding(vocab_size, d_emb)
        self.freq_emb = nn.Embedding(vocab_size, d_emb)
        self.init_phase_emb = nn.Embedding(vocab_size, d_emb)

    def get_embedding(self, x):
        b, length = x.shape

        amp = self.word_emb(x) # [b, seq_len, d_model]
        freq = self.freq_emb(x) # [b, seq_len, d_model]
        self.init_phase_emb.weight = nn.Parameter(self.init_phase_emb.weight % (2 * math.pi))
        
        pos = torch.arange(1, length + 1, 1.0, device=x.device) # [seq_len]
        pos = pos.unsqueeze(0).unsqueeze(-1) 
        pos = pos.repeat([b, 1, amp.shape[-1]]) # [b, seq_len, d_model]
        dim_bias = self.init_phase_emb(x) # [b, seq_len, d_model]
        out_phase = torch.mul(pos, freq) + dim_bias # [b, seq_len, d_model]
        out_real = amp * torch.cos(out_phase) # [b, seq_len, d_model]
        out_im = amp * torch.sin(out_phase) # [b, seq_len, d_model]
        
        return out_real, out_im

    def forward(self, x):
        return self.get_embedding(x)

if __name__ == '__main__':
    seq = torch.tensor(np.random.randint(
        0, 100, size=(1, 6)), requires_grad=False) #([225, 24, 95, 34, 26, 71]).unsqueeze(0)
    
    cplx_emb = ComplexEmbedding(100, 512)
    result = cplx_emb(seq)
    print(result[1].shape)