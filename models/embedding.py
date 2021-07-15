import torch 
import torch.nn as nn 
import math
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding.  
    """
    def __init__(self, embedding_num, d_model, max_len=5000, padding_idx=1, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.embedding_num = embedding_num
        self.embedding = nn.Embedding(embedding_num, d_model, padding_idx=padding_idx)
        
        pe = torch.zeros(max_len, d_model)
        
        pos = torch.arange(0, max_len).unsqueeze(1)
        pos = pos.float()
        
        div_term = torch.exp((torch.arange(0, d_model, 2) * 
                              (-math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(pos * div_term) # dim 2i
        pe[:, 1::2] = torch.cos(pos * div_term) # dim 2i + 1
        pe = pe.unsqueeze(0) # [b, max_len, d_model]
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [b, seq_len]
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        
        return x # [b, seq_len, d_model]

if __name__ == '__main__':
    import numpy as np
    sequence = torch.tensor(np.random.randint(0, 100, size=(3, 5)), requires_grad=False)
    pe = PositionalEncoding(100, 512)
    out = pe(sequence)
    print(out)
