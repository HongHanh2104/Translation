import torch
import torch.nn as nn
import math

class TransformerEmbedding(nn.Module):
    """
    Transformer embedding = Token embedding + Positional encoding (sinusoid) 

    @param:
    :vocab_size: size of vocabulary
    :max_len:  max setence length
    :d_model: dimension of model
    """

    def __init__(self,
                 vocab_size,
                 d_model,
                 max_len=5000,
                 padding_idx=1,
                 dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )
        # Positional encoding
        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1)
        pos = pos.float()

        div_term = torch.exp((torch.arange(0, d_model, 2) *
                              (-math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(pos * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(pos * div_term)  # dim 2i + 1
        pe = pe.unsqueeze(0)  # [b, max_len, d_model]

        # Resigter buffer in order to save the positional encodings inside state_dict
        # If not, these would be excluded from the state_dict becuase they are not learnable
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [b, seq_len]
        _, length = x.shape
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        #print(f'length: {length}')
        #print(x.shape)
        x = x + self.pe[:, :length]
        x = self.dropout(x)

        return x  # [b, seq_len, d_model]


if __name__ == '__main__':
    import numpy as np
    sequence = torch.tensor(np.random.randint(
        0, 100, size=(3, 5)), requires_grad=False)
    pe = TransformerEmbedding(100, 512)
    out = pe(sequence)
    print(out)
