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
        self.word_emb = nn.Embedding(
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

        x = self.word_emb(x) * math.sqrt(self.d_model)

        x = x + self.pe[:, :length]
        x = self.dropout(x)

        return x  # [b, seq_len, d_model]


class ComplexEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model):
        super(ComplexEmbedding, self).__init__()

        d_emb = d_model // 2

        self.word_emb = nn.Embedding(vocab_size, d_emb)
        self.freq_emb = nn.Embedding(vocab_size, d_emb)
        self.init_phase_emb = nn.Embedding(vocab_size, d_emb)

    def get_embedding(self, x):
        b, length = x.shape

        amp = self.word_emb(x)  # [b, seq_len, d_emb]
        freq = self.freq_emb(x)  # [b, seq_len, d_emb]
        self.init_phase_emb.weight = nn.Parameter(
            self.init_phase_emb.weight % (2 * math.pi))

        pos = torch.arange(1, length + 1, 1.0, device=x.device)  # [seq_len]
        pos = pos.unsqueeze(0).unsqueeze(-1)
        pos = pos.repeat([b, 1, amp.shape[-1]])  # [b, seq_len, d_emb]
        dim_bias = self.init_phase_emb(x)  # [b, seq_len, d_emb]
        out_phase = torch.mul(pos, freq) + dim_bias  # [b, seq_len, d_emb]
        out_real = amp * torch.cos(out_phase)  # [b, seq_len, d_emb]
        out_im = amp * torch.sin(out_phase)  # [b, seq_len, d_emb]

        return torch.cat([out_real, out_im], -1)  # [b, seq_len, d_model]

    def forward(self, x):
        return self.get_embedding(x)


if __name__ == '__main__':
    import numpy as np
    sequence = torch.tensor(np.random.randint(
        0, 100, size=(3, 5)), requires_grad=False)
    pe = TransformerEmbedding(100, 512)
    out = pe(sequence)
    print(out)
