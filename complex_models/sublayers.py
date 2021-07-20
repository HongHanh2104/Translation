import torch
from torch import nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot product attention

    # Params: 
    :scale (float): math.sqrt(d_model)
    :attn_drop (float): for dropout

    # Inputs: 
    :q (Query) [batch, q_len, d_model]: 
           given sentence that we focused on (decoder)
    :k (Key) [batch, k_len, d_model]: 
           every sentence to check relationship with Query (encoder)
    :v (Value) [batch, v_len, d_model]: 
           every sentence same with Key (encoder)
    :mask: matrix containing indices to be masked

    """

    def __init__(self, scale, attn_drop, continue_complex):
        super().__init__()
        # apply the softmax along the last dimension
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = scale
        self.continue_complex = continue_complex

    def forward(self, x_real, x_phase, mask=None):
        # q: [b, n_head, q_len, d_k]
        # k: [b, n_head, k_len, d_k]
        # v: [b, n_head, v_len, d_v]

        q_real, k_real, v_real = x_real
        q_phase, k_phase, v_phase = x_phase

        attn_real = (q_real @ k_real.transpose(2, 3)) \
                - (q_phase @ k_phase.transpose(2, 3)) # [b, n_head, q_len, k_len]
        
        attn_phase = (q_real @ k_phase.transpose(2, 3)) \
                + (q_phase @ k_real.transpose(2, 3)) # [b, n_head, q_len, k_len]
        
        if self.continue_complex:
            attn_real = attn_real / self.scale
            attn_phase = attn_phase / self.scale

            if mask is not None:
                mask = mask.unsqueeze(1)
                attn_real = attn_real.masked_fill(mask == 0, -1e9)
                attn_phase = attn_real.masked_fill(mask == 0, -1e9)

            attn_real = self.attn_drop(self.softmax(attn_real))
            attn_phase = self.attn_drop(self.softmax(attn_phase))

            x_real = (attn_real @ v_real) - (attn_phase @ v_phase)
            x_phase = (attn_real @ v_phase) + (attn_phase @ v_real)
        else:
            attn = attn_real * attn_real + attn_phase * attn_phase
            attn = torch.sqrt(attn)
            attn /= self.scale

            if mask is not None:
                mask = mask.unsqueeze(1)
                attn = attn.masked_fill(mask == 0, -1e9)
            
            # paper k dropout ????
            attn = self.softmax(attn)

            x_real = attn @ v_real # [b, n_head, seq_len, d_v]
            x_phase = attn @ v_phase # [b, n_head, seq_len, d_v]

        return x_real, x_phase, attn_real


class MultiHeadAttention(nn.Module):
    '''
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
    where head_i = Attention(Q * W_q, K * W_k, V * W_v)

    # Params:
    :d_model (default: 512): dimension of model which is also of queries, keys, values
    :n_head (default: 8): number of heads.

    # Inputs:
    :q (Query) [batch, q_len, d_model]
    :k (Key) [batch, k_len, d_model]
    :v (Value) [batch, v_len, d_model]
    :mask: matrix containing indices to be masked

    # Outputs:
    :scores [batch_ out_len, d_model]

    '''

    def __init__(self,
                 d_model=512,
                 d_k=64, d_v=64,
                 n_head=8,
                 attn_drop=0.,
                 continue_complex=False,
                 ):
        super().__init__()

        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model

        # head dimension
        self.d_k = self.d_model // self.n_head
        self.scale = (self.d_k * 2) ** (0.5)
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)

        self.out_proj = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(scale=self.scale,
                                                   attn_drop=attn_drop,
                                                   continue_complex=continue_complex)

    def forward(self, x_real, x_phase, mask=None):
        # x_real is a list of 3 elements: q_real, k_real, v_real
        # x_phase is a list of 3 elements: q_phase, k_phase, v_phase
        
        # q, k, v: [b, seq_len, d_model]
        # mask: [b, query_len, key_len]

        q_real, k_real, v_real = x_real
        q_phase, k_phase, v_phase = x_phase

        b = q_real.shape[0]

        # q, k, v_{real/ phase}: [b, n_head, seq_len, d_k or d_v]
        q_real = self.w_q(q_real).reshape(b, -1, self.n_head, self.d_k).transpose(1, 2)
        k_real = self.w_k(k_real).reshape(b, -1, self.n_head, self.d_k).transpose(1, 2)
        v_real = self.w_v(v_real).reshape(b, -1, self.n_head, self.d_v).transpose(1, 2)

        q_phase = self.w_q(q_phase).reshape(b, -1, self.n_head, self.d_k).transpose(1, 2)
        k_phase = self.w_k(k_phase).reshape(b, -1, self.n_head, self.d_k).transpose(1, 2)
        v_phase = self.w_v(v_phase).reshape(b, -1, self.n_head, self.d_v).transpose(1, 2)

        x_real, x_phase, _ = self.attention(
                                [q_real, k_real, v_real], 
                                [q_phase, k_phase, v_phase], 
                                mask=mask
                                ) # [b, n_head, seq_len, d_v]

        x_real = x_real.transpose(1, 2).reshape(b, -1, self.n_head * self.d_v)
        x_phase = x_phase.transpose(1, 2).reshape(b, -1, self.n_head * self.d_v)

        x_real = self.out_proj(x_real) # [b, seq_len, d_model]
        x_phase = self.out_proj(x_phase) # [b, seq_len, d_model]

        return x_real, x_phase 


class PositionwiseFeedForward(nn.Module):
    '''
    Fully connected feed-forward network
    This consists of 2 linear transformations with a ReLU activation in between. 
    '''

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        
        self.linear_1 = nn.Linear(d_model, d_ffn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ffn, d_model)

    def forward(self, x_real, x_phase):
        # x: [b, seq_len, d_model]
        w_real = self.relu(self.linear_1(x_real) - \
                            self.linear_1(x_phase)) # [b, seq_len, d_ffn]
        w_phase = self.relu(self.linear_1(x_phase) + \
                             self.linear_1(x_real)) # [b, seq_len, d_ffn]
        
        w_real = self.dropout(w_real)
        w_phase = self.dropout(w_phase)
        
        x_real = self.linear_2(w_real) - self.linear_2(w_phase) # [b, seq_len, d_model]
        x_phase = self.linear_2(w_phase) + self.linear_2(w_real) # [b, seq_len, d_model]

        return x_real, x_phase

if __name__ == '__main__':
    # x_real = torch.randn(3, 10, 512)
    # x_phase = torch.randn(3, 10, 512)
    # ffn = PositionwiseFeedForward(512, 2048)
    # out = ffn(x_real, x_phase)
    # print(out[0].size())

    q_real = torch.randn(1, 10, 512)
    k_real = torch.randn(1, 10, 512)
    v_real = torch.randn(1, 10, 512)
    q_phase = torch.randn(1, 10, 512)
    k_phase = torch.randn(1, 10, 512)
    v_phase = torch.randn(1, 10, 512)

    x_real = [q_real, k_real, v_real]
    x_phase = [q_phase, k_phase, v_phase]
    
    # multi_head_attn = MultiHeadAttention(d_model=512, n_head=8, continue_complex=False)
    # out = multi_head_attn([q_real, k_real, v_real], 
    #                       [q_phase, k_phase, v_phase])
    # print(out[0].shape)
