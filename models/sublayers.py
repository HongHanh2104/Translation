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
    def __init__(self, scale, attn_drop):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1) # apply the softmax along the last dimension
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        # q: [b, n_head, q_len, d_k]
        # k: [b, n_head, k_len, d_k]
        # v: [b, n_head, v_len, d_k]
        
        # Step 1: dot product q with k^T to compute similarity
        # k_tranpose: [b, n_head, d_k, k_len]
        # attn: [b, n_head, q_len, k_len]
        k_T = k.transpose(-2, -1)
        attn = (q @ k_T) * self.scale
        
        # Step 2: Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1) # [b, 1, k_len, k_len]
            attn = attn.masked_fill(mask == 0, 1e-12)
        
        # Step 3: Pass to softmax to make [0, 1] range
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Step 4: Multiply with v
        scores = attn @ v
        return scores

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
    def __init__(self, d_model, 
                 n_head,
                 attn_drop=0.,
                ):
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        
        # head dimension
        self.d_head = self.d_model // self.n_head
        self.scale = self.d_head ** (-0.5)
        
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attention = ScaledDotProductAttention(scale=self.scale, 
                                                   attn_drop=attn_drop)
        
    def forward(self, q, k, v, mask=None):
        # q, k, v: [b, seq_len, d_model]
        # mask: [b, query_len, key_len]
        b, _, _ = q.shape
        #  Step 1: dot product with weight matrices 
        q = self.w_q(q).reshape(b, -1, self.n_head, self.d_head) # [batch, q_len, n_head, d_head]
        k = self.w_k(k).reshape(b, -1, self.n_head, self.d_head) # [batch, k_len, n_head, d_head]
        v = self.w_v(v).reshape(b, -1, self.n_head, self.d_head) # [batch, v_len, n_head, d_head]

        # Step 2: split by number of heads
        q = q.permute(0, 2, 1, 3) # [batch, n_head, q_len, d_head]
        k = k.permute(0, 2, 1, 3) # [batch, n_head, k_len, d_head]
        v = v.permute(0, 2, 1, 3) # [batch, n_head, v_len, d_head]

        # Step 3: scale dot product
        scores = self.attention(q, k, v, mask=mask)
        
        # Step 4: concat and pass to linear layer
        scores = scores.transpose(1, 2).reshape(b, -1, self.n_head * self.d_head)
        scores = self.out_proj(scores)

        return scores # [b, seq_len, d_model]
    
    def split(self, tensor):
        """
        Split tensor by number of head

        @tensor: [b, length, d_model]
        @return: [b, head, length, d_tensor]

        """
        b, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.reshape(b, self.n_head, length, d_tensor)
        return tensor

class PositionwiseFeedForward(nn.Module):
    '''
    Fully connected feed-forward network
    This consists of 2 linear transformations with a ReLU activation in between. 
    '''
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: [b, seq_len, d_model]
        return self.feed_forward(x)

if __name__ == '__main__':
    # inputs = torch.randn(3, 5, 512).cpu()
    # ffn = PositionwiseFeedForward(512, 2048)
    # out = ffn(inputs).cpu()
    # print(out.size())

    q = torch.randn(1, 10, 512).cpu()
    k = torch.randn(1, 10, 512).cpu()
    v = torch.randn(1, 10, 512).cpu()
    multi_head_attn = MultiHeadAttention(d_model=512, n_head=8)
    out = multi_head_attn(q, k, v)
    print(out.shape)
