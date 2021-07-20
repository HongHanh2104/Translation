import torch
import torch.nn as nn
from complex_models.sublayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model,
                 d_ffn,
                 n_head,
                 dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            attn_drop=dropout
        )
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_1 = nn.Dropout(dropout)

        self.pos_ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout
        )
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [b, seq_len, d_model]
        residual_1 = x
        x = self.self_attn(x, x, x, mask)
        x = self.norm_1(residual_1 + self.dropout_1(x))
        # x = self.dropout_1(x)

        residual_2 = x
        x = self.pos_ffn(x)
        x = self.norm_2(residual_2 + self.dropout_2(x))
        # x = self.dropout_2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model,
                 d_ffn,
                 n_head,
                 dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            attn_drop=dropout
        )
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_1 = nn.Dropout(dropout)

        self.enc_dec_attn = MultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            attn_drop=dropout
        )
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_2 = nn.Dropout(dropout)

        self.pos_ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout
        )
        self.norm_3 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, ecd_outs, dec_enc_attn_mask=None, slf_attn_mask=None):
        residual_1 = x
        x = self.masked_self_attn(x, x, x, slf_attn_mask)
        x = self.norm_1(residual_1 + self.dropout_1(x))
        # x = self.dropout_1(x)

        residual_2 = x
        x = self.enc_dec_attn(x, ecd_outs, ecd_outs, dec_enc_attn_mask)
        x = self.norm_2(residual_2 + self.dropout_2(x))
        # x = self.dropout_2(x)

        residual_3 = x
        x = self.pos_ffn(x)
        x = self.norm_3(residual_3 + self.dropout_3(x))
        # x = self.dropout_3(x)

        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 4).cpu()
    encoderlayer = EncoderLayer(d_model=4, d_ffn=4, n_head=2)
    ecd_out = encoderlayer(x).cpu()
    print(ecd_out.size())

    y = torch.randn(1, 3, 4).cpu()
    decoderlayer = DecoderLayer(d_model=4, d_ffn=4, n_head=2)
    dcd_out = decoderlayer(y, ecd_out).cpu()
    print(dcd_out.size())
